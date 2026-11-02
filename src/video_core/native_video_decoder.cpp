// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstring>

#include "common/logging.h"
#include "video_core/native_video_decoder.h"

extern "C" {
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#endif
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
}

#include "gav1/decoder.h"

namespace VideoCommon {
namespace {

AVCodecID ToCodecId(NativeVideoCodec codec) {
    switch (codec) {
    case NativeVideoCodec::H264:
        return AV_CODEC_ID_H264;
    case NativeVideoCodec::Vp8:
        return AV_CODEC_ID_VP8;
    case NativeVideoCodec::Vp9:
        return AV_CODEC_ID_VP9;
    case NativeVideoCodec::Av1:
        return AV_CODEC_ID_AV1;
    }
    return AV_CODEC_ID_NONE;
}

class FFmpegNativeVideoDecoder final : public NativeVideoDecoder {
public:
    ~FFmpegNativeVideoDecoder() override {
        sws_freeContext(scaler);
        av_frame_free(&converted);
        av_frame_free(&frame);
        av_packet_free(&packet);
        avcodec_free_context(&context);
    }

    explicit FFmpegNativeVideoDecoder(NativeVideoOutput output_) : output_format{output_} {}

    bool Open(NativeVideoCodec codec, std::span<const u8> extradata) {
        const AVCodec* decoder = avcodec_find_decoder(ToCodecId(codec));
        if (!decoder) {
            LOG_ERROR(HW_GPU, "No FFmpeg decoder for native video codec {}", static_cast<u32>(codec));
            return false;
        }
        context = avcodec_alloc_context3(decoder);
        packet = av_packet_alloc();
        frame = av_frame_alloc();
        converted = av_frame_alloc();
        if (!context || !packet || !frame || !converted) {
            return false;
        }
        if (!extradata.empty()) {
            context->extradata = static_cast<u8*>(
                av_mallocz(extradata.size() + AV_INPUT_BUFFER_PADDING_SIZE));
            if (!context->extradata) {
                return false;
            }
            std::memcpy(context->extradata, extradata.data(), extradata.size());
            context->extradata_size = static_cast<int>(extradata.size());
        }
        context->thread_count = 0;
        // Frame threading holds back one picture per thread (up to 16), while the console's
        // decoder returns each picture once it is decoded; clients that wait for the first
        // pictures before submitting more would stall.
        context->thread_type = FF_THREAD_SLICE;
        const int result = avcodec_open2(context, decoder, nullptr);
        if (result < 0) {
            LOG_ERROR(HW_GPU, "avcodec_open2 failed for native video codec {}: {}",
                      static_cast<u32>(codec), result);
            return false;
        }
        return true;
    }

    bool Decode(std::span<const u8> data, s64 pts, std::vector<DecodedVideoFrame>& frames) override {
        if (av_new_packet(packet, static_cast<int>(data.size())) < 0) {
            return false;
        }
        std::memcpy(packet->data, data.data(), data.size());
        packet->pts = pts;
        const int sent = avcodec_send_packet(context, packet);
        av_packet_unref(packet);
        if (sent < 0 && sent != AVERROR(EAGAIN)) {
            LOG_WARNING(HW_GPU, "Native video packet of {} bytes was rejected: {}", data.size(), sent);
            return false;
        }
        Receive(frames);
        return true;
    }

    void Drain(std::vector<DecodedVideoFrame>& frames) override {
        avcodec_send_packet(context, nullptr);
        Receive(frames);
        avcodec_flush_buffers(context);
    }

    void Flush() override {
        avcodec_flush_buffers(context);
    }

private:
    void Receive(std::vector<DecodedVideoFrame>& frames) {
        while (avcodec_receive_frame(context, frame) == 0) {
            DecodedVideoFrame output;
            if (Convert(output)) {
                frames.push_back(std::move(output));
            }
            av_frame_unref(frame);
        }
    }

    bool Convert(DecodedVideoFrame& output) {
        const int width = frame->width;
        const int height = frame->height;
        output.width = static_cast<u32>(width);
        output.height = static_cast<u32>(height);
        output.pts = frame->best_effort_timestamp != AV_NOPTS_VALUE ? frame->best_effort_timestamp
                                                                   : frame->pts;
        const bool rgba = output_format == NativeVideoOutput::Rgba;
        const AVPixelFormat target = rgba ? AV_PIX_FMT_RGBA : AV_PIX_FMT_NV12;
        const AVFrame* source = frame;
        if (frame->format != target) {
            scaler = sws_getCachedContext(scaler, width, height,
                                          static_cast<AVPixelFormat>(frame->format), width, height,
                                          target, SWS_BILINEAR, nullptr, nullptr, nullptr);
            if (!scaler) {
                LOG_ERROR(HW_GPU, "No conversion from pixel format {} to {}", frame->format,
                          static_cast<int>(target));
                return false;
            }
            if (rgba) {
                // Follow the stream's matrix and range; unspecified streams are BT.709 video
                const int space = frame->colorspace == AVCOL_SPC_UNSPECIFIED ? AVCOL_SPC_BT709
                                                                             : frame->colorspace;
                sws_setColorspaceDetails(scaler, sws_getCoefficients(space),
                                         frame->color_range == AVCOL_RANGE_JPEG,
                                         sws_getCoefficients(AVCOL_SPC_BT709), 1, 0, 1 << 16,
                                         1 << 16);
            }
            if (converted->width != width || converted->height != height) {
                av_frame_unref(converted);
                converted->format = target;
                converted->width = width;
                converted->height = height;
                if (av_frame_get_buffer(converted, 0) < 0) {
                    return false;
                }
            }
            sws_scale(scaler, frame->data, frame->linesize, 0, height, converted->data,
                      converted->linesize);
            source = converted;
        }
        const auto copy_plane = [](std::vector<u8>& plane, const u8* data, int pitch, int row,
                                   int rows) {
            plane.resize(static_cast<size_t>(row) * rows);
            for (int y = 0; y < rows; ++y) {
                std::memcpy(plane.data() + static_cast<size_t>(row) * y, data + pitch * y, row);
            }
        };
        if (rgba) {
            copy_plane(output.luma, source->data[0], source->linesize[0], width * 4, height);
            output.chroma.clear();
            return true;
        }
        copy_plane(output.luma, source->data[0], source->linesize[0], width, height);
        copy_plane(output.chroma, source->data[1], source->linesize[1], (width + 1) & ~1,
                   (height + 1) / 2);
        return true;
    }

    AVCodecContext* context{};
    AVPacket* packet{};
    AVFrame* frame{};
    AVFrame* converted{};
    SwsContext* scaler{};
    NativeVideoOutput output_format;
};

/// AV1 through libgav1: the bundled FFmpeg of some platforms has no AV1 decoder, and Fire Emblem:
/// Fortune's Weave aborts when its movies cannot be decoded.
class Gav1NativeVideoDecoder final : public NativeVideoDecoder {
public:
    explicit Gav1NativeVideoDecoder(NativeVideoOutput output_) : output_format{output_} {}

    ~Gav1NativeVideoDecoder() override {
        sws_freeContext(scaler);
        av_frame_free(&converted);
    }

    bool Open() {
        libgav1::DecoderSettings settings;
        settings.threads = 4;
        converted = av_frame_alloc();
        return converted && decoder.Init(&settings) == libgav1::kStatusOk;
    }

    bool Decode(std::span<const u8> data, s64 pts, std::vector<DecodedVideoFrame>& frames) override {
        const auto enqueued = decoder.EnqueueFrame(data.data(), data.size(), pts, nullptr);
        if (enqueued != libgav1::kStatusOk) {
            LOG_WARNING(HW_GPU, "AV1 packet of {} bytes was rejected: {}", data.size(),
                        libgav1::GetErrorString(enqueued));
            return false;
        }
        return Dequeue(frames) != libgav1::kStatusNothingToDequeue;
    }

    void Drain(std::vector<DecodedVideoFrame>& frames) override {
        decoder.SignalEOS();
        while (Dequeue(frames) == libgav1::kStatusOk && has_output) {
        }
    }

    void Flush() override {
        decoder.SignalEOS();
    }

private:
    libgav1::StatusCode Dequeue(std::vector<DecodedVideoFrame>& frames) {
        const libgav1::DecoderBuffer* buffer{};
        const auto status = decoder.DequeueFrame(&buffer);
        has_output = status == libgav1::kStatusOk && buffer != nullptr;
        if (has_output) {
            DecodedVideoFrame output;
            if (Convert(*buffer, output)) {
                frames.push_back(std::move(output));
            }
        }
        return status;
    }

    bool Convert(const libgav1::DecoderBuffer& buffer, DecodedVideoFrame& output) {
        if (buffer.image_format != kLibgav1ImageFormatYuv420 ||
            (buffer.bitdepth != 8 && buffer.bitdepth != 10)) {
            LOG_ERROR(HW_GPU, "Unsupported AV1 picture: format {}, {} bits",
                      static_cast<int>(buffer.image_format), buffer.bitdepth);
            return false;
        }
        const int width = buffer.displayed_width[0];
        const int height = buffer.displayed_height[0];
        output.width = static_cast<u32>(width);
        output.height = static_cast<u32>(height);
        output.pts = buffer.user_private_data;
        const bool rgba = output_format == NativeVideoOutput::Rgba;
        const AVPixelFormat target = rgba ? AV_PIX_FMT_RGBA : AV_PIX_FMT_NV12;
        const AVPixelFormat source =
            buffer.bitdepth == 10 ? AV_PIX_FMT_YUV420P10LE : AV_PIX_FMT_YUV420P;
        scaler = sws_getCachedContext(scaler, width, height, source, width, height, target,
                                      SWS_BILINEAR, nullptr, nullptr, nullptr);
        if (!scaler) {
            return false;
        }
        // Follow the stream's matrix and range; unspecified streams are BT.709 video. The AV1
        // matrix coefficients use the same numbers as FFmpeg's color spaces.
        const int space = buffer.matrix_coefficients == kLibgav1MatrixCoefficientsUnspecified
                              ? AVCOL_SPC_BT709
                              : static_cast<int>(buffer.matrix_coefficients);
        const int full_range = buffer.color_range == kLibgav1ColorRangeFull ? 1 : 0;
        sws_setColorspaceDetails(scaler, sws_getCoefficients(space), full_range,
                                 sws_getCoefficients(AVCOL_SPC_BT709), rgba ? 1 : full_range, 0,
                                 1 << 16, 1 << 16);
        if (converted->width != width || converted->height != height) {
            av_frame_unref(converted);
            converted->format = target;
            converted->width = width;
            converted->height = height;
            if (av_frame_get_buffer(converted, 0) < 0) {
                return false;
            }
        }
        const std::array<const u8*, 4> planes{buffer.plane[0], buffer.plane[1], buffer.plane[2],
                                              nullptr};
        const std::array<int, 4> strides{buffer.stride[0], buffer.stride[1], buffer.stride[2], 0};
        sws_scale(scaler, planes.data(), strides.data(), 0, height, converted->data,
                  converted->linesize);
        const auto copy_plane = [](std::vector<u8>& plane, const u8* data, int pitch, int row,
                                   int rows) {
            plane.resize(static_cast<size_t>(row) * rows);
            for (int y = 0; y < rows; ++y) {
                std::memcpy(plane.data() + static_cast<size_t>(row) * y, data + pitch * y, row);
            }
        };
        if (rgba) {
            copy_plane(output.luma, converted->data[0], converted->linesize[0], width * 4, height);
            output.chroma.clear();
            return true;
        }
        copy_plane(output.luma, converted->data[0], converted->linesize[0], width, height);
        copy_plane(output.chroma, converted->data[1], converted->linesize[1], (width + 1) & ~1,
                   (height + 1) / 2);
        return true;
    }

    libgav1::Decoder decoder;
    AVFrame* converted{};
    SwsContext* scaler{};
    NativeVideoOutput output_format;
    bool has_output{};
};

} // Anonymous namespace

std::unique_ptr<NativeVideoDecoder> NativeVideoDecoder::Create(NativeVideoCodec codec,
                                                               std::span<const u8> extradata,
                                                               NativeVideoOutput output) {
    if (codec == NativeVideoCodec::Av1) {
        auto decoder = std::make_unique<Gav1NativeVideoDecoder>(output);
        return decoder->Open() ? std::move(decoder) : nullptr;
    }
    auto decoder = std::make_unique<FFmpegNativeVideoDecoder>(output);
    if (!decoder->Open(codec, extradata)) {
        return nullptr;
    }
    return decoder;
}

} // namespace VideoCommon
