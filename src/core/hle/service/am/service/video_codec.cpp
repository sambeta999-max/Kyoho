// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <optional>
#include <chrono>
#include <thread>
#include <cstring>
#include <deque>
#include <span>

#include "common/alignment.h"
#include "common/logging.h"
#include "core/core.h"
#include "core/hle/kernel/k_client_session.h"
#include "core/hle/kernel/k_event.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/k_readable_event.h"
#include "core/hle/kernel/svc_results.h"
#include "core/hle/service/am/service/graphics_session.h"
#include "core/hle/service/am/service/video_codec.h"
#include "core/hle/service/ipc_helpers.h"
#include "core/hle/service/kernel_helpers.h"
#include "core/hle/service/nvdrv/core/container.h"
#include "core/hle/service/nvdrv/core/syncpoint_manager.h"
#include "core/hle/service/nvdrv/nvdrv.h"
#include "core/hle/service/service.h"
#include "core/memory.h"
#include "video_core/host1x/host1x.h"
#include "video_core/native_video_decoder.h"
#include "video_core/textures/decoders.h"

namespace Service::AM {

std::optional<VideoImageFormat> VideoImageFormatFromPreference(u32 type) {
    switch (type) {
    case 2:
    case 5:
        return VideoImageFormat::Rgba;
    case 1:
    case 4:
        return VideoImageFormat::Nv12;
    default:
        return std::nullopt;
    }
}

u32 VideoImageBlockHeight(u32 height) {
    u32 log2 = 0;
    while (log2 < 4 && (8U << log2) < height) {
        ++log2;
    }
    return log2;
}

u64 VideoImageSize(u32 width, u32 height) {
    const u64 pitch = Common::AlignUp(u64{width} * 4, 64);
    return Common::AlignUp(pitch * Common::AlignUp(u64{height}, 128), 0x10000);
}

VideoImageLayout MakeVideoImageLayout(VideoImageFormat format, u32 width, u32 height, u32 index) {
    const u64 base = VideoImageSize(width, height) * index;
    if (format == VideoImageFormat::Rgba) {
        return VideoImageLayout{
            .luma_offset = base,
            .luma_pitch = Common::AlignUp(u64{width} * 4, 64),
        };
    }
    const u64 pitch = Common::AlignUp(u64{width}, 64);
    return VideoImageLayout{
        .luma_offset = base,
        .luma_pitch = pitch,
        .chroma_offset = base + pitch * Common::AlignUp(u64{height}, 128),
        .chroma_pitch = pitch,
    };
}

void VideoImagePoolRegistry::Add(const Pool& pool) {
    const std::scoped_lock lock{mutex};
    pools.push_back(pool);
}

bool VideoImagePoolRegistry::Remove(u32 handle) {
    const std::scoped_lock lock{mutex};
    return std::erase_if(pools, [handle](const Pool& pool) { return pool.handle == handle; }) != 0;
}

std::optional<VideoImagePoolRegistry::Image> VideoImagePoolRegistry::Find(u32 image_handle) const {
    const std::scoped_lock lock{mutex};
    for (const Pool& pool : pools) {
        if (image_handle >= pool.handle && image_handle - pool.handle < pool.count) {
            const u32 index = image_handle - pool.handle;
            return Image{pool.format, pool.width, pool.height, pool.address,
                         MakeVideoImageLayout(pool.format, pool.width, pool.height, index),
                         pool.block_linear, pool.memory_id};
        }
    }
    return std::nullopt;
}

bool VideoImagePoolRegistry::SetBlockLinear(u32 image_handle) {
    const std::scoped_lock lock{mutex};
    for (Pool& pool : pools) {
        if (image_handle >= pool.handle && image_handle - pool.handle < pool.count) {
            pool.block_linear = true;
            return true;
        }
    }
    return false;
}

namespace {

/// TIPC session whose unidentified commands are logged and rejected.
template <typename T>
class LoggingTipcSession : public ServiceFramework<T> {
public:
    LoggingTipcSession(Core::System& system_, const char* name)
        : ServiceFramework<T>{system_, name} {}

    Result HandleSyncRequest(Kernel::KServerSession& session, HLERequestContext& ctx) override {
        if (!ctx.IsTipc() || this->handlers_tipc.contains(ctx.GetCommand())) {
            LOG_DEBUG(Service_AM, "{} command={}", this->GetServiceName(), ctx.GetCommand());
            return ServiceFrameworkBase::HandleSyncRequest(session, ctx);
        }
        const auto guard = this->LockService();
        const u32* packet = ctx.CommandBuffer();
        LOG_WARNING(Service_AM, "{} unsupported command={}, header={:08x},{:08x} words={:08x}",
                    this->GetServiceName(), ctx.GetCommand(), packet[0], packet[1],
                    fmt::join(std::span<const u32>(packet + 2, std::min<u32>(packet[1] & 0x3ff, 32)),
                              ","));
        this->InvokeRequestTipc(ctx);
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(Kernel::ResultNotImplemented.raw);
        return this->system.IsPoweredOn() ? ctx.WriteToOutgoingCommandBuffer() : ResultSuccess;
    }
};

/// nn::codec::video::detail::ipc::VideoDecService: one decoder instance. Pictures are decoded
/// when their input is submitted and written into the registered pool images in the order the
/// images were registered and released. The fence returned for an image is a syncpoint value
/// reached when the next picture has been written into it.
class IVideoDecService final : public LoggingTipcSession<IVideoDecService> {
    struct Fence {
        u32 syncpoint{};
        u64 value{};
    };
    struct Properties {
        u32 image{};
        u32 width{};
        u32 height{};
        u64 pts{};
        bool end_of_stream{};
    };

public:
    IVideoDecService(Core::System& system_, Kernel::KProcess* process_,
                     std::shared_ptr<Nvidia::Module> nv_module_,
                     std::shared_ptr<VideoImagePoolRegistry> pools_)
        : LoggingTipcSession{system_, "IVideoDecService"}, process{process_},
          nv_module{std::move(nv_module_)}, pools{std::move(pools_)},
          service_context{system_, "VideoDecService"} {
        static const FunctionInfo functions[] = {
            {3, &IVideoDecService::GetSubmissionReadyEvent, "GetSubmissionReadyEvent"},
            {6, &IVideoDecService::Start, "Start"},
            {8, &IVideoDecService::Flush, "Flush"},
            {101, &IVideoDecService::Configure, "Configure"},
            {109, &IVideoDecService::SubmitInputBuffer, "SubmitInputBuffer"},
            {115, &IVideoDecService::Finalize, "Finalize"},
            {200, &IVideoDecService::Initialize, "Initialize"},
            {201, &IVideoDecService::Initialize, "InitializeH264"},
            {207, &IVideoDecService::Stop, "Stop"},
            {210, &IVideoDecService::Initialize, "InitializeWithoutWorkMemory"},
            {212, &IVideoDecService::UnregisterBuffer, "UnregisterBuffer"},
            {213, &IVideoDecService::PopOutputProperties, "PopOutputProperties"},
            {311, &IVideoDecService::RegisterBuffer, "RegisterBuffer"},
            {314, &IVideoDecService::ReleaseOutputBuffer, "ReleaseOutputBuffer"},
        };
        RegisterHandlersTipc(functions);
        submission_ready = service_context.CreateEvent("VideoDecSubmissionReady");
        submission_ready->Signal(kernel);
    }

    ~IVideoDecService() override {
        service_context.CloseEvent(submission_ready);
        if (syncpoint) {
            nv_module->GetContainer().GetSyncpointManager().FreeSyncpoint(*syncpoint);
        }
    }

private:
    void ReplyResult(HLERequestContext& ctx, Result result) {
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(result.raw);
    }

    // nn::codec::Initialize{H264,Vp8,Vp9,Av1}Decoder. All forms carry a copy handle (the work
    // memory, or the process for 210) followed by:
    //   200: u64 work size, codec, input format, output format
    //   201: u64 work size, codec (0), input format, output format, output mode (H.264 only)
    //   210: codec, input format, output format
    // The service's codec numbers are 0 H.264, 1 VP9, 3 AV1 and 4 VP8.
    void Initialize(HLERequestContext& ctx) {
        const auto* packet = ctx.CommandBuffer();
        const u32 command = ctx.GetCommand();
        const u32 words = command == 200 ? 5 : command == 201 ? 6 : 3;
        const u32 first = command == 210 ? 4 : 6;
        Result result = Kernel::ResultInvalidCombination;
        std::optional<VideoCommon::NativeVideoCodec> native;
        if (packet[0] == command + 16 && packet[1] == (0x8000'0000U | words) && packet[2] == 2) {
            switch (packet[first]) {
            case 0:
                native = VideoCommon::NativeVideoCodec::H264;
                break;
            case 1:
                native = VideoCommon::NativeVideoCodec::Vp9;
                break;
            case 3:
                native = VideoCommon::NativeVideoCodec::Av1;
                break;
            case 4:
                native = VideoCommon::NativeVideoCodec::Vp8;
                break;
            }
            result = Kernel::ResultNotImplemented;
        }
        if (native) {
            codec = *native;
            input_format = packet[first + 1];
            output_format = packet[first + 2];
            auto& syncpoints = nv_module->GetContainer().GetSyncpointManager();
            syncpoint = syncpoints.AllocateSyncpoint(true);
            // A syncpoint freed by an earlier decoder keeps its count; fences continue from it,
            // or the client would see its first images as already written.
            next_target = system.Host1x().GetSyncpointManager().GetGuestSyncpointValue(*syncpoint);
            result = ResultSuccess;
        }
        LOG_INFO(Service_AM, "video decoder command={}, codec={}, input format={}, output format={}, result={:#x}",
                 command, packet[first], packet[first + 1], packet[first + 2], result.raw);
        ReplyResult(ctx, result);
    }

    // Codec configuration (avcC for H.264) in an A buffer, then the u16 width and height
    void Configure(HLERequestContext& ctx) {
        const auto* packet = ctx.CommandBuffer();
        Result result = Kernel::ResultInvalidCombination;
        if (packet[0] == 0x0010'0075 && (packet[1] & 0x3ff) == 8) {
            width = packet[5] & 0xffff;
            height = packet[6] & 0xffff;
            const auto config = ctx.ReadBuffer();
            // nn::movie reads output format 2 as RGBA and 3 and 6 as NV12
            const auto output = output_format == 2 ? VideoCommon::NativeVideoOutput::Rgba
                                                   : VideoCommon::NativeVideoOutput::Nv12;
            decoder = VideoCommon::NativeVideoDecoder::Create(codec, config, output);
            result = decoder ? ResultSuccess : Kernel::ResultNotImplemented;
            LOG_INFO(Service_AM, "video decoder configure {}x{}, config={} bytes, result={:#x}",
                     width, height, config.size(), result.raw);
        }
        ReplyResult(ctx, result);
    }

    void GetSubmissionReadyEvent(HLERequestContext& ctx) {
        IPC::ResponseBuilder rb{ctx, 2, 1};
        rb.Push(ResultSuccess.raw);
        rb.PushCopyObjects(ctx, submission_ready->GetReadableEvent());
    }

    void Start(HLERequestContext& ctx) {
        started = true;
        end_of_stream_queued = false;
        // The decoder accepts input once started; a client that cleared the event when it
        // attached it waits for this signal before its first submission.
        SignalSubmissionReady();
        ReplyResult(ctx, ResultSuccess);
    }

    void Stop(HLERequestContext& ctx) {
        started = false;
        ReplyResult(ctx, ResultSuccess);
    }

    void Flush(HLERequestContext& ctx) {
        if (decoder) {
            decoder->Flush();
        }
        pending.clear();
        end_of_stream_queued = false;
        SignalSubmissionReady();
        ReplyResult(ctx, ResultSuccess);
    }

    void Finalize(HLERequestContext& ctx) {
        decoder.reset();
        ReplyResult(ctx, ResultSuccess);
    }

    void PushFence(HLERequestContext& ctx, Result result, const Fence& fence) {
        IPC::ResponseBuilder rb{ctx, 6};
        rb.Push(result.raw);
        // nn::gs::NativeSyncHandle of a syncpoint: the inverted id, a zero word, the value
        rb.Push<u32>(~fence.syncpoint);
        rb.Push<u32>(0);
        rb.Push<u64>(fence.value);
    }

    /// Queues an image for the next picture and returns the fence of that picture. The image is
    /// written once the application's release fence, if any, has been reached.
    Fence Enqueue(u32 image, std::optional<Fence> release = std::nullopt) {
        const Fence fence{*syncpoint, ++next_target};
        free_images.push_back({image, release});
        WritePending();
        return fence;
    }

    // u32 image handle, a reserved word and a reserved u64
    void RegisterBuffer(HLERequestContext& ctx) {
        const auto* packet = ctx.CommandBuffer();
        Result result = Kernel::ResultInvalidCombination;
        Fence fence{};
        if (packet[0] == 0x147 && (packet[1] & 0x3ff) == 4 && syncpoint) {
            result = pools->Find(packet[2]) ? ResultSuccess : Kernel::ResultNotFound;
            if (result.IsSuccess()) {
                fence = Enqueue(packet[2]);
            }
        }
        LOG_DEBUG(Service_AM, "video decoder register image={:#x}, fence={}, result={:#x}",
                  packet[2], fence.value, result.raw);
        PushFence(ctx, result, fence);
    }

    void UnregisterBuffer(HLERequestContext& ctx) {
        const auto* packet = ctx.CommandBuffer();
        std::erase_if(free_images, [&](const auto& entry) { return entry.first == packet[2]; });
        ReplyResult(ctx, ResultSuccess);
    }

    // u32 image handle, a reserved word and the fence after which the application is done with it
    void ReleaseOutputBuffer(HLERequestContext& ctx) {
        const auto* packet = ctx.CommandBuffer();
        Result result = Kernel::ResultInvalidCombination;
        Fence fence{};
        if (packet[0] == 0x14a && (packet[1] & 0x3ff) == 8 && syncpoint) {
            result = pools->Find(packet[2]) ? ResultSuccess : Kernel::ResultNotFound;
            if (result.IsSuccess()) {
                const Fence release{~packet[4], u64{packet[6]} | (u64{packet[7]} << 32)};
                fence = Enqueue(packet[2], release);
            }
        }
        if (result.IsError() && release_failures++ < 4) {
            LOG_WARNING(Service_AM, "video decoder release failed: words={:08x}, result={:#x}",
                        fmt::join(std::span<const u32>(packet, 10), ","), result.raw);
        }
        PushFence(ctx, result, fence);
    }

    // Input in an A buffer, then the presentation time in microseconds and an end of stream flag
    void SubmitInputBuffer(HLERequestContext& ctx) {
        const auto* packet = ctx.CommandBuffer();
        Result result = Kernel::ResultInvalidCombination;
        if (packet[0] == 0x0010'007d && (packet[1] & 0x3ff) == 3) {
            result = Kernel::ResultInvalidState;
            if (decoder && pending.size() >= MaxPendingPictures) {
                // The input queue is full: the console's decoder answers with this result and
                // nn::codec clients (Donkey Kong Bananza) submit again after the submission-ready
                // event, instead of having the server buffer the whole stream.
                result = ResultInputQueueFull;
            } else if (decoder) {
                const s64 pts = static_cast<s64>(u64{packet[5]} | (u64{packet[6]} << 32));
                const bool end_of_stream = (packet[7] & 0xff) != 0;
                // Submissions that only signal the end of the stream carry no input
                const auto data = ctx.CanReadBuffer() ? ctx.ReadBuffer() : std::span<const u8>{};
                std::vector<VideoCommon::DecodedVideoFrame> frames;
                if (!data.empty()) {
                    decoder->Decode(data, pts, frames);
                }
                if (end_of_stream) {
                    decoder->Drain(frames);
                }
                if (!frames.empty()) {
                    last_pts = frames.back().pts;
                }
                // Clients wait for the record that ends the stream (Fire Emblem: Fortune's Weave
                // submits again until it comes), so it takes an image like a picture
                if (end_of_stream && !end_of_stream_queued) {
                    end_of_stream_queued = true;
                    VideoCommon::DecodedVideoFrame end{};
                    end.pts = last_pts;
                    end.end_of_stream = true;
                    frames.push_back(std::move(end));
                }
                for (auto& frame : frames) {
                    pending.push_back(std::move(frame));
                }
                decoded += frames.size();
                WritePending();
                // A client that submits without waiting for the event and never hands images
                // back (Kirby Air Riders' background video) would grow this without bound; the
                // oldest pictures are dropped instead.
                while (pending.size() > MaxBufferedPictures) {
                    if (dropped++ == 0) {
                        LOG_WARNING(Service_AM, "video decoder output is not consumed, dropping pictures");
                    }
                    pending.pop_front();
                }
                ++submitted;
                SignalSubmissionReady();
                if (submitted % 500 == 1) {
                    LOG_DEBUG(Service_AM, "video decoder submit #{}: {} bytes, pts={}, decoded={}, pending={}, free={}, ready={}",
                              submitted, data.size(), pts, decoded, pending.size(), free_images.size(),
                              properties.size());
                }
                result = ResultSuccess;
            }
        }
        IPC::ResponseBuilder rb{ctx, 4};
        rb.Push(result.raw);
        rb.Push<u64>(submitted);
    }

    /// Waits until the GPU work that samples a released image has been processed, so a picture is
    /// not written while the image is still read. Bounded in case the fence is never signaled.
    void WaitForRelease(const std::optional<Fence>& release) {
        auto& host_syncpoints = system.Host1x().GetSyncpointManager();
        if (!release || release->syncpoint >= Nvidia::MaxSyncPoints) {
            return;
        }
        const u32 value = static_cast<u32>(release->value);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds{200};
        while (!host_syncpoints.IsReadyGuest(release->syncpoint, value)) {
            if (std::chrono::steady_clock::now() >= deadline) {
                LOG_WARNING(Service_AM, "Video image release fence {}:{} timed out",
                            release->syncpoint, value);
                return;
            }
            std::this_thread::sleep_for(std::chrono::microseconds{200});
        }
    }

    /// Writes decoded pictures into queued images and signals their fences
    // Pictures decoded ahead of the images the client has handed back. Clients submit whenever
    // the submission-ready event is signaled, so it is only signaled while this backlog is
    // short; otherwise a whole stream would be decoded into host memory at once.
    static constexpr size_t MaxPendingPictures = 2;
    // Module 111 (codec), description 3603: the result nn::codec clients treat as "try again"
    static constexpr Result ResultInputQueueFull{static_cast<ErrorModule>(111), 3603};
    static constexpr size_t MaxBufferedPictures = 16;

    void SignalSubmissionReady() {
        if (pending.size() < MaxPendingPictures) {
            submission_ready->Signal(kernel);
        }
    }

    void WritePending() {
        auto& memory = process->GetMemory();
        auto& host_syncpoints = system.Host1x().GetSyncpointManager();
        while (!pending.empty() && !free_images.empty()) {
            const auto& frame = pending.front();
            const u32 image = free_images.front().first;
            WaitForRelease(free_images.front().second);
            free_images.pop_front();
            const auto target = pools->Find(image);
            if (frame.end_of_stream) {
                properties.push_back({image, width, height, u64(frame.pts), true});
            } else if (target) {
                const auto& layout = target->layout;
                const u32 rows = std::min(frame.height, target->height);
                const u32 row = std::min(frame.width, target->width);
                const bool rgba = target->format == VideoImageFormat::Rgba;
                if (rgba != frame.chroma.empty()) {
                    LOG_ERROR(Service_AM, "Decoded picture format does not match image {:#x}", image);
                }
                if (rgba && target->block_linear) {
                    // The texture of nvnTextureInitializeFromNativeImage: block linear with the
                    // builder's default block height, rows of the full image width
                    std::vector<u8> linear(size_t{target->width} * 4 * target->height);
                    for (u32 y = 0; y < rows; ++y) {
                        std::memcpy(linear.data() + size_t{target->width} * 4 * y,
                                    frame.luma.data() + size_t{frame.width} * 4 * y,
                                    size_t{row} * 4);
                    }
                    const u32 block_height = VideoImageBlockHeight(target->height);
                    std::vector<u8> tiled(Tegra::Texture::CalculateSize(
                        true, 4, target->width, target->height, 1, block_height, 0));
                    // Native applications sample with the console's GOB layout
                    Tegra::Texture::SwizzleTexture(tiled, linear, 4, target->width,
                                                   target->height, 1, block_height, 0, 1, true);
                    memory.WriteBlock(target->address + layout.luma_offset, tiled.data(),
                                      tiled.size());
                }
                if (!rgba && target->block_linear) {
                    LOG_WARNING(Service_AM, "Block linear NV12 decoder images are not supported");
                }
                for (u32 y = 0; rgba && !target->block_linear && y < rows; ++y) {
                    memory.WriteBlock(target->address + layout.luma_offset + layout.luma_pitch * y,
                                      frame.luma.data() + size_t{frame.width} * 4 * y, size_t{row} * 4);
                }
                for (u32 y = 0; !rgba && y < rows; ++y) {
                    memory.WriteBlock(target->address + layout.luma_offset + layout.luma_pitch * y,
                                      frame.luma.data() + size_t{frame.width} * y, row);
                }
                const u32 chroma_row = std::min((frame.width + 1) & ~1U, target->width);
                for (u32 y = 0; !rgba && y < (rows + 1) / 2; ++y) {
                    memory.WriteBlock(
                        target->address + layout.chroma_offset + layout.chroma_pitch * y,
                        frame.chroma.data() + size_t{(frame.width + 1) & ~1U} * y, chroma_row);
                }
                properties.push_back({image, frame.width, frame.height, u64(frame.pts), false});
            }
            pending.pop_front();
            nv_module->GetContainer().GetSyncpointManager().IncrementSyncpointMaxExt(*syncpoint, 1);
            // The picture is complete when it is written, for the GPU and for the guest
            host_syncpoints.IncrementHost(*syncpoint);
            host_syncpoints.IncrementGuest(*syncpoint);
            SignalSubmissionReady();
        }
    }

    // One record per written picture: the image handle, four u16 sizes as words, a reserved word,
    // the presentation time and a flag byte
    void PopOutputProperties(HLERequestContext& ctx) {
        Result result = Kernel::ResultNotFound;
        Properties record{};
        if (!properties.empty()) {
            record = properties.front();
            properties.pop_front();
            result = ResultSuccess;
        }
        IPC::ResponseBuilder rb{ctx, 11};
        rb.Push(result.raw);
        rb.Push<u32>(record.image);
        rb.Push<u32>(record.width);
        rb.Push<u32>(record.height);
        rb.Push<u32>(record.width);
        rb.Push<u32>(record.height);
        rb.Push<u32>(0);
        rb.Push<u64>(record.pts * 1000);
        rb.Push<u32>(record.end_of_stream ? 1 : 0);
    }

    Kernel::KProcess* process;
    std::shared_ptr<Nvidia::Module> nv_module;
    std::shared_ptr<VideoImagePoolRegistry> pools;
    KernelHelpers::ServiceContext service_context;
    Kernel::KEvent* submission_ready{};

    VideoCommon::NativeVideoCodec codec{};
    u32 input_format{};
    u32 output_format{};
    u32 width{};
    u32 height{};
    bool started{};
    bool end_of_stream_queued{};
    s64 last_pts{};
    std::unique_ptr<VideoCommon::NativeVideoDecoder> decoder;

    std::optional<u32> syncpoint;
    u64 next_target{};
    u64 submitted{};
    u32 release_failures{};
    u64 decoded{};
    u64 dropped{};
    std::deque<std::pair<u32, std::optional<Fence>>> free_images;
    std::deque<VideoCommon::DecodedVideoFrame> pending;
    std::deque<Properties> properties;
};

class IVideoEncService final : public LoggingTipcSession<IVideoEncService> {
public:
    explicit IVideoEncService(Core::System& system_)
        : LoggingTipcSession{system_, "IVideoEncService"} {}
};

class IVideoCodecRootSession final : public LoggingTipcSession<IVideoCodecRootSession> {
public:
    IVideoCodecRootSession(Core::System& system_, VideoCodecKind kind_, Kernel::KProcess* process_,
                           std::shared_ptr<Nvidia::Module> nv_module_,
                           std::shared_ptr<VideoImagePoolRegistry> pools_)
        : LoggingTipcSession{system_, kind_ == VideoCodecKind::Decoder ? "IVideoDecRoot"
                                                                     : "IVideoEncRoot"},
          kind{kind_}, process{process_}, nv_module{std::move(nv_module_)},
          pools{std::move(pools_)} {
        static const FunctionInfo functions[] = {
            {1, &IVideoCodecRootSession::CreateService, "CreateService"},
        };
        RegisterHandlersTipc(functions);
    }

private:
    // nn::codec::video::detail::ipc::CreateVideo{De,En}coderService: no arguments; the reply
    // holds a move handle and the result
    void CreateService(HLERequestContext& ctx) {
        const auto* packet = ctx.CommandBuffer();
        Result result = Kernel::ResultInvalidCombination;
        Kernel::KClientSession* client{};
        if (packet[0] == 0x11 && packet[1] == 0) {
            std::shared_ptr<SessionRequestHandler> handler;
            if (kind == VideoCodecKind::Decoder) {
                handler = std::make_shared<IVideoDecService>(system, process, nv_module, pools);
            } else {
                handler = std::make_shared<IVideoEncService>(system);
            }
            result = CreateGraphicsSession(system, ctx, std::move(handler), &client);
        }
        LOG_DEBUG(Service_AM, "video codec service kind={}, result={:#x}", static_cast<int>(kind),
                  result.raw);
        IPC::ResponseBuilder rb{ctx, 2, 0, 1};
        rb.Push(result.raw);
        ctx.AddMoveObject(client);
    }

    VideoCodecKind kind;
    Kernel::KProcess* process;
    std::shared_ptr<Nvidia::Module> nv_module;
    std::shared_ptr<VideoImagePoolRegistry> pools;
};

} // Anonymous namespace

std::shared_ptr<SessionRequestHandler> MakeVideoCodecRootSession(
    Core::System& system, VideoCodecKind kind, Kernel::KProcess* process,
    std::shared_ptr<Nvidia::Module> nv_module, std::shared_ptr<VideoImagePoolRegistry> pools) {
    return std::make_shared<IVideoCodecRootSession>(system, kind, process, std::move(nv_module),
                                                    std::move(pools));
}

} // namespace Service::AM
