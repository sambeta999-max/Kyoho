// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <memory>
#include <span>
#include <vector>

#include "common/common_types.h"

namespace VideoCommon {

/// Codecs of nn::codec::video decoders, by the codec type of their initialization request
enum class NativeVideoCodec : u32 {
    H264 = 1,
    Vp8 = 2,
    Vp9 = 3,
    Av1 = 4,
};

enum class NativeVideoOutput {
    Nv12,
    Rgba,
};

/// A decoded picture, tightly packed (pitch equal to the row width in bytes). NV12 has a luma
/// plane and an interleaved chroma plane; RGBA has 8-bit RGBA pixels in the first plane.
struct DecodedVideoFrame {
    u32 width{};
    u32 height{};
    s64 pts{};
    bool end_of_stream{}; ///< No picture: the output record that ends the stream
    std::vector<u8> luma;
    std::vector<u8> chroma;
};

/// Software decoder for the compressed streams of native (Switch 2) applications, independent of
/// the NVDEC engine emulation.
class NativeVideoDecoder {
public:
    virtual ~NativeVideoDecoder() = default;

    /// Creates a decoder; extradata is the codec configuration (avcC for H.264), possibly empty
    static std::unique_ptr<NativeVideoDecoder> Create(NativeVideoCodec codec,
                                                      std::span<const u8> extradata,
                                                      NativeVideoOutput output);

    /// Decodes one compressed access unit and appends the pictures that became available
    virtual bool Decode(std::span<const u8> data, s64 pts,
                        std::vector<DecodedVideoFrame>& frames) = 0;

    /// Appends the pictures still held for reordering at the end of the stream
    virtual void Drain(std::vector<DecodedVideoFrame>& frames) = 0;

    /// Discards held pictures, as after a seek
    virtual void Flush() = 0;

    /// Whether input given to Decode is still being decoded (frame parallel decoders)
    virtual bool Pending() const {
        return false;
    }

    /// Waits for the oldest input still being decoded and appends its picture
    virtual void Output(std::vector<DecodedVideoFrame>& frames) {}
};

} // namespace VideoCommon
