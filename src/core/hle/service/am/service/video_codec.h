// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include "common/common_types.h"

namespace Core {
class System;
}
namespace Kernel {
class KProcess;
}
namespace Service {
class SessionRequestHandler;
}
namespace Service::Nvidia {
class Module;
}

namespace Service::AM {

enum class VideoCodecKind {
    Decoder,
    Encoder,
};

/// Pixel formats of decoder images. nn::codec::MakeDecodeImagePreference maps output formats 1-6
/// to preference types 3, 2, 1, 0xa, 0xb, 0xe (codec mode 1) or 6, 5, 4, 0xc, 0xd, 0xf (mode 2);
/// nn::movie reads output format 2 as RGBA and 3 as NV12.
enum class VideoImageFormat {
    Rgba,
    Nv12,
};

std::optional<VideoImageFormat> VideoImageFormatFromPreference(u32 type);

/// Placement of one decoded picture in a pool, pitch-linear, offsets relative to the pool buffer.
/// RGBA uses the luma fields only. This is the layout described to nn::gs::sdk::ReadUserImage*.
struct VideoImageLayout {
    u64 luma_offset{};
    u64 luma_pitch{};
    u64 chroma_offset{};
    u64 chroma_pitch{};
};

/// Log2 of the block height in GOBs that NVN's texture builder picks for a 2D block linear
/// texture of the given height: the smallest power of two covering it, at most 16 GOBs.
u32 VideoImageBlockHeight(u32 height);

/// Bytes reserved per decoded picture: four bytes per pixel, which also fits 4:2:0 with 16-bit
/// samples, with rows aligned to 128 and pictures aligned to 64 KiB.
u64 VideoImageSize(u32 width, u32 height);

/// Layout of picture index in a pool of pictures of the given size and format
VideoImageLayout MakeVideoImageLayout(VideoImageFormat format, u32 width, u32 height, u32 index);

/// Decoder image pools (nn::codec::InitializeDecoderImagePool). Image handles are the pool
/// handle plus the picture index (nn::gs::sdk::CalculateNativeImageHandle).
class VideoImagePoolRegistry {
public:
    struct Pool {
        VideoImageFormat format{};
        u32 handle{};
        u32 count{};
        u32 width{};
        u32 height{};
        u64 address{};
        u64 size{};
        /// nvmap handle of the native memory holding the pool
        u32 memory_id{};
        /// Pictures are sampled by the GPU (nn::gpu::sdk::QueryNativeImage) and stored block
        /// linear instead of pitch linear
        bool block_linear{};
    };
    struct Image {
        VideoImageFormat format{};
        u32 width{};
        u32 height{};
        u64 address{};
        VideoImageLayout layout{};
        bool block_linear{};
        u32 memory_id{};
    };

    void Add(const Pool& pool);
    bool Remove(u32 handle);
    std::optional<Image> Find(u32 image_handle) const;
    /// Switches the pool of an image to block linear storage
    bool SetBlockLinear(u32 image_handle);

private:
    mutable std::mutex mutex;
    std::vector<Pool> pools;
};

/// Root session of nn::codec::video, opened by IGraphicsSupervisor 902/905 (decoder) and 1002
/// (encoder). Its command 1 opens one VideoDecService or VideoEncService per codec instance.
std::shared_ptr<SessionRequestHandler> MakeVideoCodecRootSession(
    Core::System& system, VideoCodecKind kind, Kernel::KProcess* process,
    std::shared_ptr<Nvidia::Module> nv_module, std::shared_ptr<VideoImagePoolRegistry> pools);

} // namespace Service::AM
