// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <optional>
#include <span>
#include "common/common_types.h"

namespace Service::AM {

struct NativeImageRequirements {
    u64 size;
    u64 alignment;
};

// Inter-module block image layout. Format mapping from NVN format indices in sdk+0x5e0550
// and the 44-byte NVN format table at sdk+0xe03968 (accessed by sdk+0x5eb9c0):
// 1 = RGBA8 / RGBA8_SRGB (NVN 0x25 / 0x38, 4 BPP)
// 2 = RGBX8 / RGBX8_SRGB (NVN 0x37 / 0x6d, 4 BPP)
// 3 = BGRA8 / BGRA8_SRGB (NVN 0x9a / 0x9c, 4 BPP)
// 4 = RGB10A2 (NVN 0x3d, 4 BPP)
// 5 = RGBA16F (NVN 0x29, 8 BPP)
// 6 = RGBX16F (NVN 0x71, 8 BPP)
// Row pitch is in 64-byte GOBs; block height is in pixels.
constexpr u64 NativeImageBytesPerPixel(u32 format) {
    switch (format) {
    case 1:
    case 2:
    case 3:
    case 4:
        return 4;
    case 5:
    case 6:
        return 8;
    default:
        return 0;
    }
}

inline std::optional<NativeImageRequirements> CalculateNativeImageRequirements(
    std::span<const u32> preference, u32 count) {
    const u64 bpp = preference.size() == 16 ? NativeImageBytesPerPixel(preference[6]) : 0;
    if (preference.size() != 16 || count != 1 || preference[0] != 0 ||
        (preference[1] != 1 && preference[1] != 2) || preference[3] != 0 || bpp == 0 ||
        preference[7] != 1 || preference[4] < 3 || preference[4] > 8) {
        return std::nullopt;
    }
    for (size_t i = 8; i < preference.size(); ++i) {
        if (preference[i] != 0) { return std::nullopt; }
    }
    const u64 width = (preference[2] & 0xffffU) + 1;
    const u64 height = (preference[2] >> 16) + 1;
    const u64 pitch_gobs = preference[5];
    if (pitch_gobs < (width * bpp + 63) / 64 || pitch_gobs > 4096) {
        return std::nullopt;
    }
    const u64 block_height = 1ULL << preference[4];
    const u64 aligned_height = (height + block_height - 1) & ~(block_height - 1);
    return NativeImageRequirements{pitch_gobs * 64 * aligned_height, 64 * block_height};
}

} // namespace Service::AM
