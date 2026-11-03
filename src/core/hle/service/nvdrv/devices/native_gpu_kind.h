// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <optional>
#include "common/common_types.h"
#include "video_core/pte_kind.h"

namespace Service::Nvidia::Devices {

inline std::optional<u8> TranslateNativeUncompressedKind(u8 kind, u8 fallback) {
    // NVIDIA's Turing MMU uses 6 for GENERIC_MEMORY and 7 for INVALID:
    // open-gpu-doc/manuals/turing/tu104/dev_mmu.ref.txt. Eden's older kind
    // numbering uses 0xfe for generic memory. Equal kinds need no compression.
    if (kind != fallback) { return std::nullopt; }
    switch (kind) {
    case 0: return static_cast<u8>(Tegra::PTEKind::PITCH);
    case 1: return static_cast<u8>(Tegra::PTEKind::Z16);
    case 2: return static_cast<u8>(Tegra::PTEKind::S8);
    case 3: return static_cast<u8>(Tegra::PTEKind::S8Z24);
    case 4: return static_cast<u8>(Tegra::PTEKind::ZF32_X24S8);
    case 5: return static_cast<u8>(Tegra::PTEKind::Z24S8);
    case 6: return static_cast<u8>(Tegra::PTEKind::GENERIC_16BX2);
    // SMSKED_MESSAGE (ampere/ga100/dev_mmu.h) holds SM-to-SKED messages. It is
    // linear and uncompressed, so it is backed like generic memory.
    case 0xf: return static_cast<u8>(Tegra::PTEKind::GENERIC_16BX2);
    default: return std::nullopt;
    }
}

struct NativeMemoryKind {
    u8 backend;
    u8 selected;
};

inline std::optional<NativeMemoryKind> SelectNativeMemoryKind(u8 requested, u8 fallback) {
    if (const auto kind = TranslateNativeUncompressedKind(requested, fallback)) {
        return NativeMemoryKind{*kind, requested};
    }
    // Compressible kinds are not emulated: the caller supplies the uncompressed alternative, and
    // returning it as the selected kind makes the SDK drop its compressed preference. TU104
    // (dev_mmu.ref.txt): 8 and 9 are compressible GENERIC_MEMORY (6), 0xa-0xe the compressible
    // S8 (2), Z16 (1), S8Z24 (3), ZF32_X24S8 (4) and Z24S8 (5).
    static constexpr std::array<u8, 7> COMPRESSED_OF{0, 0xb, 0xa, 0xc, 0xd, 0xe, 8};
    if (fallback >= 1 && fallback <= 6 &&
        (requested == COMPRESSED_OF[fallback] || (fallback == 6 && requested == 9))) {
        if (const auto kind = TranslateNativeUncompressedKind(fallback, fallback)) {
            return NativeMemoryKind{*kind, fallback};
        }
    }
    return std::nullopt;
}

} // namespace Service::Nvidia::Devices
