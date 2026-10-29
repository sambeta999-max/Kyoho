// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <optional>

#include "common/common_types.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/semaphore.h"

namespace Tegra::Engines {

/// Queue meta data 3.0, the compute launch descriptor of AMPERE_COMPUTE_B. Bit positions are
/// those of NVC7C0_QMDV03_00 in clc7c0qmd.h (open-gpu-doc).
struct QmdV3 {
    static constexpr u32 NUM_WORDS = 64;
    static constexpr u32 NUM_RELEASES = 2;

    enum class DependentAction : u32 {
        IncrementPut = 0,
        Schedule = 1,
        InvalidateCopySchedule = 3,
        DecrementDependence = 4,
    };

    struct Dependent {
        GPUVAddr address;
        DependentAction action;
    };

    std::array<u32, NUM_WORDS> words{};

    /// Bits [lo, hi] of the descriptor; no field crosses a word boundary
    [[nodiscard]] constexpr u32 Field(u32 hi, u32 lo) const {
        const u32 width = hi - lo + 1;
        const u32 value = words[lo / 32] >> (lo % 32);
        return width == 32 ? value : value & ((1U << width) - 1);
    }

    [[nodiscard]] constexpr u32 MajorVersion() const {
        return Field(583, 580);
    }

    [[nodiscard]] constexpr u32 DependenceCounter() const {
        return Field(157, 142);
    }

    [[nodiscard]] constexpr GPUVAddr ProgramAddress() const {
        return (GPUVAddr{Field(1584, 1568)} << 32) | Field(1567, 1536);
    }

    [[nodiscard]] constexpr bool HasWork() const {
        return Field(415, 384) != 0 && Field(431, 416) != 0 && Field(463, 448) != 0;
    }

    [[nodiscard]] constexpr std::optional<SemaphoreRelease> GetRelease(u32 index) const {
        const u32 base = 768 + index * 128;
        if (Field(base + 55, base + 55) == 0) {
            return std::nullopt;
        }
        const bool payload_64 = Field(base + 61, base + 61) != 0;
        SemaphoreRelease release{
            .address = (GPUVAddr{Field(base + 39, base + 32)} << 32) | Field(base + 31, base),
            .payload = (u64{payload_64 ? Field(base + 127, base + 96) : 0} << 32) |
                       Field(base + 95, base + 64),
            .payload_64 = payload_64,
            // STRUCTURE_SIZE zero is four words, the payload and a timestamp
            .timestamp = Field(base + 63, base + 62) == 0,
            .reduction = std::nullopt,
            .reduction_signed = Field(base + 57, base + 56) == 1,
        };
        if (Field(base + 58, base + 58) != 0) {
            release.reduction = static_cast<SemaphoreReduction>(Field(base + 54, base + 52));
        }
        return release;
    }

    /// The first dependent QMD; the second has no pointer of its own
    [[nodiscard]] constexpr std::optional<Dependent> GetDependent() const {
        if (Field(512, 512) == 0) {
            return std::nullopt;
        }
        return Dependent{
            .address = GPUVAddr{Field(511, 480)} << 8,
            .action = static_cast<DependentAction>(Field(515, 513)),
        };
    }

    [[nodiscard]] constexpr bool HasSecondDependent() const {
        return Field(517, 517) != 0;
    }

    /// The launch parameters in the QMD 0.x layout the rest of the GPU code reads. The program
    /// address is absolute, so program_start is left at zero.
    [[nodiscard]] KeplerCompute::LaunchParams ToLaunchParams() const {
        KeplerCompute::LaunchParams params{};
        params.linked_tsc.Assign(Field(382, 382));
        params.grid_dim_x.Assign(Field(415, 384));
        params.grid_dim_y.Assign(Field(431, 416));
        params.grid_dim_z.Assign(Field(463, 448));
        params.shared_alloc.Assign(Field(561, 544));
        params.block_dim_x.Assign(Field(607, 592));
        params.block_dim_y.Assign(Field(623, 608));
        params.block_dim_z.Assign(Field(639, 624));
        params.const_buffer_enable_mask.Assign(Field(647, 640));
        for (u32 i = 0; i < KeplerCompute::NumConstBuffers; ++i) {
            const u32 base = 1024 + i * 64;
            auto& config = params.const_buffer_config[i];
            config.address_low = Field(base + 31, base);
            config.address_high.Assign(Field(base + 48, base + 32));
            config.size.Assign(Field(base + 63, base + 51) << 4);
        }
        params.local_pos_alloc.Assign(Field(759, 736));
        params.local_neg_alloc.Assign(Field(1623, 1600));
        params.barrier_alloc.Assign(Field(767, 763));
        params.sass_version.Assign(Field(1663, 1656));
        return params;
    }
};

} // namespace Tegra::Engines
