// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <optional>

#include "common/common_types.h"

namespace VideoCore {
class RasterizerInterface;
}

namespace Tegra {
class MemoryManager;
}

// Semaphores of the Ampere classes: SEM_EXECUTE of the host class (clc56f.h),
// REPORT_SEMAPHORE_EXECUTE of the 3D and compute classes (clc797.h, clc7c0.h) and the releases of
// QMD 3.0 (clc7c0qmd.h).
namespace Tegra::Engines {

/// Reduction operations, numbered as in the 3D, compute and QMD encodings
enum class SemaphoreReduction : u32 {
    Add = 0,
    Min = 1,
    Max = 2,
    Inc = 3,
    Dec = 4,
    And = 5,
    Or = 6,
    Xor = 7,
};

struct SemaphoreRelease {
    GPUVAddr address{};
    u64 payload{};
    bool payload_64{};
    /// Four word structure: the payload followed by a 64-bit timestamp
    bool timestamp{};
    std::optional<SemaphoreReduction> reduction;
    bool reduction_signed{};
};

/// Value of a semaphore after a reduction, in 32 or 64 bits
[[nodiscard]] u64 ReduceSemaphore(SemaphoreReduction op, u64 old_value, u64 value, bool is_signed,
                                  bool is_64);

/// Writes a release once the preceding GPU work is done
void ReleaseSemaphore(VideoCore::RasterizerInterface& rasterizer, MemoryManager& memory_manager,
                      const SemaphoreRelease& release);

/// SEM_EXECUTE of the host class, with SEM_ADDR_LO/HI and SEM_PAYLOAD_LO/HI before it
struct HostSemaphore {
    enum class Operation : u32 {
        Acquire = 0,
        Release = 1,
        AcquireStrictGeq = 2,
        AcquireCircGeq = 3,
        AcquireAnd = 4,
        AcquireNor = 5,
        Reduction = 6,
    };

    Operation operation{};
    GPUVAddr address{};
    u64 payload{};
    bool payload_64{};
    bool timestamp{};
    std::optional<SemaphoreReduction> reduction;
    bool reduction_signed{};

    [[nodiscard]] static HostSemaphore Decode(u32 address_lo, u32 address_hi, u32 payload_lo,
                                              u32 payload_hi, u32 execute);

    [[nodiscard]] bool IsAcquire() const;

    /// Whether an acquire is satisfied by the current semaphore value
    [[nodiscard]] bool IsAcquired(u64 value) const;

    [[nodiscard]] SemaphoreRelease ToRelease() const;
};

/// REPORT_SEMAPHORE_EXECUTE of the 3D and compute classes, with
/// SET_REPORT_SEMAPHORE_PAYLOAD_LOWER/UPPER and SET_REPORT_SEMAPHORE_ADDRESS_LOWER/UPPER before it
struct ReportSemaphore {
    enum class Operation : u32 {
        Release = 0,
        Acquire = 1,
        ReportOnly = 2,
        Trap = 3,
    };

    Operation operation{};
    /// Counter to report, numbered as VideoCommon::QueryType; zero reports the payload
    u32 report{};
    u32 sub_report{};
    /// One or two words, without a timestamp
    bool is_short{};
    SemaphoreRelease release;

    [[nodiscard]] static ReportSemaphore Decode(u32 payload_lower, u32 payload_upper,
                                                u32 address_lower, u32 address_upper,
                                                u32 execute);

    /// Performs a release or a report; acquires and traps are not implemented
    void Execute(VideoCore::RasterizerInterface& rasterizer, MemoryManager& memory_manager) const;
};

} // namespace Tegra::Engines
