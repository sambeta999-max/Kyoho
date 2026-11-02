// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>

#include "common/logging.h"
#include "video_core/engines/semaphore.h"
#include "video_core/memory_manager.h"
#include "video_core/query_cache/types.h"
#include "video_core/rasterizer_interface.h"

namespace Tegra::Engines {
namespace {

constexpr u32 Bits(u32 value, u32 lo, u32 count) {
    return (value >> lo) & ((1U << count) - 1);
}

constexpr GPUVAddr Address(u32 lower, u32 upper) {
    return (GPUVAddr{upper & 0xFF} << 32) | lower;
}

} // Anonymous namespace

u64 ReduceSemaphore(SemaphoreReduction op, u64 old_value, u64 value, bool is_signed, bool is_64) {
    const u64 mask = is_64 ? ~u64{0} : u64{0xFFFFFFFF};
    const auto as_signed = [is_64](u64 x) {
        return is_64 ? static_cast<s64>(x) : s64{static_cast<s32>(static_cast<u32>(x))};
    };
    old_value &= mask;
    value &= mask;
    switch (op) {
    case SemaphoreReduction::Add:
        return (old_value + value) & mask;
    case SemaphoreReduction::Min:
        return is_signed ? (as_signed(old_value) < as_signed(value) ? old_value : value)
                         : (std::min)(old_value, value);
    case SemaphoreReduction::Max:
        return is_signed ? (as_signed(old_value) > as_signed(value) ? old_value : value)
                         : (std::max)(old_value, value);
    case SemaphoreReduction::Inc:
        return old_value >= value ? 0 : old_value + 1;
    case SemaphoreReduction::Dec:
        return old_value == 0 || old_value > value ? value : old_value - 1;
    case SemaphoreReduction::And:
        return old_value & value;
    case SemaphoreReduction::Or:
        return old_value | value;
    case SemaphoreReduction::Xor:
        return old_value ^ value;
    }
    return old_value;
}

void ReleaseSemaphore(VideoCore::RasterizerInterface& rasterizer, MemoryManager& memory_manager,
                      const SemaphoreRelease& release) {
    // The query cache writes a 32-bit payload, or with a timestamp the payload zero extended to
    // 64 bits followed by the timestamp
    const bool fits_query = !release.payload_64 || (release.timestamp && (release.payload >> 32) == 0);
    if (!release.reduction && fits_query) {
        VideoCommon::QueryPropertiesFlags flags{VideoCommon::QueryPropertiesFlags::IsAFence};
        if (release.timestamp) {
            flags |= VideoCommon::QueryPropertiesFlags::HasTimeout;
        }
        rasterizer.Query(release.address, VideoCommon::QueryType::Payload, flags,
                         static_cast<u32>(release.payload), 0);
        return;
    }
    if (release.timestamp) {
        LOG_WARNING(HW_GPU, "Semaphore release at {:#x} without its timestamp", release.address);
    }
    // What the release will leave, for acquires that come before the host GPU reaches it
    u64 pending_value = release.payload;
    if (release.reduction) {
        pending_value = ReduceSemaphore(
            *release.reduction, memory_manager.ReadSemaphore(release.address, release.payload_64),
            pending_value, release.reduction_signed, release.payload_64);
    }
    const u64 ticket = memory_manager.SetPendingSemaphore(release.address, pending_value);
    rasterizer.SignalFence([&memory_manager, release, ticket] {
        u64 value = release.payload;
        if (release.reduction) {
            const u64 old_value = release.payload_64 ? memory_manager.Read<u64>(release.address)
                                                     : memory_manager.Read<u32>(release.address);
            value = ReduceSemaphore(*release.reduction, old_value, value, release.reduction_signed,
                                    release.payload_64);
        }
        if (release.payload_64) {
            memory_manager.Write<u64>(release.address, value);
        } else {
            memory_manager.Write<u32>(release.address, static_cast<u32>(value));
        }
        memory_manager.ClearPendingSemaphore(release.address, ticket);
    });
}

HostSemaphore HostSemaphore::Decode(u32 address_lo, u32 address_hi, u32 payload_lo,
                                    u32 payload_hi, u32 execute) {
    // REDUCTION at [27..31) in the host order IMIN, IMAX, IXOR, IAND, IOR, IADD, INC, DEC
    static constexpr std::array<SemaphoreReduction, 8> HOST_REDUCTIONS{
        SemaphoreReduction::Min, SemaphoreReduction::Max, SemaphoreReduction::Xor,
        SemaphoreReduction::And, SemaphoreReduction::Or,  SemaphoreReduction::Add,
        SemaphoreReduction::Inc, SemaphoreReduction::Dec,
    };
    HostSemaphore semaphore{
        .operation = static_cast<Operation>(Bits(execute, 0, 3)),
        .address = Address(address_lo & ~3U, address_hi),
        .payload = (u64{payload_hi} << 32) | payload_lo,
        .payload_64 = Bits(execute, 24, 1) != 0,
        .timestamp = Bits(execute, 25, 1) != 0,
        .reduction = std::nullopt,
        // REDUCTION_FORMAT is zero for signed
        .reduction_signed = Bits(execute, 31, 1) == 0,
    };
    if (semaphore.operation == Operation::Reduction) {
        const u32 op = Bits(execute, 27, 4);
        if (op < HOST_REDUCTIONS.size()) {
            semaphore.reduction = HOST_REDUCTIONS[op];
        } else {
            LOG_ERROR(HW_GPU, "Unknown host semaphore reduction {}", op);
        }
    }
    if (!semaphore.payload_64) {
        semaphore.payload &= 0xFFFFFFFF;
    }
    return semaphore;
}

bool HostSemaphore::IsAcquire() const {
    switch (operation) {
    case Operation::Acquire:
    case Operation::AcquireStrictGeq:
    case Operation::AcquireCircGeq:
    case Operation::AcquireAnd:
    case Operation::AcquireNor:
        return true;
    default:
        return false;
    }
}

bool HostSemaphore::IsAcquired(u64 value) const {
    const u64 mask = payload_64 ? ~u64{0} : u64{0xFFFFFFFF};
    value &= mask;
    switch (operation) {
    case Operation::Acquire:
        return value == payload;
    case Operation::AcquireStrictGeq:
        return value >= payload;
    case Operation::AcquireCircGeq:
        // Wrapping comparison: the difference is non-negative in the payload width
        return payload_64 ? static_cast<s64>(value - payload) >= 0
                          : static_cast<s32>(static_cast<u32>(value - payload)) >= 0;
    case Operation::AcquireAnd:
        return (value & payload) != 0;
    case Operation::AcquireNor:
        return (~(value | payload) & mask) != 0;
    default:
        return true;
    }
}

SemaphoreRelease HostSemaphore::ToRelease() const {
    return {
        .address = address,
        .payload = payload,
        .payload_64 = payload_64,
        .timestamp = timestamp,
        .reduction = reduction,
        .reduction_signed = reduction_signed,
    };
}

ReportSemaphore ReportSemaphore::Decode(u32 payload_lower, u32 payload_upper, u32 address_lower,
                                        u32 address_upper, u32 execute) {
    const u32 structure_size = Bits(execute, 13, 2);
    ReportSemaphore semaphore{
        .operation = static_cast<Operation>(Bits(execute, 0, 2)),
        .report = Bits(execute, 7, 5),
        .sub_report = Bits(execute, 15, 3),
        .is_short = structure_size != 0,
        .release =
            {
                .address = Address(address_lower, address_upper),
                .payload = (u64{payload_upper} << 32) | payload_lower,
                .payload_64 = Bits(execute, 27, 1) != 0,
                .timestamp = structure_size == 0,
                .reduction = std::nullopt,
                .reduction_signed = Bits(execute, 24, 2) == 1,
            },
    };
    if (Bits(execute, 20, 1) != 0) {
        semaphore.release.reduction = static_cast<SemaphoreReduction>(Bits(execute, 21, 3));
    }
    if (!semaphore.release.payload_64) {
        semaphore.release.payload &= 0xFFFFFFFF;
    }
    return semaphore;
}

void ReportSemaphore::Execute(VideoCore::RasterizerInterface& rasterizer,
                              MemoryManager& memory_manager) const {
    switch (operation) {
    case Operation::Release:
    case Operation::ReportOnly: {
        if (report == 0 && (release.reduction || release.payload_64)) {
            ReleaseSemaphore(rasterizer, memory_manager, release);
            return;
        }
        // As SET_REPORT_SEMAPHORE_D: long reports carry a timestamp, short releases are fences.
        // Two word counter reports are written as one word.
        VideoCommon::QueryPropertiesFlags flags{};
        if (!is_short) {
            flags |= VideoCommon::QueryPropertiesFlags::HasTimeout;
        } else if (operation == Operation::Release) {
            flags |= VideoCommon::QueryPropertiesFlags::IsAFence;
        }
        rasterizer.Query(release.address, static_cast<VideoCommon::QueryType>(report), flags,
                         static_cast<u32>(release.payload), sub_report);
        return;
    }
    case Operation::Acquire:
        LOG_ERROR(HW_GPU, "Unimplemented report semaphore acquire");
        return;
    case Operation::Trap:
        LOG_ERROR(HW_GPU, "Unimplemented report semaphore trap");
        return;
    }
}

} // namespace Tegra::Engines
