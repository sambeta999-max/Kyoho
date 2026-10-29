// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "common/common_types.h"
#include "video_core/engines/semaphore.h"

using Tegra::Engines::HostSemaphore;
using Tegra::Engines::ReduceSemaphore;
using Tegra::Engines::ReportSemaphore;
using Tegra::Engines::SemaphoreReduction;

TEST_CASE("Semaphore: reductions", "[video_core][semaphore]") {
    REQUIRE(ReduceSemaphore(SemaphoreReduction::Add, 0xFFFFFFFF, 2, false, false) == 1);
    REQUIRE(ReduceSemaphore(SemaphoreReduction::Add, 0xFFFFFFFF, 2, false, true) == 0x100000001);
    // -1 is the signed minimum of -1 and 1, but the unsigned maximum
    REQUIRE(ReduceSemaphore(SemaphoreReduction::Min, 0xFFFFFFFF, 1, true, false) == 0xFFFFFFFF);
    REQUIRE(ReduceSemaphore(SemaphoreReduction::Min, 0xFFFFFFFF, 1, false, false) == 1);
    REQUIRE(ReduceSemaphore(SemaphoreReduction::Max, 0xFFFFFFFF, 1, true, false) == 1);
    // INC wraps to zero at the payload, DEC wraps to the payload at zero
    REQUIRE(ReduceSemaphore(SemaphoreReduction::Inc, 3, 4, false, false) == 4);
    REQUIRE(ReduceSemaphore(SemaphoreReduction::Inc, 4, 4, false, false) == 0);
    REQUIRE(ReduceSemaphore(SemaphoreReduction::Dec, 0, 4, false, false) == 4);
    REQUIRE(ReduceSemaphore(SemaphoreReduction::Dec, 3, 4, false, false) == 2);
    REQUIRE(ReduceSemaphore(SemaphoreReduction::Xor, 0b1100, 0b1010, false, false) == 0b0110);
}

TEST_CASE("Semaphore: host SEM_EXECUTE", "[video_core][semaphore]") {
    // RELEASE, 64-bit payload, timestamp
    auto release = HostSemaphore::Decode(0x12345678, 0x1ab, 0x11, 0x22,
                                         1 | (1U << 24) | (1U << 25));
    REQUIRE(release.operation == HostSemaphore::Operation::Release);
    REQUIRE(!release.IsAcquire());
    // SEM_ADDR_LO holds the offset at [2..32), SEM_ADDR_HI eight bits
    REQUIRE(release.address == 0xab12345678);
    REQUIRE(release.payload == 0x2200000011);
    REQUIRE(release.ToRelease().payload_64);
    REQUIRE(release.ToRelease().timestamp);

    // A 32-bit payload ignores SEM_PAYLOAD_HI
    REQUIRE(HostSemaphore::Decode(0x1000, 0, 7, 0x22, 1).payload == 7);

    // REDUCTION IADD (5), unsigned format (bit 31)
    const auto add = HostSemaphore::Decode(0x1000, 0, 1, 0, 6 | (5U << 27) | (1U << 31));
    REQUIRE(add.reduction == SemaphoreReduction::Add);
    REQUIRE(!add.reduction_signed);
    // REDUCTION IMIN (0), signed format
    const auto min = HostSemaphore::Decode(0x1000, 0, 1, 0, 6);
    REQUIRE(min.reduction == SemaphoreReduction::Min);
    REQUIRE(min.reduction_signed);

    const auto acquire = HostSemaphore::Decode(0x1000, 0, 5, 0, 0);
    REQUIRE(acquire.IsAcquire());
    REQUIRE(acquire.IsAcquired(5));
    REQUIRE(!acquire.IsAcquired(6));

    const auto strict_geq = HostSemaphore::Decode(0x1000, 0, 5, 0, 2);
    REQUIRE(strict_geq.IsAcquired(6));
    REQUIRE(!strict_geq.IsAcquired(4));
    REQUIRE(strict_geq.IsAcquired(0xFFFFFFFF));

    // Circular: 2 is past 0xFFFFFFFE after wrapping, 0xFFFFFFFF is not past 2
    const auto circ_geq = HostSemaphore::Decode(0x1000, 0, 0xFFFFFFFE, 0, 3);
    REQUIRE(circ_geq.IsAcquired(2));
    REQUIRE(!HostSemaphore::Decode(0x1000, 0, 2, 0, 3).IsAcquired(0xFFFFFFFF));

    REQUIRE(HostSemaphore::Decode(0x1000, 0, 0b100, 0, 4).IsAcquired(0b110));
    REQUIRE(!HostSemaphore::Decode(0x1000, 0, 0b100, 0, 4).IsAcquired(0b010));
    REQUIRE(HostSemaphore::Decode(0x1000, 0, 0b100, 0, 5).IsAcquired(0b010));
    REQUIRE(!HostSemaphore::Decode(0x1000, 0, 0xFFFFFFFF, 0, 5).IsAcquired(0));
}

TEST_CASE("Semaphore: REPORT_SEMAPHORE_EXECUTE", "[video_core][semaphore]") {
    // RELEASE of the payload, one word
    const auto release = ReportSemaphore::Decode(42, 0, 0xdead0000, 0x01, 1U << 13);
    REQUIRE(release.operation == ReportSemaphore::Operation::Release);
    REQUIRE(release.report == 0);
    REQUIRE(release.is_short);
    REQUIRE(release.release.address == 0x1dead0000);
    REQUIRE(release.release.payload == 42);
    REQUIRE(!release.release.timestamp);
    REQUIRE(!release.release.reduction);

    // REPORT_ONLY of ZPASS_PIXEL_CNT (2), sub report 3, four words
    const auto report = ReportSemaphore::Decode(0, 0, 0x1000, 0, 2 | (2U << 7) | (3U << 15));
    REQUIRE(report.operation == ReportSemaphore::Operation::ReportOnly);
    REQUIRE(report.report == 2);
    REQUIRE(report.sub_report == 3);
    REQUIRE(!report.is_short);
    REQUIRE(report.release.timestamp);

    // RELEASE with a signed MAX reduction of a 64-bit payload, two words
    const auto reduction = ReportSemaphore::Decode(
        1, 2, 0x1000, 0, (2U << 13) | (1U << 20) | (2U << 21) | (1U << 24) | (1U << 27));
    REQUIRE(reduction.release.payload == 0x200000001);
    REQUIRE(reduction.release.payload_64);
    REQUIRE(reduction.release.reduction == SemaphoreReduction::Max);
    REQUIRE(reduction.release.reduction_signed);
    REQUIRE(!reduction.release.timestamp);

    // The upper payload word only counts with PAYLOAD_SIZE64
    REQUIRE(ReportSemaphore::Decode(1, 2, 0x1000, 0, 1U << 13).release.payload == 1);
}
