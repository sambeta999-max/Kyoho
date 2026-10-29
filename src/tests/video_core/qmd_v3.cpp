// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "common/common_types.h"
#include "video_core/engines/qmd_v3.h"

using Tegra::Engines::QmdV3;

namespace {

// Writes bits [lo, hi] as NVC7C0_QMDV03_00 MW(hi:lo) describes them
void Set(QmdV3& qmd, u32 hi, u32 lo, u64 value) {
    for (u32 bit = lo; bit <= hi; ++bit) {
        const u32 mask = 1U << (bit % 32);
        u32& word = qmd.words[bit / 32];
        word = ((value >> (bit - lo)) & 1) != 0 ? word | mask : word & ~mask;
    }
}

} // Anonymous namespace

TEST_CASE("QMD 3.0: launch parameters", "[video_core][qmd]") {
    QmdV3 qmd;
    Set(qmd, 583, 580, 3);                 // QMD_MAJOR_VERSION
    Set(qmd, 157, 142, 2);                 // DEPENDENCE_COUNTER
    Set(qmd, 382, 382, 1);                 // SAMPLER_INDEX_VIA_HEADER_INDEX
    Set(qmd, 415, 384, 40);                // CTA_RASTER_WIDTH
    Set(qmd, 431, 416, 30);                // CTA_RASTER_HEIGHT
    Set(qmd, 463, 448, 2);                 // CTA_RASTER_DEPTH
    Set(qmd, 561, 544, 0x8000);            // SHARED_MEMORY_SIZE
    Set(qmd, 607, 592, 64);                // CTA_THREAD_DIMENSION0
    Set(qmd, 623, 608, 4);                 // CTA_THREAD_DIMENSION1
    Set(qmd, 639, 624, 1);                 // CTA_THREAD_DIMENSION2
    Set(qmd, 640 + 3, 640 + 3, 1);         // CONSTANT_BUFFER_VALID(3)
    Set(qmd, 1055 + 192, 1024 + 192, 0x12345600); // CONSTANT_BUFFER_ADDR_LOWER(3)
    Set(qmd, 1072 + 192, 1056 + 192, 0x7f);       // CONSTANT_BUFFER_ADDR_UPPER(3)
    Set(qmd, 1087 + 192, 1075 + 192, 0x100);      // CONSTANT_BUFFER_SIZE_SHIFTED4(3)
    Set(qmd, 759, 736, 0x800);             // SHADER_LOCAL_MEMORY_LOW_SIZE
    Set(qmd, 1623, 1600, 0x400);           // SHADER_LOCAL_MEMORY_HIGH_SIZE
    Set(qmd, 767, 763, 1);                 // BARRIER_COUNT
    Set(qmd, 1567, 1536, 0x89abcd00);      // PROGRAM_ADDRESS_LOWER
    Set(qmd, 1584, 1568, 0x12);            // PROGRAM_ADDRESS_UPPER

    REQUIRE(qmd.MajorVersion() == 3);
    REQUIRE(qmd.DependenceCounter() == 2);
    REQUIRE(qmd.HasWork());
    REQUIRE(qmd.ProgramAddress() == 0x1289abcd00ULL);

    const auto params = qmd.ToLaunchParams();
    REQUIRE(params.program_start == 0);
    REQUIRE(params.linked_tsc == 1);
    REQUIRE(params.grid_dim_x == 40);
    REQUIRE(params.grid_dim_y == 30);
    REQUIRE(params.grid_dim_z == 2);
    REQUIRE(params.shared_alloc == 0x8000);
    REQUIRE(params.block_dim_x == 64);
    REQUIRE(params.block_dim_y == 4);
    REQUIRE(params.block_dim_z == 1);
    REQUIRE(params.const_buffer_enable_mask == 0b1000);
    REQUIRE(params.const_buffer_config[3].Address() == 0x7f12345600ULL);
    REQUIRE(params.const_buffer_config[3].size == 0x1000);
    REQUIRE(params.local_pos_alloc == 0x800);
    REQUIRE(params.local_neg_alloc == 0x400);
    REQUIRE(params.barrier_alloc == 1);

    Set(qmd, 463, 448, 0);
    REQUIRE(!qmd.HasWork());
}

TEST_CASE("QMD 3.0: releases and dependent QMD", "[video_core][qmd]") {
    QmdV3 qmd;
    REQUIRE(!qmd.GetRelease(0));
    REQUIRE(!qmd.GetRelease(1));
    REQUIRE(!qmd.GetDependent());

    Set(qmd, 799, 768, 0xdead0000);        // RELEASE0_ADDRESS_LOWER
    Set(qmd, 807, 800, 0x01);              // RELEASE0_ADDRESS_UPPER
    Set(qmd, 823, 823, 1);                 // RELEASE0_ENABLE
    Set(qmd, 831, 830, 1);                 // RELEASE0_STRUCTURE_SIZE_SEMAPHORE_ONE_WORD
    Set(qmd, 863, 832, 7);                 // RELEASE0_PAYLOAD_LOWER

    Set(qmd, 927, 896, 0xbeef0000);        // RELEASE1_ADDRESS_LOWER
    Set(qmd, 951, 951, 1);                 // RELEASE1_ENABLE
    Set(qmd, 950, 948, 6);                 // RELEASE1_REDUCTION_OP_RED_OR
    Set(qmd, 953, 952, 1);                 // RELEASE1_REDUCTION_FORMAT_SIGNED
    Set(qmd, 954, 954, 1);                 // RELEASE1_REDUCTION_ENABLE
    Set(qmd, 957, 957, 1);                 // RELEASE1_PAYLOAD64B
    Set(qmd, 991, 960, 0x11);              // RELEASE1_PAYLOAD_LOWER
    Set(qmd, 1023, 992, 0x22);             // RELEASE1_PAYLOAD_UPPER

    Set(qmd, 511, 480, 0x123456);          // DEPENDENT_QMD0_POINTER
    Set(qmd, 512, 512, 1);                 // DEPENDENT_QMD0_ENABLE
    Set(qmd, 515, 513, 4);                 // DEPENDENT_QMD0_ACTION_QMD_DECREMENT_DEPENDENCE

    const auto first = qmd.GetRelease(0);
    REQUIRE(first);
    REQUIRE(first->address == 0x1dead0000ULL);
    REQUIRE(first->payload == 7);
    REQUIRE(!first->payload_64);
    REQUIRE(first->structure_size == 1);
    REQUIRE(!first->reduction);

    const auto second = qmd.GetRelease(1);
    REQUIRE(second);
    REQUIRE(second->address == 0xbeef0000ULL);
    REQUIRE(second->payload == 0x2200000011ULL);
    REQUIRE(second->payload_64);
    REQUIRE(second->reduction == QmdV3::ReductionOp::Or);
    REQUIRE(second->reduction_signed);

    const auto dependent = qmd.GetDependent();
    REQUIRE(dependent);
    REQUIRE(dependent->address == 0x12345600ULL);
    REQUIRE(dependent->action == QmdV3::DependentAction::DecrementDependence);
    REQUIRE(!qmd.HasSecondDependent());
}
