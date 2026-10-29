// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <catch2/catch_test_macros.hpp>
#include "core/hle/kernel/k_memory_block.h"
#include "core/hle/service/nvdrv/devices/native_gpu_kind.h"
#include "core/hle/service/nvdrv/devices/native_gpu_context.h"
#include "core/hle/service/nvdrv/devices/native_gpu_characteristics.h"
#include "core/hle/service/nvdrv/devices/gpu_address_space_layout.h"

TEST_CASE("Native GPU characteristics replace topology with ioctl interface levels",
          "[native-gpu]") {
    std::array<u8, 0xa0> legacy;
    legacy.fill(0xcc);
    const u32_le arch{0x120};
    const u32_le impl{0xb};
    std::memcpy(legacy.data(), &arch, sizeof(arch));
    std::memcpy(legacy.data() + 4, &impl, sizeof(impl));
    const u32_le available{0x30000};
    std::memcpy(legacy.data() + 0x34, &available, sizeof(available));
    const auto native = Service::Nvidia::Devices::MakeNativeGpuCharacteristics(legacy);
    REQUIRE(native.size() == 0xd0);
    u32_le native_arch, native_impl;
    std::memcpy(&native_arch, native.data(), sizeof(native_arch));
    std::memcpy(&native_impl, native.data() + 4, sizeof(native_impl));
    REQUIRE(static_cast<u32>(native_arch) == 0x170);
    REQUIRE(static_cast<u32>(native_impl) == 0xe);
    REQUIRE(std::equal(legacy.begin() + 8, legacy.begin() + 0x28, native.begin() + 8));
    REQUIRE(std::equal(legacy.begin() + 0x2c, legacy.begin() + 0x34, native.begin() + 0x2c));
    REQUIRE(std::equal(legacy.begin() + 0x38, legacy.begin() + 0x3c, native.begin() + 0x38));
    REQUIRE(std::equal(legacy.begin() + 0x44, legacy.begin() + 0x50, native.begin() + 0x44));
    REQUIRE(std::equal(legacy.begin() + 0x58, legacy.begin() + 0x70, native.begin() + 0x58));
    u32_le sm_version, spa_version;
    std::memcpy(&sm_version, native.data() + 0x3c, sizeof(sm_version));
    std::memcpy(&spa_version, native.data() + 0x40, sizeof(spa_version));
    REQUIRE(static_cast<u32>(sm_version) == 0x807);
    REQUIRE(static_cast<u32>(spa_version) == 0x807);
    u32_le available_sizes;
    std::memcpy(&available_sizes, native.data() + 0x34, sizeof(available_sizes));
    REQUIRE(static_cast<u32>(available_sizes) == 0x11000);
    u32_le big_page_size;
    std::memcpy(&big_page_size, native.data() + 0x28, sizeof(big_page_size));
    REQUIRE(static_cast<u32>(big_page_size) == 0x1000);
    const auto level = [&](size_t offset) {
        u16_le value;
        std::memcpy(&value, native.data() + offset, sizeof(value));
        return static_cast<s16>(static_cast<u16>(value));
    };
    REQUIRE(level(0x70) == 0x1c);
    REQUIRE(level(0x72) == 2);
    u64_le flags;
    std::memcpy(&flags, native.data() + 0x50, sizeof(flags));
    REQUIRE((static_cast<u64>(flags) & (1ULL << 8)) != 0);
    REQUIRE((static_cast<u64>(flags) & ~((1ULL << 8) | (1ULL << 26) | (1ULL << 34) | (0xfULL << 40))) == (0xccccccccccccccccULL & ~((1ULL << 8) | (1ULL << 26) | (1ULL << 34) | (0xfULL << 40))));
    REQUIRE((static_cast<u64>(flags) & (0xfULL << 40)) == (0xfULL << 40));
    REQUIRE((static_cast<u64>(flags) & (1ULL << 34)) != 0);
    REQUIRE(level(0x74) == -1);
    REQUIRE(level(0x76) == 0x1d);
    // The captured SDK rejects the old topology value 2 here, requiring >= 6.
    REQUIRE(level(0x78) == 0x14);
    REQUIRE(std::all_of(native.begin() + 0x7a, native.end(), [](u8 b) { return b == 0; }));
}

TEST_CASE("Native GPU VA layout accommodates the captured NVN bootstrap reservation",
          "[native-gpu]") {
    using Service::Nvidia::Devices::GetGpuAddressSpaceLayout;
    constexpr auto native = GetGpuAddressSpaceLayout(true);
    constexpr auto legacy = GetGpuAddressSpaceLayout(false);
    REQUIRE(native.split - native.start >= 0x7ff8000000ULL + (1ULL << 32));
    REQUIRE(native.start == legacy.start);
    REQUIRE(native.end == 1ULL << 40);
    REQUIRE(native.end > native.split);
    REQUIRE(legacy.split == 1ULL << 34);
    REQUIRE(legacy.end == 1ULL << 37);
}

TEST_CASE("GPU region addresses retain bits above the 32-bit page index", "[native-gpu]") {
    using Service::Nvidia::Devices::GpuPageAddress;
    REQUIRE(GpuPageAddress(0x400000, 17) == 0x8000000000ULL);
    REQUIRE(GpuPageAddress(0x20000, 17) == 0x400000000ULL);
    REQUIRE(GpuPageAddress(0x8000, 12) == 0x8000000ULL);
}

TEST_CASE("GPU memory eligibility survives splitting and remains separate from cache attributes",
          "[native-gpu]") {
    using namespace Kernel;
    KMemoryBlock right{KProcessAddress{0x10000}, 4, KMemoryState::Normal,
                       KMemoryPermission::UserReadWrite, KMemoryAttribute::Uncached};
    right.UpdateAttribute(KMemoryAttribute::GpuAccessible, KMemoryAttribute::GpuAccessible);
    KMemoryBlock left;
    right.Split(&left, KProcessAddress{0x12000});
    REQUIRE(left.GetNumPages() == 2);
    REQUIRE(right.GetNumPages() == 2);
    REQUIRE(left.GetMemoryInfo().GetSvcMemoryInfo().attribute ==
            (Svc::MemoryAttribute::Uncached | Svc::MemoryAttribute::GpuAccessible));
    REQUIRE(right.GetAttribute() == left.GetAttribute());
    right.UpdateAttribute(KMemoryAttribute::GpuAccessible, KMemoryAttribute::None);
    REQUIRE(right.GetAttribute() == KMemoryAttribute::Uncached);
    REQUIRE(left.GetAttribute() ==
            (KMemoryAttribute::Uncached | KMemoryAttribute::GpuAccessible));
    REQUIRE(right.GetPermission() == KMemoryPermission::UserReadWrite);
    REQUIRE(right.GetState() == KMemoryState::Normal);
}

TEST_CASE("Native 4 KiB allocation preference retains a separate large-page GPU pool", "[native-gpu]") {
    using Service::Nvidia::Devices::GetGpuAddressSpaceBigPageSize;
    REQUIRE(GetGpuAddressSpaceBigPageSize(true, 0x1000) == 0x10000);
    REQUIRE(GetGpuAddressSpaceBigPageSize(true, 0x20000) == 0x20000);
    REQUIRE(GetGpuAddressSpaceBigPageSize(false, 0x1000) == 0x1000);
    REQUIRE(GetGpuAddressSpaceBigPageSize(false, 0) == 0);
}

TEST_CASE("Native generic memory kind is translated rather than treated as legacy depth memory",
          "[native-gpu]") {
    using Service::Nvidia::Devices::TranslateNativeUncompressedKind;
    REQUIRE(TranslateNativeUncompressedKind(0, 0) == static_cast<u8>(Tegra::PTEKind::PITCH));
    REQUIRE(TranslateNativeUncompressedKind(6, 6) == static_cast<u8>(Tegra::PTEKind::GENERIC_16BX2));
    REQUIRE(TranslateNativeUncompressedKind(6, 6) != static_cast<u8>(Tegra::PTEKind::Z16_MS16_2C));
    REQUIRE_FALSE(TranslateNativeUncompressedKind(8, 6));
    REQUIRE_FALSE(TranslateNativeUncompressedKind(7, 7));
    REQUIRE_FALSE(TranslateNativeUncompressedKind(0xfe, 0xfe));
}

TEST_CASE("Native mappings report the actual selected memory kind", "[native-gpu]") {
    using Service::Nvidia::Devices::SelectNativeMemoryKind;
    const auto normal = SelectNativeMemoryKind(6, 6);
    REQUIRE(normal.has_value());
    REQUIRE(normal->selected == 6);
    REQUIRE(normal->backend == 0xfe);
    const auto fallback = SelectNativeMemoryKind(8, 6);
    REQUIRE(fallback.has_value());
    REQUIRE(fallback->selected == 6);
    REQUIRE(fallback->backend == normal->backend);
    REQUIRE(SelectNativeMemoryKind(0, 0)->selected == 0);
    const auto depth = SelectNativeMemoryKind(0xc, 3);
    REQUIRE(depth.has_value());
    REQUIRE(depth->selected == 3);
    REQUIRE(depth->backend == static_cast<u8>(Tegra::PTEKind::S8Z24));
    REQUIRE(SelectNativeMemoryKind(3, 3)->backend == depth->backend);
    const auto depth16 = SelectNativeMemoryKind(0xb, 1);
    REQUIRE(depth16.has_value());
    REQUIRE(depth16->selected == 1);
    REQUIRE(depth16->backend == static_cast<u8>(Tegra::PTEKind::Z16));
    REQUIRE(SelectNativeMemoryKind(1, 1)->backend == depth16->backend);
    REQUIRE_FALSE(SelectNativeMemoryKind(0xb, 6));
    REQUIRE_FALSE(SelectNativeMemoryKind(0xc, 6));
    REQUIRE_FALSE(SelectNativeMemoryKind(8, 8));
    REQUIRE_FALSE(SelectNativeMemoryKind(8, 0));
    REQUIRE_FALSE(SelectNativeMemoryKind(9, 6));
}

TEST_CASE("Native GPU context allocations classify explicit known engine identities", "[native-gpu]") {
    using Service::Nvidia::Devices::GetNativeGpuContextSlot;
    REQUIRE(GetNativeGpuContextSlot(0xc797) == 1U);
    REQUIRE(GetNativeGpuContextSlot(0xc7c0) == 2U);
    REQUIRE_FALSE(GetNativeGpuContextSlot(0xa143));
    REQUIRE(GetNativeGpuContextSlot(0xc7b5) == 4U);
    REQUIRE(GetNativeGpuContextSlot(0xc56f) == 5U);
    REQUIRE_FALSE(GetNativeGpuContextSlot(0));
    REQUIRE_FALSE(GetNativeGpuContextSlot(0xb197));
    REQUIRE_FALSE(GetNativeGpuContextSlot(0xffffc797));
    REQUIRE_FALSE(GetNativeGpuContextSlot(0xdeadbeef));
}
