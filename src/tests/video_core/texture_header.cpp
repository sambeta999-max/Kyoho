// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "common/common_types.h"
#include "video_core/textures/texture.h"

using Tegra::Texture::FromAmpereTIC;
using Tegra::Texture::TextureType;
using Tegra::Texture::TICEntry;
using Tegra::Texture::TICHeaderVersion;

namespace {

TICEntry Header(TICHeaderVersion version) {
    TICEntry entry{};
    entry.header_version.Assign(version);
    entry.texture_type.Assign(TextureType::Texture2DArray);
    entry.address_high.Assign(0x81);
    entry.address_low = 0x00870000;
    entry.width_minus_one.Assign(1919);
    entry.height_minus_1.Assign(1079);
    entry.depth_minus_1.Assign(5);
    return entry;
}

} // Anonymous namespace

TEST_CASE("Ampere texture header: no layer base", "[video_core][texture_header]") {
    // Bits 93 to 95 are reserved in the Ampere header; Maxwell kept layer_base_8_10 there
    TICEntry entry = Header(TICHeaderVersion::BlockLinear);
    entry.layer_base_8_10.Assign(1);
    const TICEntry converted = FromAmpereTIC(entry);
    REQUIRE(converted.BaseLayer() == 0);
    REQUIRE(converted.Address() == 0x8100870000ULL);
    REQUIRE(converted.Width() == 1920);
    REQUIRE(converted.Height() == 1080);
    REQUIRE(converted.Depth() == 6);
    REQUIRE(converted.texture_type == TextureType::Texture2DArray);
}

TEST_CASE("Ampere texture header: widened fields", "[video_core][texture_header]") {
    // WIDTH_MINUS_ONE bit 16 (bit 144) is layer_base_0_2 bit 0 in the Maxwell layout
    TICEntry wide = Header(TICHeaderVersion::BlockLinear);
    wide.layer_base_0_2.Assign(1);
    REQUIRE(FromAmpereTIC(wide).Address() == 0);

    // HEIGHT_MINUS_ONE bit 16 (bit 146)
    TICEntry tall = Header(TICHeaderVersion::Pitch);
    tall.layer_base_0_2.Assign(4);
    REQUIRE(FromAmpereTIC(tall).Address() == 0);

    // ADDRESS bit 48 (bit 80) is layer_base_3_7 bit 0 in the Maxwell layout
    TICEntry far = Header(TICHeaderVersion::BlockLinear);
    far.layer_base_3_7.Assign(1);
    REQUIRE(FromAmpereTIC(far).Address() == 0);

    // Bits 144 to 146 are reserved in 1D buffer headers
    TICEntry buffer = Header(TICHeaderVersion::OneDBuffer);
    buffer.layer_base_0_2.Assign(7);
    const TICEntry converted = FromAmpereTIC(buffer);
    REQUIRE(converted.Address() == 0x8100870000ULL);
    REQUIRE(converted.layer_base_0_2 == 0);
}
