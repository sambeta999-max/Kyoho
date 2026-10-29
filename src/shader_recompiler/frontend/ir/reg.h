// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <fmt/ranges.h>

#include "common/common_types.h"
#include "shader_recompiler/exception.h"

namespace Shader::IR {

enum class Reg : u64 {
    R0,
    R1,
    R2,
    R3,
    R4,
    R5,
    R6,
    R7,
    R8,
    R9,
    R10,
    R11,
    R12,
    R13,
    R14,
    R15,
    R16,
    R17,
    R18,
    R19,
    R20,
    R21,
    R22,
    R23,
    R24,
    R25,
    R26,
    R27,
    R28,
    R29,
    R30,
    R31,
    R32,
    R33,
    R34,
    R35,
    R36,
    R37,
    R38,
    R39,
    R40,
    R41,
    R42,
    R43,
    R44,
    R45,
    R46,
    R47,
    R48,
    R49,
    R50,
    R51,
    R52,
    R53,
    R54,
    R55,
    R56,
    R57,
    R58,
    R59,
    R60,
    R61,
    R62,
    R63,
    R64,
    R65,
    R66,
    R67,
    R68,
    R69,
    R70,
    R71,
    R72,
    R73,
    R74,
    R75,
    R76,
    R77,
    R78,
    R79,
    R80,
    R81,
    R82,
    R83,
    R84,
    R85,
    R86,
    R87,
    R88,
    R89,
    R90,
    R91,
    R92,
    R93,
    R94,
    R95,
    R96,
    R97,
    R98,
    R99,
    R100,
    R101,
    R102,
    R103,
    R104,
    R105,
    R106,
    R107,
    R108,
    R109,
    R110,
    R111,
    R112,
    R113,
    R114,
    R115,
    R116,
    R117,
    R118,
    R119,
    R120,
    R121,
    R122,
    R123,
    R124,
    R125,
    R126,
    R127,
    R128,
    R129,
    R130,
    R131,
    R132,
    R133,
    R134,
    R135,
    R136,
    R137,
    R138,
    R139,
    R140,
    R141,
    R142,
    R143,
    R144,
    R145,
    R146,
    R147,
    R148,
    R149,
    R150,
    R151,
    R152,
    R153,
    R154,
    R155,
    R156,
    R157,
    R158,
    R159,
    R160,
    R161,
    R162,
    R163,
    R164,
    R165,
    R166,
    R167,
    R168,
    R169,
    R170,
    R171,
    R172,
    R173,
    R174,
    R175,
    R176,
    R177,
    R178,
    R179,
    R180,
    R181,
    R182,
    R183,
    R184,
    R185,
    R186,
    R187,
    R188,
    R189,
    R190,
    R191,
    R192,
    R193,
    R194,
    R195,
    R196,
    R197,
    R198,
    R199,
    R200,
    R201,
    R202,
    R203,
    R204,
    R205,
    R206,
    R207,
    R208,
    R209,
    R210,
    R211,
    R212,
    R213,
    R214,
    R215,
    R216,
    R217,
    R218,
    R219,
    R220,
    R221,
    R222,
    R223,
    R224,
    R225,
    R226,
    R227,
    R228,
    R229,
    R230,
    R231,
    R232,
    R233,
    R234,
    R235,
    R236,
    R237,
    R238,
    R239,
    R240,
    R241,
    R242,
    R243,
    R244,
    R245,
    R246,
    R247,
    R248,
    R249,
    R250,
    R251,
    R252,
    R253,
    R254,
    RZ,
    // Uniform datapath registers (SM75+)
    UR0,
    UR1,
    UR2,
    UR3,
    UR4,
    UR5,
    UR6,
    UR7,
    UR8,
    UR9,
    UR10,
    UR11,
    UR12,
    UR13,
    UR14,
    UR15,
    UR16,
    UR17,
    UR18,
    UR19,
    UR20,
    UR21,
    UR22,
    UR23,
    UR24,
    UR25,
    UR26,
    UR27,
    UR28,
    UR29,
    UR30,
    UR31,
    UR32,
    UR33,
    UR34,
    UR35,
    UR36,
    UR37,
    UR38,
    UR39,
    UR40,
    UR41,
    UR42,
    UR43,
    UR44,
    UR45,
    UR46,
    UR47,
    UR48,
    UR49,
    UR50,
    UR51,
    UR52,
    UR53,
    UR54,
    UR55,
    UR56,
    UR57,
    UR58,
    UR59,
    UR60,
    UR61,
    UR62,
    URZ,
};
static_assert(static_cast<int>(Reg::RZ) == 255);
static_assert(static_cast<int>(Reg::URZ) == 319);

constexpr size_t NUM_USER_REGS = 255;
constexpr size_t NUM_USER_UNIFORM_REGS = 63;
constexpr size_t NUM_REGS = 320;

[[nodiscard]] constexpr bool IsUniform(Reg reg) noexcept {
    return reg > Reg::RZ;
}

[[nodiscard]] constexpr bool IsZero(Reg reg) noexcept {
    return reg == Reg::RZ || reg == Reg::URZ;
}

/// Uniform register from its encoded index, where 63 is URZ
[[nodiscard]] constexpr Reg UniformReg(u32 index) noexcept {
    return static_cast<Reg>(static_cast<u32>(Reg::UR0) + (index < 63 ? index : 63));
}

[[nodiscard]] constexpr Reg operator+(Reg reg, int num) {
    if (IsZero(reg)) {
        // Adding or subtracting registers from RZ or URZ yields the same zero register
        return reg;
    }
    const int first{IsUniform(reg) ? static_cast<int>(Reg::UR0) : 0};
    const int zero{static_cast<int>(IsUniform(reg) ? Reg::URZ : Reg::RZ)};
    const int result{static_cast<int>(reg) + num};
    if (result >= zero) {
        throw LogicError("Overflow on register arithmetic");
    }
    if (result < first) {
        throw LogicError("Underflow on register arithmetic");
    }
    return static_cast<Reg>(result);
}

[[nodiscard]] constexpr Reg operator-(Reg reg, int num) {
    return reg + (-num);
}

constexpr Reg operator++(Reg& reg) {
    reg = reg + 1;
    return reg;
}

constexpr Reg operator++(Reg& reg, int) {
    const Reg copy{reg};
    reg = reg + 1;
    return copy;
}

[[nodiscard]] constexpr size_t RegIndex(Reg reg) noexcept {
    return static_cast<size_t>(reg);
}

[[nodiscard]] constexpr bool IsAligned(Reg reg, size_t align) {
    const size_t index{IsUniform(reg) ? RegIndex(reg) - RegIndex(Reg::UR0) : RegIndex(reg)};
    return index % align == 0 || IsZero(reg);
}

} // namespace Shader::IR

template <>
struct fmt::formatter<Shader::IR::Reg> {
    constexpr auto parse(format_parse_context& ctx) {
        return ctx.begin();
    }
    template <typename FormatContext>
    auto format(const Shader::IR::Reg& reg, FormatContext& ctx) const {
        if (reg == Shader::IR::Reg::RZ) {
            return fmt::format_to(ctx.out(), "RZ");
        } else if (reg == Shader::IR::Reg::URZ) {
            return fmt::format_to(ctx.out(), "URZ");
        } else if (static_cast<int>(reg) >= 0 && static_cast<int>(reg) < 255) {
            return fmt::format_to(ctx.out(), "R{}", static_cast<int>(reg));
        } else if (Shader::IR::IsUniform(reg)) {
            return fmt::format_to(ctx.out(), "UR{}",
                                  static_cast<int>(reg) - static_cast<int>(Shader::IR::Reg::UR0));
        } else {
            throw Shader::LogicError("Invalid register with raw value {}", static_cast<int>(reg));
        }
    }
};
