// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include "common/common_types.h"
#include "shader_recompiler/environment.h"
#include "shader_recompiler/frontend/ir/modifiers.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/object_pool.h"

namespace {

using namespace Shader;

constexpr u64 CONTROL{0x000fc00000000000ULL};
constexpr u64 RZ{255};
constexpr u64 PT{7};

// Encoding builder for the 128-bit SM86 format, field positions as accepted by nvdisasm
struct Encoding {
    explicit Encoding(u32 opcode, u64 pred = PT, bool pred_negated = false) {
        Set(0, 12, opcode);
        Set(12, 3, pred);
        Set(15, 1, pred_negated ? 1 : 0);
        hi |= CONTROL;
    }

    Encoding& Set(u32 offset, u32 count, u64 value) {
        for (u32 i = 0; i < count; ++i) {
            const u64 bit{(value >> i) & 1};
            const u32 position{offset + i};
            u64& word{position < 64 ? lo : hi};
            const u64 mask{u64{1} << (position % 64)};
            word = bit != 0 ? word | mask : word & ~mask;
        }
        return *this;
    }

    u64 lo{};
    u64 hi{};
};

Encoding MovImm(u64 dest, u32 value) {
    return Encoding{0x802}.Set(16, 8, dest).Set(32, 32, value).Set(72, 4, 0xf);
}

Encoding Exit() {
    return Encoding{0x94d}.Set(87, 3, PT);
}

class TestEnvironment final : public Environment {
public:
    explicit TestEnvironment(const std::vector<Encoding>& program) {
        stage = Stage::Fragment;
        is_ampere = true;
        sph.ps.omap.target = 0xf;
        for (const Encoding& insn : program) {
            code.push_back(insn.lo);
            code.push_back(insn.hi);
        }
    }

    // Raw instruction words, as compiled for a compute kernel
    explicit TestEnvironment(std::vector<u64> words, Stage stage_) : code{std::move(words)} {
        stage = stage_;
        is_ampere = true;
    }

    u64 ReadInstruction(u32 address) override {
        return code.at(address / 8);
    }
    u32 ReadCbufValue(u32, u32) override {
        return 0;
    }
    TextureType ReadTextureType(u32) override {
        return TextureType::Color2D;
    }
    TexturePixelFormat ReadTexturePixelFormat(u32) override {
        return {};
    }
    bool IsTexturePixelFormatInteger(u32) override {
        return false;
    }
    u32 ReadViewportTransformState() override {
        return 0;
    }
    u32 TextureBoundBuffer() const override {
        return 0;
    }
    u32 LocalMemorySize() const override {
        return 0;
    }
    u32 SharedMemorySize() const override {
        return 0;
    }
    std::array<u32, 3> WorkgroupSize() const override {
        return {1, 1, 1};
    }
    bool HasHLEMacroState() const override {
        return false;
    }
    std::optional<ReplaceConstant> GetReplaceConstBuffer(u32, u32) override {
        return std::nullopt;
    }
    void Dump(u64, u64) override {}

private:
    std::vector<u64> code;
};

struct Translation {
    ObjectPool<IR::Inst> inst_pool;
    ObjectPool<IR::Block> block_pool;
    ObjectPool<Maxwell::Flow::Block> flow_pool;
    IR::Program program;
};

void Translate(Translation& t, TestEnvironment& env) {
    Maxwell::Flow::CFG cfg{env, t.flow_pool, u32{0}};
    HostTranslateInfo host_info{};
    host_info.support_float64 = true;
    host_info.support_float16 = true;
    host_info.support_int64 = true;
    host_info.support_conditional_barrier = true;
    t.program = Maxwell::TranslateProgram(t.inst_pool, t.block_pool, env, cfg, host_info);
}

void Translate(Translation& t, const std::vector<Encoding>& code) {
    TestEnvironment env{code};
    Translate(t, env);
}

// Opcodes present anywhere in a program
std::vector<IR::Opcode> Opcodes(const IR::Program& program) {
    std::vector<IR::Opcode> opcodes;
    for (const IR::Block* const block : program.blocks) {
        for (const IR::Inst& inst : block->Instructions()) {
            opcodes.push_back(inst.GetOpcode());
        }
    }
    return opcodes;
}

bool Contains(const std::vector<IR::Opcode>& opcodes, IR::Opcode opcode) {
    return std::ranges::find(opcodes, opcode) != opcodes.end();
}

IR::FpRounding Rounding(const IR::Inst& inst) {
    return inst.Flags<IR::FpControl>().rounding;
}

// Rounds an exactly representable double to float in the given direction
f32 RoundToF32(f64 value, IR::FpRounding rounding) {
    const f32 nearest{static_cast<f32>(value)};
    const f64 rounded{static_cast<f64>(nearest)};
    switch (rounding) {
    case IR::FpRounding::RM:
        return rounded > value ? std::nextafter(nearest, -INFINITY) : nearest;
    case IR::FpRounding::RP:
        return rounded < value ? std::nextafter(nearest, INFINITY) : nearest;
    case IR::FpRounding::RZ:
        return std::fabs(rounded) > std::fabs(value) ? std::nextafter(nearest, 0.0f) : nearest;
    default:
        return nearest;
    }
}

// Rounds an exact value to half precision in the given direction
u64 F64ToF16(f64 value, IR::FpRounding rounding) {
    const u32 sign{std::signbit(value) ? 0x8000u : 0u};
    if (std::isnan(value)) {
        return sign | 0x7e00;
    }
    if (std::isinf(value)) {
        return sign | 0x7c00;
    }
    const f64 magnitude{std::fabs(value)};
    // Scale to units of the smallest half denormal, then round to an integer of 2^-24
    const auto round_units{[&](f64 units) {
        switch (rounding) {
        case IR::FpRounding::RZ:
            return std::trunc(units);
        case IR::FpRounding::RM:
            return sign != 0 ? std::ceil(units) : std::floor(units);
        case IR::FpRounding::RP:
            return sign != 0 ? std::floor(units) : std::ceil(units);
        default:
            return std::nearbyint(units);
        }
    }};
    if (magnitude < std::ldexp(1.0, -14)) {
        return sign | static_cast<u32>(round_units(magnitude * std::ldexp(1.0, 24)));
    }
    const int exponent{std::ilogb(magnitude)};
    const f64 mantissa{round_units(std::ldexp(magnitude, 10 - exponent))};
    const u32 result{static_cast<u32>(((exponent + 15) << 10) + (static_cast<u32>(mantissa) - 1024))};
    if (exponent > 15 || result >= 0x7c00) {
        const bool to_infinity{rounding == IR::FpRounding::RN ||
                               rounding == IR::FpRounding::DontCare ||
                               (rounding == IR::FpRounding::RP && sign == 0) ||
                               (rounding == IR::FpRounding::RM && sign != 0)};
        return sign | (to_infinity ? 0x7c00 : 0x7bff);
    }
    return sign | result;
}

u64 F32ToF16(f32 value, IR::FpRounding rounding) {
    if (std::isnan(value)) {
        const u32 bits{std::bit_cast<u32>(value)};
        return ((bits >> 16) & 0x8000) | 0x7e00 | ((bits >> 13) & 0x3ff);
    }
    return F64ToF16(static_cast<f64>(value), rounding);
}

f32 F16ToF32(u16 half) {
    const u32 sign{(half & 0x8000u) << 16};
    const u32 exponent{(half >> 10) & 0x1fu};
    const u32 mantissa{half & 0x3ffu};
    if (exponent == 0x1f) {
        return std::bit_cast<f32>(sign | 0x7f800000 | (mantissa << 13));
    }
    const f32 magnitude{exponent == 0 ? std::ldexp(static_cast<f32>(mantissa), -24)
                                      : std::ldexp(static_cast<f32>(mantissa | 0x400),
                                                   static_cast<int>(exponent) - 25)};
    return std::bit_cast<f32>(sign | std::bit_cast<u32>(magnitude));
}

// Evaluates the straight-line computation feeding a value. Every value is kept as raw bits:
// 32-bit types in the low word, pairs as lo | hi << 32 and booleans as 0 or 1.
u64 Eval(const IR::Value& value) {
    if (value.IsImmediate()) {
        switch (value.Type()) {
        case IR::Type::U1:
            return value.U1() ? 1 : 0;
        case IR::Type::U32:
            return value.U32();
        case IR::Type::F32:
            return std::bit_cast<u32>(value.F32());
        case IR::Type::U64:
            return value.U64();
        default:
            FAIL("Unexpected immediate type");
            return 0;
        }
    }
    const IR::Inst& inst{*value.InstRecursive()};
    const auto arg{[&](size_t index) { return Eval(inst.Arg(index)); }};
    const auto u32_arg{[&](size_t index) { return static_cast<u32>(arg(index)); }};
    const auto s32_arg{[&](size_t index) { return static_cast<s32>(arg(index)); }};
    const auto f32_arg{[&](size_t index) { return std::bit_cast<f32>(u32_arg(index)); }};
    const auto f32_bits{[](f32 result) { return u64{std::bit_cast<u32>(result)}; }};
    const auto u32_bits{[](auto result) { return u64{static_cast<u32>(result)}; }};
    const auto f16_arg{[&](size_t index) {
        return static_cast<f64>(F16ToF32(static_cast<u16>(arg(index))));
    }};
    const auto f16_bits{[](f64 result) { return F64ToF16(result, IR::FpRounding::RN); }};
    // Comparisons: ordered ones are false and unordered ones true when an operand is NaN
    const auto compare{[&](bool is_16, auto relation, bool ordered) -> u64 {
        const f64 a{is_16 ? f16_arg(0) : static_cast<f64>(f32_arg(0))};
        const f64 b{is_16 ? f16_arg(1) : static_cast<f64>(f32_arg(1))};
        if (std::isnan(a) || std::isnan(b)) {
            return ordered ? 0 : 1;
        }
        return relation(a, b) ? 1 : 0;
    }};
    // Saturation is max(x, +0) then min(x, 1): NaN and -0 become +0
    const auto saturate{[](f64 x) { return std::isnan(x) || x <= 0.0 ? 0.0 : std::min(x, 1.0); }};
    switch (inst.GetOpcode()) {
#define COMPARE(name, rel)                                                                             case IR::Opcode::FPOrd##name##16:                                                                      return compare(true, rel, true);                                                               case IR::Opcode::FPUnord##name##16:                                                                    return compare(true, rel, false);                                                              case IR::Opcode::FPOrd##name##32:                                                                      return compare(false, rel, true);                                                              case IR::Opcode::FPUnord##name##32:                                                                    return compare(false, rel, false);
        COMPARE(Equal, std::equal_to<>{})
        COMPARE(NotEqual, std::not_equal_to<>{})
        COMPARE(LessThan, std::less<>{})
        COMPARE(LessThanEqual, std::less_equal<>{})
        COMPARE(GreaterThan, std::greater<>{})
        COMPARE(GreaterThanEqual, std::greater_equal<>{})
#undef COMPARE
    case IR::Opcode::FPAdd16:
        return f16_bits(f16_arg(0) + f16_arg(1));
    case IR::Opcode::FPMul16:
        return f16_bits(f16_arg(0) * f16_arg(1));
    case IR::Opcode::FPFma16:
        return f16_bits(std::fma(f16_arg(0), f16_arg(1), f16_arg(2)));
    case IR::Opcode::FPAbs16:
        return arg(0) & 0x7fff;
    case IR::Opcode::FPNeg16:
        return arg(0) ^ 0x8000;
    case IR::Opcode::FPSaturate16:
        return f16_bits(saturate(f16_arg(0)));
    case IR::Opcode::FPSaturate32:
        return f32_bits(static_cast<f32>(saturate(f32_arg(0))));
    case IR::Opcode::FPMin32:
        return f32_bits(std::fmin(f32_arg(0), f32_arg(1)));
    case IR::Opcode::FPMax32:
        return f32_bits(std::fmax(f32_arg(0), f32_arg(1)));
    case IR::Opcode::Identity:
    case IR::Opcode::BitCastU32F32:
    case IR::Opcode::BitCastF32U32:
    case IR::Opcode::PackUint2x32:
    case IR::Opcode::UnpackUint2x32:
        return arg(0);
    case IR::Opcode::CompositeConstructU32x2:
        return arg(0) | (arg(1) << 32);
    case IR::Opcode::CompositeExtractU32x2:
        return (arg(0) >> (32 * inst.Arg(1).U32())) & 0xffffffff;
    case IR::Opcode::IAdd32:
        return u32_bits(u32_arg(0) + u32_arg(1));
    case IR::Opcode::ISub32:
        return u32_bits(u32_arg(0) - u32_arg(1));
    case IR::Opcode::IMul32:
        return u32_bits(u32_arg(0) * u32_arg(1));
    case IR::Opcode::INeg32:
        return u32_bits(0u - u32_arg(0));
    case IR::Opcode::ShiftLeftLogical32:
        return u32_bits(u32_arg(0) << u32_arg(1));
    case IR::Opcode::ShiftRightLogical32:
        return u32_bits(u32_arg(0) >> u32_arg(1));
    case IR::Opcode::ShiftRightArithmetic32:
        return u32_bits(s32_arg(0) >> u32_arg(1));
    case IR::Opcode::ShiftLeftLogical64:
        return arg(0) << u32_arg(1);
    case IR::Opcode::ShiftRightLogical64:
        return arg(0) >> u32_arg(1);
    case IR::Opcode::ShiftRightArithmetic64:
        return static_cast<u64>(static_cast<s64>(arg(0)) >> u32_arg(1));
    case IR::Opcode::BitwiseAnd32:
        return arg(0) & arg(1);
    case IR::Opcode::BitwiseOr32:
        return arg(0) | arg(1);
    case IR::Opcode::BitwiseXor32:
        return arg(0) ^ arg(1);
    case IR::Opcode::BitwiseNot32:
        return u32_bits(~u32_arg(0));
    case IR::Opcode::BitCount32:
        return static_cast<u64>(std::popcount(u32_arg(0)));
    case IR::Opcode::FindUMsb32:
        return u32_arg(0) == 0 ? 0xffffffff : 31 - std::countl_zero(u32_arg(0));
    case IR::Opcode::UMin32:
        return std::min(u32_arg(0), u32_arg(1));
    case IR::Opcode::UMax32:
        return std::max(u32_arg(0), u32_arg(1));
    case IR::Opcode::SMin32:
        return u32_bits(std::min(s32_arg(0), s32_arg(1)));
    case IR::Opcode::SMax32:
        return u32_bits(std::max(s32_arg(0), s32_arg(1)));
    case IR::Opcode::ULessThan:
        return u32_arg(0) < u32_arg(1) ? 1 : 0;
    case IR::Opcode::SLessThan:
        return s32_arg(0) < s32_arg(1) ? 1 : 0;
    case IR::Opcode::IEqual:
        return arg(0) == arg(1) ? 1 : 0;
    case IR::Opcode::INotEqual:
        return arg(0) != arg(1) ? 1 : 0;
    case IR::Opcode::SelectU1:
    case IR::Opcode::SelectU32:
    case IR::Opcode::SelectF32:
        return arg(0) != 0 ? arg(1) : arg(2);
    case IR::Opcode::LogicalAnd:
        return arg(0) & arg(1);
    case IR::Opcode::LogicalOr:
        return arg(0) | arg(1);
    case IR::Opcode::LogicalXor:
        return arg(0) ^ arg(1);
    case IR::Opcode::LogicalNot:
        return arg(0) ^ 1;
    case IR::Opcode::FPAdd32:
        return f32_bits(f32_arg(0) + f32_arg(1));
    case IR::Opcode::FPMul32:
        return f32_bits(f32_arg(0) * f32_arg(1));
    case IR::Opcode::FPFma32:
        return f32_bits(std::fma(f32_arg(0), f32_arg(1), f32_arg(2)));
    case IR::Opcode::FPNeg32:
        return f32_bits(-f32_arg(0));
    case IR::Opcode::FPAbs32:
        return f32_bits(std::fabs(f32_arg(0)));
    case IR::Opcode::FPRecip32:
        return f32_bits(1.0f / f32_arg(0));
    case IR::Opcode::FPRoundEven32:
        return f32_bits(std::nearbyint(f32_arg(0)));
    case IR::Opcode::FPFloor32:
        return f32_bits(std::floor(f32_arg(0)));
    case IR::Opcode::FPCeil32:
        return f32_bits(std::ceil(f32_arg(0)));
    case IR::Opcode::FPTrunc32:
        return f32_bits(std::trunc(f32_arg(0)));
    case IR::Opcode::FPClamp32:
        return f32_bits(std::fmin(std::fmax(f32_arg(0), f32_arg(1)), f32_arg(2)));
    case IR::Opcode::FPIsNan32:
        return std::isnan(f32_arg(0)) ? 1 : 0;
    case IR::Opcode::ConvertS32F32:
        return u32_bits(static_cast<s32>(f32_arg(0)));
    case IR::Opcode::ConvertU32F32:
        return u32_bits(static_cast<u32>(f32_arg(0)));
    case IR::Opcode::ConvertF32S32:
        return f32_bits(RoundToF32(static_cast<f64>(s32_arg(0)), Rounding(inst)));
    case IR::Opcode::ConvertF32U32:
        return f32_bits(RoundToF32(static_cast<f64>(u32_arg(0)), Rounding(inst)));
    case IR::Opcode::ConvertF16F32:
        return F32ToF16(f32_arg(0), Rounding(inst));
    case IR::Opcode::ConvertF32F16:
        return f32_bits(F16ToF32(static_cast<u16>(arg(0))));
    case IR::Opcode::CompositeConstructF16x2:
        return arg(0) | (arg(1) << 16);
    case IR::Opcode::CompositeExtractF16x2:
        return (arg(0) >> (16 * inst.Arg(1).U32())) & 0xffff;
    case IR::Opcode::PackFloat2x16:
    case IR::Opcode::UnpackFloat2x16:
        return arg(0);
    case IR::Opcode::BitFieldUExtract: {
        const u32 count{u32_arg(2)};
        const u32 mask{count >= 32 ? 0xffffffffu : (1u << count) - 1};
        return (u32_arg(0) >> u32_arg(1)) & mask;
    }
    case IR::Opcode::BitFieldSExtract: {
        const u32 count{u32_arg(2)};
        const u32 field{static_cast<u32>(u32_arg(0) << (32 - u32_arg(1) - count))};
        return u32_bits(static_cast<s32>(field) >> (32 - count));
    }
    case IR::Opcode::BitFieldInsert: {
        const u32 offset{u32_arg(2)};
        const u32 count{u32_arg(3)};
        const u32 mask{(count >= 32 ? 0xffffffffu : (1u << count) - 1) << offset};
        return (u32_arg(0) & ~mask) | ((u32_arg(1) << offset) & mask);
    }
    default:
        FAIL(fmt::format("Evaluator does not support {}", IR::NameOf(inst.GetOpcode())));
        return 0;
    }
}

// Raw bits written to a component of render target 0
std::optional<u32> FragColor(const IR::Program& program, u32 component) {
    for (const IR::Block* const block : program.blocks) {
        for (const IR::Inst& inst : block->Instructions()) {
            if (inst.GetOpcode() == IR::Opcode::SetFragColor && inst.Arg(0).U32() == 0 &&
                inst.Arg(1).U32() == component) {
                return static_cast<u32>(Eval(inst.Arg(2)));
            }
        }
    }
    return std::nullopt;
}

std::array<std::optional<u32>, 4> FragColors(const IR::Program& program) {
    return {FragColor(program, 0), FragColor(program, 1), FragColor(program, 2),
            FragColor(program, 3)};
}

} // Anonymous namespace

TEST_CASE("Ampere translate: integer arithmetic", "[shader][ampere]") {
    Translation t;
    Translate(t, {
        MovImm(4, 0xffffffff),
        MovImm(5, 1),
        MovImm(6, 0x80000001),
        MovImm(7, 1),
        // IADD3 R0, P0, R4, R5, RZ
        Encoding{0x210}.Set(16, 8, 0).Set(24, 8, 4).Set(32, 8, 5).Set(64, 8, RZ)
            .Set(81, 3, 0).Set(84, 3, PT),
        // IADD3.X R1, RZ, RZ, RZ, P0, !PT
        Encoding{0x210}.Set(16, 8, 1).Set(24, 8, RZ).Set(32, 8, RZ).Set(64, 8, RZ).Set(74, 1, 1)
            .Set(81, 3, PT).Set(84, 3, PT).Set(87, 3, 0).Set(77, 3, PT).Set(80, 1, 1),
        // SHF.L.U64.HI R2, R6, 0x4, R7
        Encoding{0x819}.Set(16, 8, 2).Set(24, 8, 6).Set(32, 32, 4).Set(64, 8, 7).Set(73, 2, 1)
            .Set(80, 1, 1),
        // LOP3.LUT R3, R4, R5, R6, 0x96, !PT
        Encoding{0x212}.Set(16, 8, 3).Set(24, 8, 4).Set(32, 8, 5).Set(64, 8, 6).Set(72, 8, 0x96)
            .Set(81, 3, PT).Set(87, 3, PT).Set(90, 1, 1),
        Exit(),
    });
    const auto colors{FragColors(t.program)};
    REQUIRE(colors[0] == 0u);
    REQUIRE(colors[1] == 1u);
    REQUIRE(colors[2] == 0x18u);
    REQUIRE(colors[3] == 0x7fffffffu);
}

TEST_CASE("Ampere translate: wide multiply and compare", "[shader][ampere]") {
    Translation t;
    Translate(t, {
        MovImm(4, 0xfffffffe),
        MovImm(5, 3),
        // IMAD.WIDE R0, R4, R5, RZ
        Encoding{0x225}.Set(16, 8, 0).Set(24, 8, 4).Set(32, 8, 5).Set(64, 8, RZ).Set(73, 1, 1)
            .Set(81, 3, PT).Set(84, 3, PT),
        // ISETP.LT.AND P1, PT, R4, R5, PT
        Encoding{0x20c}.Set(24, 8, 4).Set(32, 8, 5).Set(73, 1, 1).Set(76, 3, 1).Set(81, 3, 1)
            .Set(84, 3, PT).Set(87, 3, PT),
        // SEL R2, R4, R5, P1
        Encoding{0x207}.Set(16, 8, 2).Set(24, 8, 4).Set(32, 8, 5).Set(87, 3, 1),
        // IMNMX R3, R4, R5, !PT
        Encoding{0x217}.Set(16, 8, 3).Set(24, 8, 4).Set(32, 8, 5).Set(73, 1, 1).Set(87, 3, PT)
            .Set(90, 1, 1),
        Exit(),
    });
    const auto colors{FragColors(t.program)};
    REQUIRE(colors[0] == 0xfffffffau);
    REQUIRE(colors[1] == 0xffffffffu);
    REQUIRE(colors[2] == 0xfffffffeu);
    REQUIRE(colors[3] == 3u);
}

TEST_CASE("Ampere translate: floating point", "[shader][ampere]") {
    Translation t;
    Translate(t, {
        MovImm(4, std::bit_cast<u32>(2.0f)),
        MovImm(5, std::bit_cast<u32>(3.0f)),
        // FFMA R0, R4, R5, R5
        Encoding{0x223}.Set(16, 8, 0).Set(24, 8, 4).Set(32, 8, 5).Set(64, 8, 5),
        // FADD R1, -R4, 1
        Encoding{0x421}.Set(16, 8, 1).Set(24, 8, 4).Set(72, 1, 1)
            .Set(32, 32, std::bit_cast<u32>(1.0f)).Set(64, 8, RZ),
        // FMUL R2, R4, R5
        Encoding{0x220}.Set(16, 8, 2).Set(24, 8, 4).Set(32, 8, 5).Set(84, 3, 4),
        // MUFU.RCP R3, R4
        Encoding{0x308}.Set(16, 8, 3).Set(32, 8, 4).Set(74, 6, 4),
        Exit(),
    });
    const auto colors{FragColors(t.program)};
    REQUIRE(colors[0] == std::bit_cast<u32>(9.0f));
    REQUIRE(colors[1] == std::bit_cast<u32>(-1.0f));
    REQUIRE(colors[2] == std::bit_cast<u32>(6.0f));
    REQUIRE(colors[3] == std::bit_cast<u32>(0.5f));
}

TEST_CASE("Ampere translate: loop", "[shader][ampere]") {
    // Branch targets are relative to the next instruction, in 32-bit words
    const auto bra_back{[](u64 pred, s64 instructions) {
        return Encoding{0x947, pred}.Set(34, 48, static_cast<u64>(instructions * 4))
            .Set(87, 3, PT);
    }};
    Translation t;
    Translate(t, {
        MovImm(0, 0),
        MovImm(1, 5),
        // IADD3 R0, R0, R1, RZ
        Encoding{0x210}.Set(16, 8, 0).Set(24, 8, 0).Set(32, 8, 1).Set(64, 8, RZ)
            .Set(81, 3, PT).Set(84, 3, PT),
        // IADD3 R1, R1, -0x1, RZ
        Encoding{0x810}.Set(16, 8, 1).Set(24, 8, 1).Set(32, 32, 0xffffffff).Set(64, 8, RZ)
            .Set(81, 3, PT).Set(84, 3, PT),
        // ISETP.NE.AND P0, PT, R1, RZ, PT
        Encoding{0x20c}.Set(24, 8, 1).Set(32, 8, RZ).Set(76, 3, 5).Set(81, 3, 0).Set(84, 3, PT)
            .Set(87, 3, PT),
        // @P0 BRA back to the first IADD3
        bra_back(0, -4),
        MovImm(2, 7),
        MovImm(3, 9),
        Exit(),
    });
    REQUIRE(t.program.blocks.size() > 2);
    REQUIRE(FragColor(t.program, 2) == 7u);
    REQUIRE(FragColor(t.program, 3) == 9u);
}

namespace {

struct HardwareVector {
    const char* name;
    u64 lo;
    u64 hi;
    std::vector<std::pair<u64, u32>> inputs;
    u32 expected;
};

// Results of the exact encodings executed on a GA102 (SM 8.6), written to R7
const std::vector<HardwareVector> HARDWARE_VECTORS{
#include "ampere_hardware_vectors.inc"
};

bool IsF32NaN(u32 bits) {
    return (bits & 0x7f800000) == 0x7f800000 && (bits & 0x7fffff) != 0;
}

bool IsF16NaN(u32 bits) {
    return (bits & 0x7c00) == 0x7c00 && (bits & 0x3ff) != 0;
}

// NaN payloads depend on the host, only their presence is compared
bool SameResult(std::string_view name, u32 result, u32 expected) {
    if (result == expected) {
        return true;
    }
    const bool packed_halves{name.starts_with("F2F.F16") || name.starts_with("H") ||
                             name.starts_with("F2FP")};
    if (!packed_halves) {
        return (name.starts_with("F2F.F32") || name.starts_with("I2F")) && IsF32NaN(result) &&
               IsF32NaN(expected);
    }
    {
        // Packed halves, compared lane by lane
        for (const u32 shift : {0u, 16u}) {
            const u32 a{(result >> shift) & 0xffff};
            const u32 b{(expected >> shift) & 0xffff};
            if (a != b && !(IsF16NaN(a) && IsF16NaN(b))) {
                return false;
            }
        }
        return true;
    }
    return false;
}

} // Anonymous namespace

TEST_CASE("Ampere translate: hardware vectors", "[shader][ampere]") {
    std::string failures;
    for (const HardwareVector& vector : HARDWARE_VECTORS) {
        std::vector<Encoding> code;
        for (const auto& [reg, value] : vector.inputs) {
            code.push_back(MovImm(reg, value));
        }
        Encoding insn{0};
        insn.lo = vector.lo;
        insn.hi = vector.hi;
        code.push_back(insn);
        // MOV R0, R7
        code.push_back(Encoding{0x202}.Set(16, 8, 0).Set(32, 8, 7).Set(72, 4, 0xf));
        code.push_back(Exit());
        Translation t;
        Translate(t, code);
        const std::optional<u32> result{FragColor(t.program, 0)};
        REQUIRE(result.has_value());
        if (!SameResult(vector.name, *result, vector.expected)) {
            failures += fmt::format("{} inputs[0]={:08x}: got {:08x}, hardware {:08x}\n",
                                    vector.name, vector.inputs.front().second, *result,
                                    vector.expected);
        }
    }
    INFO(failures);
    REQUIRE(failures.empty());
}

TEST_CASE("Ampere translate: storage buffer through a uniform register", "[shader][ampere]") {
    Translation t;
    Translate(t, {
        // ULDC.64 UR4, c[0x0][0x110]
        Encoding{0xab9}.Set(16, 6, 4).Set(38, 16, 0x110).Set(54, 5, 0).Set(73, 3, 5),
        // LDG.E R0, [UR4.64+0x10]
        Encoding{0x981}.Set(16, 8, 0).Set(24, 8, RZ).Set(32, 6, 4).Set(40, 24, 0x10)
            .Set(72, 1, 1).Set(73, 3, 4).Set(81, 3, PT).Set(84, 3, 1).Set(91, 1, 1),
        Exit(),
    });
    const auto& descriptors{t.program.info.storage_buffers_descriptors};
    REQUIRE(descriptors.size() == 1);
    REQUIRE(descriptors[0].cbuf_index == 0);
    REQUIRE(descriptors[0].cbuf_offset == 0x110);
    REQUIRE(!descriptors[0].is_written);
}

namespace {

// Kernels from tools/ampere-sm86/templates/corpus.cu compiled with nvcc -arch=sm_86 -O3
const std::vector<u64> TEXTURE_KERNEL{
#include "ampere_kernel_texk.inc"
};

const std::vector<u64> WARP_KERNEL{
#include "ampere_kernel_warp.inc"
};

} // Anonymous namespace

TEST_CASE("Ampere translate: nvcc texture kernel", "[shader][ampere]") {
    // CUDA passes texture objects as kernel parameters at c[0x0][0x160] and c[0x0][0x168]
    Translation t;
    TestEnvironment env{TEXTURE_KERNEL, Stage::Compute};
    Translate(t, env);
    std::vector<std::pair<u32, TextureType>> textures;
    for (const TextureDescriptor& desc : t.program.info.texture_descriptors) {
        REQUIRE(desc.cbuf_index == 0);
        textures.emplace_back(desc.cbuf_offset, desc.type);
    }
    const auto has{[&](u32 offset, TextureType type) {
        return std::ranges::find(textures, std::pair{offset, type}) != textures.end();
    }};
    REQUIRE(has(0x160, TextureType::Color2D));
    REQUIRE(has(0x160, TextureType::ColorArray2D));
    REQUIRE(has(0x160, TextureType::Color1D));
    REQUIRE(has(0x168, TextureType::Color3D));
    REQUIRE(has(0x168, TextureType::ColorCube));

    // The surface read and written by SULD/SUST is the third parameter, which keeps the
    // texture results above alive
    const auto& images{t.program.info.image_descriptors};
    REQUIRE(std::ranges::any_of(images, [](const ImageDescriptor& desc) {
        return desc.cbuf_index == 0 && desc.cbuf_offset == 0x170 && desc.is_written;
    }));
    const auto opcodes{Opcodes(t.program)};
    REQUIRE(Contains(opcodes, IR::Opcode::ImageGradient));
    REQUIRE(Contains(opcodes, IR::Opcode::ImageGather));
    REQUIRE(Contains(opcodes, IR::Opcode::ImageFetch));
    REQUIRE(Contains(opcodes, IR::Opcode::ImageRead));
    REQUIRE(Contains(opcodes, IR::Opcode::ImageWrite));

    // The result is stored through a pointer read from c[0x0][0x178] by IMAD.WIDE
    const auto& storage{t.program.info.storage_buffers_descriptors};
    REQUIRE(std::ranges::any_of(storage, [](const StorageBufferDescriptor& desc) {
        return desc.cbuf_index == 0 && desc.cbuf_offset == 0x178 && desc.is_written;
    }));
}

TEST_CASE("Ampere translate: nvcc warp kernel", "[shader][ampere]") {
    Translation t;
    TestEnvironment env{WARP_KERNEL, Stage::Compute};
    Translate(t, env);
    const auto opcodes{Opcodes(t.program)};
    REQUIRE(Contains(opcodes, IR::Opcode::ShuffleIndex));
    REQUIRE(Contains(opcodes, IR::Opcode::ShuffleButterfly));
    REQUIRE(Contains(opcodes, IR::Opcode::ShuffleDown));
    REQUIRE(Contains(opcodes, IR::Opcode::VoteAny));
    REQUIRE(Contains(opcodes, IR::Opcode::SubgroupBallot));
    REQUIRE(Contains(opcodes, IR::Opcode::Barrier));
    // atomicMax on shared memory, atomicAdd and atomicExch on global memory
    REQUIRE(Contains(opcodes, IR::Opcode::SharedAtomicSMax32));
    REQUIRE((Contains(opcodes, IR::Opcode::StorageAtomicIAdd32) ||
             Contains(opcodes, IR::Opcode::GlobalAtomicIAdd32)));
    REQUIRE((Contains(opcodes, IR::Opcode::StorageAtomicExchange64) ||
             Contains(opcodes, IR::Opcode::StorageAtomicExchange32x2) ||
             Contains(opcodes, IR::Opcode::GlobalAtomicExchange64)));
}

TEST_CASE("Ampere translate: interpolation", "[shader][ampere]") {
    Translation t;
    Translate(t, {
        // IPA.PASS R0, a[0x80]
        Encoding{0x326}.Set(16, 8, 0).Set(24, 8, RZ).Set(32, 8, RZ).Set(64, 8, 0x20)
            .Set(81, 3, PT),
        // IPA.CONSTANT R1, a[0x84]
        Encoding{0x326}.Set(16, 8, 1).Set(24, 8, RZ).Set(32, 8, RZ).Set(64, 8, 0x21)
            .Set(78, 2, 1).Set(81, 3, PT),
        Exit(),
    });
    const auto color{[&](u32 component) -> const IR::Inst* {
        for (const IR::Block* const block : t.program.blocks) {
            for (const IR::Inst& inst : block->Instructions()) {
                if (inst.GetOpcode() == IR::Opcode::SetFragColor &&
                    inst.Arg(1).U32() == component) {
                    return inst.Arg(2).InstRecursive();
                }
            }
        }
        return nullptr;
    }};
    // Perspective inputs come back multiplied by W, flat inputs as they are
    const IR::Inst* const perspective{color(0)};
    REQUIRE(perspective != nullptr);
    REQUIRE(perspective->GetOpcode() == IR::Opcode::FPMul32);
    REQUIRE(perspective->Arg(0).InstRecursive()->Arg(0).Attribute() == IR::Attribute::Generic0X);
    REQUIRE(perspective->Arg(1).InstRecursive()->Arg(0).Attribute() == IR::Attribute::PositionW);
    const IR::Inst* const flat{color(1)};
    REQUIRE(flat != nullptr);
    REQUIRE(flat->GetOpcode() == IR::Opcode::GetAttribute);
    REQUIRE(flat->Arg(0).Attribute() == IR::Attribute::Generic0Y);
}

TEST_CASE("Ampere translate: attribute store", "[shader][ampere]") {
    // AST.128 a[0x80], R4 in a vertex shader
    std::vector<u64> words;
    for (const Encoding& insn :
         {MovImm(4, 1), MovImm(5, 2), MovImm(6, 3), MovImm(7, 4),
          Encoding{0x322}.Set(24, 8, RZ).Set(32, 8, 4).Set(64, 8, RZ).Set(40, 10, 0x80)
              .Set(74, 2, 3),
          Exit()}) {
        words.push_back(insn.lo);
        words.push_back(insn.hi);
    }
    Translation t;
    TestEnvironment env{words, Stage::VertexB};
    Translate(t, env);
    std::vector<IR::Attribute> written;
    for (const IR::Block* const block : t.program.blocks) {
        for (const IR::Inst& inst : block->Instructions()) {
            if (inst.GetOpcode() == IR::Opcode::SetAttribute) {
                written.push_back(inst.Arg(0).Attribute());
            }
        }
    }
    REQUIRE(written == std::vector{IR::Attribute::Generic0X, IR::Attribute::Generic0Y,
                                   IR::Attribute::Generic0Z, IR::Attribute::Generic0W});
}

TEST_CASE("Ampere translate: warp match", "[shader][ampere]") {
    // MATCH.ANY R0, R4; MATCH.ALL.U64 P0, R1, R2; AST.64 a[0x80], R0
    std::vector<u64> words;
    for (const Encoding& insn :
         {Encoding{0x3a1}.Set(16, 8, 0).Set(24, 8, 4).Set(79, 1, 1),
          Encoding{0x3a1}.Set(16, 8, 1).Set(24, 8, 2).Set(73, 1, 1).Set(81, 3, 0),
          Encoding{0x322}.Set(24, 8, RZ).Set(32, 8, 0).Set(64, 8, RZ).Set(40, 10, 0x80)
              .Set(74, 2, 1),
          Exit()}) {
        words.push_back(insn.lo);
        words.push_back(insn.hi);
    }
    Translation t;
    TestEnvironment env{words, Stage::VertexB};
    Translate(t, env);
    const auto opcodes{Opcodes(t.program)};
    REQUIRE(Contains(opcodes, IR::Opcode::ShuffleIndex));
    REQUIRE(Contains(opcodes, IR::Opcode::SubgroupBallot));
    REQUIRE(Contains(opcodes, IR::Opcode::VoteAll));
}

namespace {
// tools/ampere-sm86/templates/converge.cu compiled with nvcc -arch=sm_86 -O3
const std::vector<u64> CONVERGE_KERNEL{
#include "ampere_kernel_converge.inc"
};
} // Anonymous namespace

TEST_CASE("Ampere translate: nvcc divergent loops", "[shader][ampere]") {
    // "@!P0 BREAK B1" is followed by an explicit "@!P0 BRA"; "BRA.CONV ~URZ" selects the native
    // SHFL/REDUX over a CALL into a software fallback, which must not be reached
    Translation t;
    TestEnvironment env{CONVERGE_KERNEL, Stage::Compute};
    Translate(t, env);
    REQUIRE(t.program.blocks.size() > 4);
    const auto opcodes{Opcodes(t.program)};
    REQUIRE(Contains(opcodes, IR::Opcode::ShuffleButterfly));
    REQUIRE(Contains(opcodes, IR::Opcode::ShuffleIndex));
}
