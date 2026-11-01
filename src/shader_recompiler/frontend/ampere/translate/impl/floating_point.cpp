// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cmath>
#include <numbers>

#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ampere/translate/impl/impl.h"
#include "shader_recompiler/frontend/maxwell/translate/impl/common_funcs.h"

namespace Shader::Ampere {
namespace {

// Rounding at [78..80), .FMZ at 76 and .FTZ at 80
IR::FpControl Control(Instruction insn, bool has_fmz) {
    constexpr std::array rounding{IR::FpRounding::RN, IR::FpRounding::RM, IR::FpRounding::RP,
                                  IR::FpRounding::RZ};
    IR::FmzMode fmz{IR::FmzMode::None};
    if (has_fmz && insn.Bit(76)) {
        fmz = IR::FmzMode::FMZ;
    } else if (insn.Bit(80)) {
        fmz = IR::FmzMode::FTZ;
    }
    return {
        .no_contraction = true,
        .rounding = rounding[insn.Bits(78, 2)],
        .fmz_mode = fmz,
    };
}

IR::F32 Saturate(IR::IREmitter& ir, Instruction insn, const IR::F32& value) {
    return insn.Bit(77) ? IR::F32{ir.FPSaturate(value)} : value;
}

} // Anonymous namespace

void TranslatorVisitor::FADD(Instruction insn) {
    // The second operand is always on [32..64), the [64..72) field holds RZ
    const IR::F32 result{ir.FPAdd(FloatSrcA(insn), FloatWide(insn), Control(insn, false))};
    F(Reg(insn.Dest()), Saturate(ir, insn, result));
}

// FSWZADD Rd, Ra, Rb, swizzle: each lane of a quad adds Ra and Rb with the signs that its two
// bits of the swizzle byte at [32..40) select. The byte means the same as on Maxwell (nvdisasm
// prints lane 0 first there and lane 3 first here); Rb is at [64..72).
void TranslatorVisitor::FSWZADD(Instruction insn) {
    const IR::F32 src_b{F(Reg(insn.Bits(64, 8)))};
    const IR::U32 swizzle{ir.Imm32(static_cast<u32>(insn.Bits(32, 8)))};
    F(Reg(insn.Dest()), ir.FSwizzleAdd(F(Reg(insn.SrcA())), src_b, swizzle, Control(insn, false)));
}

void TranslatorVisitor::FMUL(Instruction insn) {
    const u64 scale{insn.Bits(84, 3)};
    if (scale == 0) {
        throw NotImplementedException("FMUL scale 0");
    }
    const IR::FpControl control{Control(insn, true)};
    IR::F32 lhs{FloatSrcA(insn)};
    if (scale != 4) {
        lhs = ir.FPMul(lhs, ir.Imm32(std::ldexp(1.0f, static_cast<int>(scale) - 4)), control);
    }
    const IR::F32 result{ir.FPMul(lhs, FloatSrcB(insn), control)};
    F(Reg(insn.Dest()), Saturate(ir, insn, result));
}

void TranslatorVisitor::FFMA(Instruction insn) {
    const IR::F32 result{
        ir.FPFma(FloatSrcA(insn), FloatSrcB(insn), FloatSrcC(insn), Control(insn, true))};
    F(Reg(insn.Dest()), Saturate(ir, insn, result));
}

// FMNMX.NAN returns the canonical NaN 0x7fffffff when either source is NaN. FMNMX.XORSIGN
// (operands printed as |Ra|, |Rb|) picks between the absolute values and gives the result the
// sign of Ra xor Rb, taken from the registers before the absolute value (GA102).
void TranslatorVisitor::FMNMX(Instruction insn) {
    const bool nan{insn.Bit(81)};
    const bool xorsign{insn.Bit(82)};
    const IR::FpControl control{.fmz_mode = insn.Bit(80) ? IR::FmzMode::FTZ : IR::FmzMode::None};
    const IR::F32 a{FloatSrcA(insn)};
    const IR::F32 b{FloatSrcB(insn)};
    const IR::U1 pick_min{P(insn.Bits(87, 3), insn.Bit(90))};
    IR::F32 result{ir.Select(pick_min, ir.FPMin(a, b, control), ir.FPMax(a, b, control))};
    if (xorsign) {
        const IR::U32 raw_b{BIsWide(insn) ? Wide(insn) : Narrow(insn)};
        const IR::U32 sign{
            ir.BitwiseAnd(ir.BitwiseXor(SrcA(insn), raw_b), ir.Imm32(0x8000'0000U))};
        result = ir.BitCast<IR::F32>(ir.BitwiseOr(ir.BitCast<IR::U32>(result), sign));
    }
    if (nan) {
        const IR::U1 any_nan{ir.LogicalOr(ir.FPIsNan(a), ir.FPIsNan(b))};
        result = IR::F32{ir.Select(any_nan, ir.BitCast<IR::F32>(ir.Imm32(0x7fff'ffffU)), result)};
    }
    F(Reg(insn.Dest()), result);
}

void TranslatorVisitor::FSET(Instruction insn) {
    const auto compare_op{static_cast<Maxwell::FPCompareOp>(insn.Bits(76, 4))};
    const auto bop{static_cast<Maxwell::BooleanOp>(insn.Bits(74, 2))};
    const IR::FpControl control{.fmz_mode = insn.Bit(80) ? IR::FmzMode::FTZ : IR::FmzMode::None};
    const IR::U1 comparison{Maxwell::FloatingPointCompare(ir, FloatSrcA(insn), FloatSrcB(insn),
                                                          compare_op, control)};
    const IR::U1 accumulator{P(insn.Bits(87, 3), insn.Bit(90))};
    const IR::U1 result{Maxwell::PredicateCombine(ir, comparison, accumulator, bop)};
    F(Reg(insn.Dest()), IR::F32{ir.Select(result, ir.Imm32(1.0f), ir.Imm32(0.0f))});
}

void TranslatorVisitor::FSETP(Instruction insn) {
    const auto compare_op{static_cast<Maxwell::FPCompareOp>(insn.Bits(76, 4))};
    const auto bop{static_cast<Maxwell::BooleanOp>(insn.Bits(74, 2))};
    const IR::FpControl control{.fmz_mode = insn.Bit(80) ? IR::FmzMode::FTZ : IR::FmzMode::None};
    const IR::U1 comparison{Maxwell::FloatingPointCompare(ir, FloatSrcA(insn), FloatSrcB(insn),
                                                          compare_op, control)};
    const IR::U1 accumulator{P(insn.Bits(87, 3), insn.Bit(90))};
    SetP(insn.Bits(81, 3), Maxwell::PredicateCombine(ir, comparison, accumulator, bop));
    SetP(insn.Bits(84, 3),
         Maxwell::PredicateCombine(ir, ir.LogicalNot(comparison), accumulator, bop));
}

void TranslatorVisitor::FSEL(Instruction insn) {
    const IR::U1 cond{P(insn.Bits(87, 3), insn.Bit(90))};
    F(Reg(insn.Dest()), IR::F32{ir.Select(cond, FloatSrcA(insn), FloatSrcB(insn))});
}

// MUFU.F16 reads the half of Rb picked by bit 60 (.H1) with |x| at 62 and -x at 63, and writes
// the F16 result to the low half with the high half zeroed (checked on a GA102).
void TranslatorVisitor::MUFU(Instruction insn) {
    const bool is_f16{insn.Bit(73)};
    if (is_f16 && insn.Form() != AluForm::RegReg) {
        throw NotImplementedException("MUFU.F16 form {}", static_cast<u32>(insn.Form()));
    }
    IR::F32 src;
    if (is_f16) {
        const IR::Value halves{ir.UnpackFloat2x16(Wide(insn))};
        const IR::F16 half{ir.CompositeExtract(halves, insn.Bit(60) ? 1 : 0)};
        src = ir.FPConvert(32, IR::F16{ir.FPAbsNeg(half, insn.Bit(62), insn.Bit(63))});
    } else {
        src = FloatSrcB(insn);
    }
    // Sine and cosine take their argument in revolutions, the compiler scales it by 1/(2*pi)
    const auto radians{[&] { return IR::F32{ir.FPMul(src, ir.Imm32(2.0f * std::numbers::pi_v<f32>))}; }};
    IR::F32 result;
    switch (insn.Bits(74, 6)) {
    case 0:
        result = ir.FPCos(radians());
        break;
    case 1:
        result = ir.FPSin(radians());
        break;
    case 2:
        result = ir.FPExp2(src);
        break;
    case 3:
        result = ir.FPLog2(src);
        break;
    case 4:
        result = IR::F32{ir.FPRecip(src)};
        break;
    case 5:
        result = IR::F32{ir.FPRecipSqrt(src)};
        break;
    case 8:
        result = ir.FPSqrt(src);
        break;
    default:
        throw NotImplementedException("MUFU operation {}", insn.Bits(74, 6));
    }
    if (is_f16) {
        const IR::F16 half{ir.FPConvert(16, result)};
        X(Reg(insn.Dest()), ir.PackFloat2x16(ir.CompositeConstruct(half, IR::F16{ir.FPConvert(16, ir.Imm32(0.0f))})));
        return;
    }
    F(Reg(insn.Dest()), result);
}

} // namespace Shader::Ampere
