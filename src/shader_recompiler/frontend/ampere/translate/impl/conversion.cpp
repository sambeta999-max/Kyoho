// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cmath>

#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ampere/translate/impl/impl.h"

namespace Shader::Ampere {
namespace {

// Sizes at [75..77) and [84..86) are log2 of the byte count
u32 DestBits(Instruction insn) {
    return 8u << insn.Bits(75, 2);
}

u32 SrcBits(Instruction insn) {
    return 8u << insn.Bits(84, 2);
}

IR::FpRounding Rounding(Instruction insn) {
    constexpr std::array rounding{IR::FpRounding::RN, IR::FpRounding::RM, IR::FpRounding::RP,
                                  IR::FpRounding::RZ};
    return rounding[insn.Bits(78, 2)];
}

IR::FmzMode Ftz(Instruction insn) {
    return insn.Bit(80) ? IR::FmzMode::FTZ : IR::FmzMode::None;
}

// Half precision sources select their half with bit 60
IR::F16F32F64 FloatSource(TranslatorVisitor& v, Instruction insn) {
    switch (SrcBits(insn)) {
    case 16: {
        const IR::Value halves{v.ir.UnpackFloat2x16(v.SrcB(insn))};
        const IR::F16 half{v.ir.CompositeExtract(halves, insn.Bit(60) ? 1 : 0)};
        const bool is_imm{insn.Form() == AluForm::ImmReg};
        return v.ir.FPAbsNeg(half, !is_imm && insn.Bit(62), !is_imm && insn.Bit(63));
    }
    case 32:
        return v.FloatSrcB(insn);
    default:
        throw NotImplementedException("Floating point source of {} bits", SrcBits(insn));
    }
}

IR::U32 PackHalf(IR::IREmitter& ir, const IR::F16& value) {
    return ir.PackFloat2x16(ir.CompositeConstruct(value, ir.FPConvert(16, ir.Imm32(0.0f))));
}

} // Anonymous namespace

void TranslatorVisitor::F2F(Instruction insn) {
    const IR::FpControl control{
        .no_contraction = true,
        .rounding = Rounding(insn),
        .fmz_mode = Ftz(insn),
    };
    const IR::F16F32F64 src{FloatSource(*this, insn)};
    switch (DestBits(insn)) {
    case 16:
        X(Reg(insn.Dest()), PackHalf(ir, IR::F16{ir.FPConvert(16, src, control)}));
        return;
    case 32:
        F(Reg(insn.Dest()), IR::F32{SrcBits(insn) == 32 ? ir.FPAdd(src, ir.Imm32(-0.0f), control)
                                                         : ir.FPConvert(32, src, control)});
        return;
    default:
        throw NotImplementedException("F2F to {} bits", DestBits(insn));
    }
}

// Rounds to an integral value in the source format. The rounding mode at [78..80) and FTZ at bit
// 80 are those of F2F, checked with nvdisasm.
void TranslatorVisitor::FRND(Instruction insn) {
    if (DestBits(insn) != 32 || SrcBits(insn) != 32) {
        throw NotImplementedException("FRND of {} to {} bits", SrcBits(insn), DestBits(insn));
    }
    const IR::FpControl control{.no_contraction = true, .fmz_mode = Ftz(insn)};
    const IR::F32 value{FloatSrcB(insn)};
    IR::F32 result;
    switch (insn.Bits(78, 2)) {
    case 0:
        result = IR::F32{ir.FPRoundEven(value, control)};
        break;
    case 1:
        result = IR::F32{ir.FPFloor(value, control)};
        break;
    case 2:
        result = IR::F32{ir.FPCeil(value, control)};
        break;
    default:
        result = IR::F32{ir.FPTrunc(value, control)};
        break;
    }
    F(Reg(insn.Dest()), result);
}

void TranslatorVisitor::F2I(Instruction insn) {
    const u32 bits{DestBits(insn)};
    if (bits > 32) {
        throw NotImplementedException("F2I to {} bits", bits);
    }
    const bool is_signed{insn.Bit(72)};
    const bool nan_to_zero{insn.Bit(77)};
    const IR::FpControl control{.no_contraction = true, .fmz_mode = Ftz(insn)};
    const IR::F16F32F64 src{FloatSource(*this, insn)};
    const IR::F32 value{SrcBits(insn) == 32 ? IR::F32{src} : IR::F32{ir.FPConvert(32, src)}};
    IR::F32 rounded;
    switch (insn.Bits(78, 2)) {
    case 0:
        rounded = IR::F32{ir.FPRoundEven(value, control)};
        break;
    case 1:
        rounded = IR::F32{ir.FPFloor(value, control)};
        break;
    case 2:
        rounded = IR::F32{ir.FPCeil(value, control)};
        break;
    default:
        rounded = IR::F32{ir.FPTrunc(value, control)};
        break;
    }
    // Out of range values saturate. NaN converts to zero with .NTZ, otherwise to the top bit of
    // the destination width, zero-extended; both measured on GA102.
    // The integer maximum is not representable as F32 for 32-bit destinations, so values at or
    // above the next power of two are selected explicitly and the rest are clamped below it.
    const u32 value_bits{is_signed ? bits - 1 : bits};
    const f32 limit{static_cast<f32>(1ull << value_bits)};
    const u32 max_int{static_cast<u32>((1ull << value_bits) - 1)};
    const f32 min_value{is_signed ? -limit : 0.0f};
    const IR::F32 clamped{ir.FPClamp(rounded, ir.Imm32(min_value),
                                     ir.Imm32(std::nextafter(limit, 0.0f)))};
    IR::U32 result{ir.ConvertFToI(32, is_signed, clamped)};
    result = IR::U32{ir.Select(ir.FPGreaterThanEqual(rounded, ir.Imm32(limit)), ir.Imm32(max_int),
                               result)};
    const IR::U32 nan_value{ir.Imm32(nan_to_zero ? 0u : 1u << (bits - 1))};
    X(Reg(insn.Dest()), IR::U32{ir.Select(ir.FPIsNan(value), nan_value, result)});
}

void TranslatorVisitor::I2F(Instruction insn) {
    if (DestBits(insn) != 32) {
        throw NotImplementedException("I2F to {} bits", DestBits(insn));
    }
    const bool is_signed{insn.Bit(74)};
    const u32 src_bits{SrcBits(insn)};
    IR::U32 src{SrcB(insn)};
    if (src_bits < 32) {
        // Narrow sources select their byte or half with [60..62)
        const u32 offset{static_cast<u32>(insn.Bits(60, 2)) * src_bits};
        src = ir.BitFieldExtract(src, ir.Imm32(offset), ir.Imm32(src_bits), is_signed);
    } else if (src_bits > 32) {
        throw NotImplementedException("I2F from {} bits", src_bits);
    }
    const IR::FpControl control{.no_contraction = true, .rounding = Rounding(insn)};
    F(Reg(insn.Dest()), IR::F32{ir.ConvertIToF(32, 32, is_signed, src, control)});
}

void TranslatorVisitor::I2FP(Instruction insn) {
    if (DestBits(insn) != 32 || SrcBits(insn) != 32) {
        throw NotImplementedException("I2FP format");
    }
    const IR::FpControl control{.no_contraction = true, .rounding = Rounding(insn)};
    F(Reg(insn.Dest()), IR::F32{ir.ConvertIToF(32, 32, insn.Bit(74), SrcB(insn), control)});
}

} // namespace Shader::Ampere
