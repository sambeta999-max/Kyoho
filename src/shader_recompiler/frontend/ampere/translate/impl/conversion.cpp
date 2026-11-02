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
// FRND.F16 rounds the selected half and zeroes the upper half of the destination (GA102); every
// F16 value is exact in F32, so it rounds in F32.
void TranslatorVisitor::FRND(Instruction insn) {
    const bool half{DestBits(insn) == 16 && SrcBits(insn) == 16};
    if (!half && (DestBits(insn) != 32 || SrcBits(insn) != 32)) {
        throw NotImplementedException("FRND of {} to {} bits", SrcBits(insn), DestBits(insn));
    }
    const IR::FpControl control{.no_contraction = true, .fmz_mode = Ftz(insn)};
    const IR::F32 value{half ? IR::F32{ir.FPConvert(32, FloatSource(*this, insn))}
                             : FloatSrcB(insn)};
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
    if (half) {
        X(Reg(insn.Dest()), PackHalf(ir, IR::F16{ir.FPConvert(16, result)}));
        return;
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

// I2F.F16 writes the half to the lower 16 bits and zeroes the upper ones (GA102).
void TranslatorVisitor::I2F(Instruction insn) {
    if (DestBits(insn) != 32 && DestBits(insn) != 16) {
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
    if (DestBits(insn) == 16) {
        X(Reg(insn.Dest()),
          PackHalf(ir, IR::F16{ir.ConvertIToF(16, 32, is_signed, src, control)}));
        return;
    }
    F(Reg(insn.Dest()), IR::F32{ir.ConvertIToF(32, 32, is_signed, src, control)});
}

// F2I with a 64-bit source or destination: an F32 or F64 in, a register or a pair out. Computed in
// F64, which holds every F32 exactly. Out of range values saturate and NaN follows .NTZ as for F2I.
void TranslatorVisitor::F2I_F64(Instruction insn) {
    const u32 bits{DestBits(insn)};
    const u32 src_bits{SrcBits(insn)};
    if ((bits != 32 && bits != 64) || (src_bits != 32 && src_bits != 64) ||
        insn.Form() != AluForm::RegReg) {
        throw NotImplementedException("F2I.F64 sizes {} {} form {}", bits, src_bits,
                                      static_cast<u32>(insn.Form()));
    }
    const bool is_signed{insn.Bit(72)};
    const IR::FpControl control{.no_contraction = true, .fmz_mode = Ftz(insn)};
    const IR::Reg src_reg{Reg(insn.Bits(32, 8))};
    IR::F64 value;
    if (src_bits == 64) {
        value = ir.PackDouble2x32(ir.CompositeConstruct(X(src_reg), X(src_reg + 1)));
    } else {
        value = IR::F64{ir.FPConvert(64, F(src_reg))};
    }
    value = IR::F64{ir.FPAbsNeg(value, insn.Bit(62), insn.Bit(63))};
    IR::F64 rounded;
    switch (insn.Bits(78, 2)) {
    case 0:
        rounded = IR::F64{ir.FPRoundEven(value, control)};
        break;
    case 1:
        rounded = IR::F64{ir.FPFloor(value, control)};
        break;
    case 2:
        rounded = IR::F64{ir.FPCeil(value, control)};
        break;
    default:
        rounded = IR::F64{ir.FPTrunc(value, control)};
        break;
    }
    const u32 value_bits{is_signed ? bits - 1 : bits};
    const f64 limit{std::ldexp(1.0, static_cast<int>(value_bits))};
    const u64 max_int{value_bits == 64 ? ~u64{0} : (u64{1} << value_bits) - 1};
    const IR::F64 clamped{ir.FPClamp(rounded, ir.Imm64(is_signed ? -limit : 0.0),
                                     ir.Imm64(std::nextafter(limit, 0.0)))};
    IR::U64 result{ir.ConvertFToI(64, is_signed, clamped)};
    result = IR::U64{ir.Select(ir.FPGreaterThanEqual(rounded, ir.Imm64(limit)), ir.Imm64(max_int),
                               result)};
    const IR::U64 nan_value{ir.Imm64(insn.Bit(77) ? u64{0} : u64{1} << (bits - 1))};
    result = IR::U64{ir.Select(ir.FPIsNan(value), nan_value, result)};
    const IR::Value words{ir.UnpackUint2x32(result)};
    const IR::Reg dest{Reg(insn.Dest())};
    X(dest, IR::U32{ir.CompositeExtract(words, 0)});
    if (bits == 64) {
        X(dest + 1, IR::U32{ir.CompositeExtract(words, 1)});
    }
}

// I2F from a 64-bit integer in a register pair, to an F32 or an F64 in a pair
void TranslatorVisitor::I2F_64(Instruction insn) {
    const u32 bits{DestBits(insn)};
    if ((bits != 32 && bits != 64) || SrcBits(insn) != 64 || insn.Form() != AluForm::RegReg) {
        throw NotImplementedException("I2F.64 sizes {} {} form {}", bits, SrcBits(insn),
                                      static_cast<u32>(insn.Form()));
    }
    const IR::Reg src_reg{Reg(insn.Bits(32, 8))};
    const IR::U64 src{ir.PackUint2x32(ir.CompositeConstruct(X(src_reg), X(src_reg + 1)))};
    const IR::FpControl control{.no_contraction = true, .rounding = Rounding(insn)};
    const IR::F16F32F64 result{ir.ConvertIToF(bits, 64, insn.Bit(74), src, control)};
    const IR::Reg dest{Reg(insn.Dest())};
    if (bits == 32) {
        F(dest, IR::F32{result});
        return;
    }
    const IR::Value words{ir.UnpackDouble2x32(IR::F64{result})};
    X(dest, IR::U32{ir.CompositeExtract(words, 0)});
    X(dest + 1, IR::U32{ir.CompositeExtract(words, 1)});
}

void TranslatorVisitor::I2FP(Instruction insn) {
    if (DestBits(insn) != 32 || SrcBits(insn) != 32) {
        throw NotImplementedException("I2FP format");
    }
    const IR::FpControl control{.no_contraction = true, .rounding = Rounding(insn)};
    F(Reg(insn.Dest()), IR::F32{ir.ConvertIToF(32, 32, insn.Bit(74), SrcB(insn), control)});
}

} // namespace Shader::Ampere
