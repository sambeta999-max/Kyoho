// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <span>

#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ampere/translate/impl/impl.h"
#include "shader_recompiler/frontend/maxwell/translate/impl/common_funcs.h"
#include "shader_recompiler/frontend/maxwell/translate/impl/half_floating_point_helper.h"

namespace Shader::Ampere {
namespace {

using Maxwell::Swizzle;

constexpr u64 PT{7};

// A packed pair of halves, or an F32 broadcast to both lanes, with its modifiers
struct HalfOperand {
    IR::F16F32F64 lhs;
    IR::F16F32F64 rhs;
};

HalfOperand MakeOperand(TranslatorVisitor& v, const IR::U32& value, u64 swizzle, bool abs,
                        bool neg) {
    auto [lhs, rhs]{Maxwell::Extract(v.ir, value, static_cast<Swizzle>(swizzle))};
    return {v.ir.FPAbsNeg(lhs, abs, neg), v.ir.FPAbsNeg(rhs, abs, neg)};
}

HalfOperand OperandA(TranslatorVisitor& v, Instruction insn) {
    return MakeOperand(v, v.SrcA(insn), insn.Bits(74, 2), insn.Bit(73), insn.Bit(72));
}

// Immediates are two packed halves and carry no modifiers
HalfOperand OperandWide(TranslatorVisitor& v, Instruction insn) {
    const bool is_imm{insn.Form() == AluForm::RegImm || insn.Form() == AluForm::ImmReg};
    if (is_imm) {
        return MakeOperand(v, v.Wide(insn), 0, false, false);
    }
    return MakeOperand(v, v.Wide(insn), insn.Bits(60, 2), insn.Bit(62), insn.Bit(63));
}

HalfOperand OperandNarrow(TranslatorVisitor& v, Instruction insn) {
    return MakeOperand(v, v.Narrow(insn), insn.Bits(81, 2), insn.Bit(83), insn.Bit(84));
}

HalfOperand OperandB(TranslatorVisitor& v, Instruction insn) {
    return v.BIsWide(insn) ? OperandWide(v, insn) : OperandNarrow(v, insn);
}

HalfOperand OperandC(TranslatorVisitor& v, Instruction insn) {
    return v.BIsWide(insn) ? OperandNarrow(v, insn) : OperandWide(v, insn);
}

// Mixed F16 and F32 operands are computed in F32, as on earlier generations
bool Promote(IR::IREmitter& ir, std::span<HalfOperand* const> operands) {
    const bool promote{std::ranges::any_of(
        operands, [](const HalfOperand* op) { return op->lhs.Type() != IR::Type::F16; })};
    if (promote) {
        for (HalfOperand* const op : operands) {
            if (op->lhs.Type() == IR::Type::F16) {
                op->lhs = ir.FPConvert(32, op->lhs);
                op->rhs = ir.FPConvert(32, op->rhs);
            }
        }
    }
    return promote;
}

IR::FpControl Control(Instruction insn) {
    return {
        .no_contraction = true,
        .rounding = IR::FpRounding::DontCare,
        .fmz_mode = insn.Bit(80) ? IR::FmzMode::FTZ : IR::FmzMode::None,
    };
}

// Results are packed halves, or the low lane as F32 with .F32 at bit 78
void WriteResult(TranslatorVisitor& v, Instruction insn, IR::F16F32F64 lhs, IR::F16F32F64 rhs,
                 bool promoted, bool has_f32_output) {
    if (insn.Bit(77)) {
        lhs = v.ir.FPSaturate(lhs);
        rhs = v.ir.FPSaturate(rhs);
    }
    const IR::Reg dest{Reg(insn.Dest())};
    if (has_f32_output && insn.Bit(78)) {
        const IR::F32 value{promoted ? IR::F32{lhs} : IR::F32{v.ir.FPConvert(32, lhs)}};
        v.F(dest, value);
        return;
    }
    if (promoted) {
        lhs = v.ir.FPConvert(16, lhs);
        rhs = v.ir.FPConvert(16, rhs);
    }
    v.X(dest, v.ir.PackFloat2x16(v.ir.CompositeConstruct(lhs, rhs)));
}

std::pair<IR::U1, IR::U1> Compare(TranslatorVisitor& v, Instruction insn) {
    HalfOperand a{OperandA(v, insn)};
    HalfOperand b{OperandWide(v, insn)};
    const std::array<HalfOperand*, 2> operands{&a, &b};
    Promote(v.ir, operands);
    const auto op{static_cast<Maxwell::FPCompareOp>(insn.Bits(76, 4))};
    const IR::FpControl control{Control(insn)};
    return {Maxwell::FloatingPointCompare(v.ir, a.lhs, b.lhs, op, control),
            Maxwell::FloatingPointCompare(v.ir, a.rhs, b.rhs, op, control)};
}

} // Anonymous namespace

// The second operand of HADD2, HSET2 and HSETP2 is always on the [32..64) field
void TranslatorVisitor::HADD2(Instruction insn) {
    HalfOperand a{OperandA(*this, insn)};
    HalfOperand b{OperandWide(*this, insn)};
    const std::array<HalfOperand*, 2> operands{&a, &b};
    const bool promoted{Promote(ir, operands)};
    const IR::FpControl control{Control(insn)};
    WriteResult(*this, insn, ir.FPAdd(a.lhs, b.lhs, control), ir.FPAdd(a.rhs, b.rhs, control),
                promoted, true);
}

void TranslatorVisitor::HMUL2(Instruction insn) {
    HalfOperand a{OperandA(*this, insn)};
    HalfOperand b{OperandB(*this, insn)};
    const std::array<HalfOperand*, 2> operands{&a, &b};
    const bool promoted{Promote(ir, operands)};
    const IR::FpControl control{Control(insn)};
    WriteResult(*this, insn, ir.FPMul(a.lhs, b.lhs, control), ir.FPMul(a.rhs, b.rhs, control),
                promoted, false);
}

void TranslatorVisitor::HFMA2(Instruction insn) {
    if (insn.Bit(79)) {
        throw NotImplementedException("HFMA2.RELU");
    }
    HalfOperand a{OperandA(*this, insn)};
    HalfOperand b{OperandB(*this, insn)};
    HalfOperand c{OperandC(*this, insn)};
    const std::array<HalfOperand*, 3> operands{&a, &b, &c};
    const bool promoted{Promote(ir, operands)};
    const IR::FpControl control{Control(insn)};
    WriteResult(*this, insn, ir.FPFma(a.lhs, b.lhs, c.lhs, control),
                ir.FPFma(a.rhs, b.rhs, c.rhs, control), promoted, true);
}

void TranslatorVisitor::HMNMX2(Instruction insn) {
    if (insn.Bit(81) || insn.Bit(82)) {
        throw NotImplementedException("HMNMX2.NAN or .XORSIGN");
    }
    HalfOperand a{OperandA(*this, insn)};
    HalfOperand b{OperandB(*this, insn)};
    // F16 min and max are computed in F32; the conversion is exact both ways
    for (HalfOperand* const op : std::array{&a, &b}) {
        if (op->lhs.Type() == IR::Type::F16) {
            op->lhs = ir.FPConvert(32, op->lhs);
            op->rhs = ir.FPConvert(32, op->rhs);
        }
    }
    const IR::FpControl control{Control(insn)};
    const IR::U1 pick_min{P(insn.Bits(87, 3), insn.Bit(90))};
    const auto select{[&](const IR::F16F32F64& x, const IR::F16F32F64& y) {
        const IR::F32 lhs{x};
        const IR::F32 rhs{y};
        return IR::F16F32F64{
            ir.Select(pick_min, ir.FPMin(lhs, rhs, control), ir.FPMax(lhs, rhs, control))};
    }};
    WriteResult(*this, insn, select(a.lhs, b.lhs), select(a.rhs, b.rhs), true, true);
}

// Each lane compares and combines with the accumulator; .BF writes 1.0 instead of all ones
void TranslatorVisitor::HSET2(Instruction insn) {
    const auto [lhs, rhs]{Compare(*this, insn)};
    const auto bop{static_cast<Maxwell::BooleanOp>(insn.Bits(69, 2))};
    const IR::U1 accumulator{P(insn.Bits(87, 3), insn.Bit(90))};
    const IR::U32 true_value{ir.Imm32(insn.Bit(71) ? 0x3c00u : 0xffffu)};
    const auto lane{[&](const IR::U1& cmp) {
        return IR::U32{ir.Select(Maxwell::PredicateCombine(ir, cmp, accumulator, bop), true_value,
                                 ir.Imm32(0u))};
    }};
    X(Reg(insn.Dest()), ir.BitFieldInsert(lane(lhs), lane(rhs), ir.Imm32(16), ir.Imm32(16)));
}

void TranslatorVisitor::HSETP2(Instruction insn) {
    const auto [lhs, rhs]{Compare(*this, insn)};
    const auto bop{static_cast<Maxwell::BooleanOp>(insn.Bits(69, 2))};
    const IR::U1 accumulator{P(insn.Bits(87, 3), insn.Bit(90))};
    IR::U1 first;
    IR::U1 second;
    if (insn.Bit(71)) {
        // .H_AND folds both lanes into one comparison
        const IR::U1 both{ir.LogicalAnd(lhs, rhs)};
        first = Maxwell::PredicateCombine(ir, both, accumulator, bop);
        second = Maxwell::PredicateCombine(ir, ir.LogicalNot(both), accumulator, bop);
    } else {
        first = Maxwell::PredicateCombine(ir, lhs, accumulator, bop);
        second = Maxwell::PredicateCombine(ir, rhs, accumulator, bop);
    }
    if (insn.Bits(81, 3) != PT) {
        SetP(insn.Bits(81, 3), first);
    }
    if (insn.Bits(84, 3) != PT) {
        SetP(insn.Bits(84, 3), second);
    }
}

// F2FP.PACK_AB converts a to the upper and b to the lower half
void TranslatorVisitor::F2FP(Instruction insn) {
    if (insn.Bit(78)) {
        throw NotImplementedException("F2FP.MERGE_C");
    }
    constexpr std::array rounding{IR::FpRounding::RN, IR::FpRounding::RM, IR::FpRounding::RP,
                                  IR::FpRounding::RZ};
    const IR::FpControl control{.no_contraction = true, .rounding = rounding[insn.Bits(79, 2)]};
    const IR::F16 high{ir.FPConvert(16, F(Reg(insn.SrcA())), control)};
    const IR::F16 low{ir.FPConvert(16, ir.BitCast<IR::F32>(SrcB(insn)), control)};
    X(Reg(insn.Dest()), ir.PackFloat2x16(ir.CompositeConstruct(low, high)));
}

} // namespace Shader::Ampere
