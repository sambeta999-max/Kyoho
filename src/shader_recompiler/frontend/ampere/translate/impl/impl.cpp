// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ampere/translate/impl/impl.h"

namespace Shader::Ampere {

IR::U32 TranslatorVisitor::X(IR::Reg reg) {
    return ir.GetReg(reg);
}

IR::F32 TranslatorVisitor::F(IR::Reg reg) {
    return ir.BitCast<IR::F32>(X(reg));
}

IR::U64 TranslatorVisitor::L(IR::Reg reg) {
    if (!IR::IsAligned(reg, 2)) {
        throw NotImplementedException("Unaligned source register {}", reg);
    }
    return IR::U64{ir.PackUint2x32(ir.CompositeConstruct(X(reg), X(reg + 1)))};
}

void TranslatorVisitor::X(IR::Reg dest_reg, const IR::U32& value) {
    ++register_writes[static_cast<u32>(dest_reg)];
    ir.SetReg(dest_reg, value);
}

void TranslatorVisitor::F(IR::Reg dest_reg, const IR::F32& value) {
    X(dest_reg, ir.BitCast<IR::U32>(value));
}

void TranslatorVisitor::L(IR::Reg dest_reg, const IR::U64& value) {
    if (!IR::IsAligned(dest_reg, 2)) {
        throw NotImplementedException("Unaligned destination register {}", dest_reg);
    }
    const IR::Value result{ir.UnpackUint2x32(value)};
    for (int i = 0; i < 2; ++i) {
        X(dest_reg + i, IR::U32{ir.CompositeExtract(result, static_cast<size_t>(i))});
    }
}

IR::U1 TranslatorVisitor::P(u64 index, bool negated) {
    return ir.GetPred(static_cast<IR::Pred>(index & 7), negated);
}

IR::U1 TranslatorVisitor::UP(u64 index, bool negated) {
    return ir.GetPred(IR::UniformPred(static_cast<u32>(index)), negated);
}

void TranslatorVisitor::SetP(u64 index, const IR::U1& value) {
    ir.SetPred(static_cast<IR::Pred>(index & 7), value);
}

void TranslatorVisitor::SetUP(u64 index, const IR::U1& value) {
    ir.SetPred(IR::UniformPred(static_cast<u32>(index)), value);
}

bool TranslatorVisitor::BIsWide(Instruction insn) const noexcept {
    switch (insn.Form()) {
    case AluForm::RegReg:
    case AluForm::ImmReg:
    case AluForm::CbufReg:
    case AluForm::URegReg:
        return true;
    default:
        return false;
    }
}

IR::U32 TranslatorVisitor::Cbuf(Instruction insn) {
    if (insn.CbufBindless()) {
        throw NotImplementedException("Bindless constant buffer operand");
    }
    return ir.GetCbuf(ir.Imm32(insn.CbufIndex()), ir.Imm32(insn.CbufOffset()));
}

IR::U32 TranslatorVisitor::Wide(Instruction insn, bool uniform) {
    switch (insn.Form()) {
    case AluForm::RegReg:
        return X(uniform ? UReg(insn.Bits(32, 6)) : Reg(insn.Bits(32, 8)));
    case AluForm::RegImm:
    case AluForm::ImmReg:
        return ir.Imm32(insn.Imm32());
    case AluForm::RegCbuf:
    case AluForm::CbufReg:
        return Cbuf(insn);
    case AluForm::URegReg:
    case AluForm::RegUReg:
        return X(UReg(insn.Bits(32, 6)));
    }
    throw NotImplementedException("ALU form {}", static_cast<u32>(insn.Form()));
}

IR::U32 TranslatorVisitor::Narrow(Instruction insn, bool uniform) {
    return X(uniform ? UReg(insn.Bits(64, 6)) : Reg(insn.Bits(64, 8)));
}

IR::U32 TranslatorVisitor::SrcA(Instruction insn, bool uniform) {
    return X(uniform ? UReg(insn.Bits(24, 6)) : Reg(insn.Bits(24, 8)));
}

IR::U32 TranslatorVisitor::SrcB(Instruction insn, bool uniform) {
    return BIsWide(insn) ? Wide(insn, uniform) : Narrow(insn, uniform);
}

IR::U32 TranslatorVisitor::SrcC(Instruction insn, bool uniform) {
    return BIsWide(insn) ? Narrow(insn, uniform) : Wide(insn, uniform);
}

namespace {
bool WideIsImm(Instruction insn) {
    return insn.Form() == AluForm::RegImm || insn.Form() == AluForm::ImmReg;
}

// Immediates take the whole [32..64) field, so they carry no modifier bits
bool WideAbs(Instruction insn) {
    return !WideIsImm(insn) && insn.Bit(62);
}

bool WideNeg(Instruction insn) {
    return !WideIsImm(insn) && insn.Bit(63);
}
} // Anonymous namespace

IR::F32 TranslatorVisitor::FloatWide(Instruction insn) {
    return ir.FPAbsNeg(ir.BitCast<IR::F32>(Wide(insn)), WideAbs(insn), WideNeg(insn));
}

IR::F32 TranslatorVisitor::FloatNarrow(Instruction insn) {
    return ir.FPAbsNeg(ir.BitCast<IR::F32>(Narrow(insn)), insn.Bit(74), insn.Bit(75));
}

IR::F32 TranslatorVisitor::FloatSrcA(Instruction insn) {
    return ir.FPAbsNeg(ir.BitCast<IR::F32>(SrcA(insn)), insn.Bit(73), insn.Bit(72));
}

IR::F32 TranslatorVisitor::FloatSrcB(Instruction insn) {
    return BIsWide(insn) ? FloatWide(insn) : FloatNarrow(insn);
}

IR::F32 TranslatorVisitor::FloatSrcC(Instruction insn) {
    return BIsWide(insn) ? FloatNarrow(insn) : FloatWide(insn);
}

bool TranslatorVisitor::NegA(Instruction insn) const noexcept {
    return insn.Bit(72);
}

bool TranslatorVisitor::NegB(Instruction insn) const noexcept {
    return BIsWide(insn) ? WideNeg(insn) : insn.Bit(75);
}

bool TranslatorVisitor::NegC(Instruction insn) const noexcept {
    return BIsWide(insn) ? insn.Bit(75) : WideNeg(insn);
}

} // namespace Shader::Ampere
