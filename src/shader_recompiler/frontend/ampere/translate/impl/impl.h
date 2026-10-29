// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "shader_recompiler/environment.h"
#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "shader_recompiler/frontend/ampere/instruction.h"

namespace Shader::Ampere {

[[nodiscard]] constexpr IR::Reg Reg(u64 index) noexcept {
    return static_cast<IR::Reg>(index & 0xff);
}

[[nodiscard]] constexpr IR::Reg UReg(u64 index) noexcept {
    return IR::UniformReg(static_cast<u32>(index & 0x3f));
}

struct TranslatorVisitor {
    explicit TranslatorVisitor(Environment& env_, IR::Block& block) : env{env_}, ir(block) {}

    Environment& env;
    IR::IREmitter ir;

    // Integer
    void IADD3(Instruction insn);
    void UIADD3(Instruction insn);
    void IMAD(Instruction insn);
    void IMAD_WIDE(Instruction insn);
    void IMNMX(Instruction insn);
    void ISETP(Instruction insn);
    void LOP3(Instruction insn);
    void ULOP3(Instruction insn);
    void SHF(Instruction insn);
    void USHF(Instruction insn);
    void SEL(Instruction insn);
    void POPC(Instruction insn);
    void FLO(Instruction insn);
    void BREV(Instruction insn);
    void PRMT(Instruction insn);
    void UPRMT(Instruction insn);
    void LEA(Instruction insn);

    // Conversion
    void F2F(Instruction insn);
    void F2I(Instruction insn);
    void I2F(Instruction insn);
    void I2FP(Instruction insn);

    // Memory
    void LDG(Instruction insn);
    void STG(Instruction insn);
    void LDS(Instruction insn);
    void STS(Instruction insn);
    void LDL(Instruction insn);
    void STL(Instruction insn);

    // Attributes
    void ALD(Instruction insn);
    void AST(Instruction insn);
    void IPA(Instruction insn);

    // Textures, bound through a constant buffer or bindless
    void TEX(Instruction insn);
    void TLD(Instruction insn);
    void TLD4(Instruction insn);
    void TXD(Instruction insn);
    void TXQ(Instruction insn);

    // Warp
    void SHFL(Instruction insn);
    void VOTE(Instruction insn);
    void VOTEU(Instruction insn);

    // Floating point
    void FADD(Instruction insn);
    void FMUL(Instruction insn);
    void FFMA(Instruction insn);
    void FMNMX(Instruction insn);
    void FSETP(Instruction insn);
    void FSEL(Instruction insn);
    void MUFU(Instruction insn);

    // Half precision pairs
    void HADD2(Instruction insn);
    void HMUL2(Instruction insn);
    void HFMA2(Instruction insn);
    void HMNMX2(Instruction insn);
    void HSET2(Instruction insn);
    void HSETP2(Instruction insn);
    void F2FP(Instruction insn);

    // Moves and constants
    void MOV(Instruction insn);
    void UMOV(Instruction insn);
    void R2UR(Instruction insn);
    void S2R(Instruction insn);
    void S2UR(Instruction insn);
    void CS2R(Instruction insn);
    void LDC(Instruction insn);
    void ULDC(Instruction insn);

    // Control and synchronization
    void EXIT(Instruction insn);
    void BAR(Instruction insn);
    void MEMBAR(Instruction insn);

    [[nodiscard]] IR::U32 X(IR::Reg reg);
    [[nodiscard]] IR::F32 F(IR::Reg reg);
    [[nodiscard]] IR::U64 L(IR::Reg reg);
    void X(IR::Reg dest_reg, const IR::U32& value);
    void F(IR::Reg dest_reg, const IR::F32& value);
    void L(IR::Reg dest_reg, const IR::U64& value);

    [[nodiscard]] IR::U1 P(u64 index, bool negated = false);
    [[nodiscard]] IR::U1 UP(u64 index, bool negated = false);
    void SetP(u64 index, const IR::U1& value);
    void SetUP(u64 index, const IR::U1& value);

    // ALU operands. Operand b and c are placed on the [32..64) and [64..72) fields depending on
    // the form; uniform datapath instructions read uniform registers for every register operand.
    [[nodiscard]] bool BIsWide(Instruction insn) const noexcept;
    [[nodiscard]] IR::U32 SrcA(Instruction insn, bool uniform = false);
    [[nodiscard]] IR::U32 SrcB(Instruction insn, bool uniform = false);
    [[nodiscard]] IR::U32 SrcC(Instruction insn, bool uniform = false);
    [[nodiscard]] IR::U32 Wide(Instruction insn, bool uniform = false);
    [[nodiscard]] IR::U32 Narrow(Instruction insn, bool uniform = false);
    [[nodiscard]] IR::U32 Cbuf(Instruction insn);

    // Floating point operands with absolute and negate modifiers applied
    [[nodiscard]] IR::F32 FloatSrcA(Instruction insn);
    [[nodiscard]] IR::F32 FloatSrcB(Instruction insn);
    [[nodiscard]] IR::F32 FloatSrcC(Instruction insn);
    [[nodiscard]] IR::F32 FloatWide(Instruction insn);
    [[nodiscard]] IR::F32 FloatNarrow(Instruction insn);

    // Integer negate modifier of each operand
    [[nodiscard]] bool NegA(Instruction insn) const noexcept;
    [[nodiscard]] bool NegB(Instruction insn) const noexcept;
    [[nodiscard]] bool NegC(Instruction insn) const noexcept;
};

} // namespace Shader::Ampere
