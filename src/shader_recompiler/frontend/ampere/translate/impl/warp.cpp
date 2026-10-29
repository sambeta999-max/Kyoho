// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ampere/translate/impl/impl.h"

namespace Shader::Ampere {
namespace {

constexpr u64 PT{7};

enum class ShuffleMode : u64 {
    Idx,
    Up,
    Down,
    Bfly,
};

// Clamp at bits [0..5) and segment mask at [8..13) of the c operand
constexpr u32 QUAD_MASK{(28u << 8) | 3u};

IR::U32 Shuffle(IR::IREmitter& ir, const IR::U32& value, const IR::U32& lane, const IR::U32& c,
                ShuffleMode mode) {
    const IR::U32 clamp{ir.BitFieldExtract(c, ir.Imm32(0), ir.Imm32(5))};
    const IR::U32 seg_mask{ir.BitFieldExtract(c, ir.Imm32(8), ir.Imm32(5))};
    switch (mode) {
    case ShuffleMode::Idx:
        return ir.ShuffleIndex(value, lane, clamp, seg_mask);
    case ShuffleMode::Up:
        return ir.ShuffleUp(value, lane, clamp, seg_mask);
    case ShuffleMode::Down:
        return ir.ShuffleDown(value, lane, clamp, seg_mask);
    case ShuffleMode::Bfly:
        return ir.ShuffleButterfly(value, lane, clamp, seg_mask);
    }
    throw NotImplementedException("SHFL mode {}", static_cast<u64>(mode));
}

} // Anonymous namespace

// The lane is a register at [32..40) or an immediate at [53..58), c a register at [64..72) or an
// immediate at [40..53), selected by the opcode form
void TranslatorVisitor::SHFL(Instruction insn) {
    const bool lane_is_imm{insn.Form() == AluForm::ImmReg || insn.Form() == AluForm::RegUReg};
    const bool c_is_imm{insn.Form() == AluForm::RegImm || insn.Form() == AluForm::RegUReg};
    const u32 lane_imm{static_cast<u32>(insn.Bits(53, 5))};
    const u32 c_imm{static_cast<u32>(insn.Bits(40, 13))};
    const IR::U32 lane{lane_is_imm ? ir.Imm32(lane_imm) : X(Reg(insn.SrcBReg()))};
    const IR::U32 c{c_is_imm ? ir.Imm32(c_imm) : X(Reg(insn.SrcCReg()))};
    const auto mode{static_cast<ShuffleMode>(insn.Bits(58, 2))};
    const IR::U32 value{SrcA(insn)};
    const IR::Reg dest{Reg(insn.Dest())};
    const u64 in_bounds_pred{insn.Bits(81, 3)};

    // Quad shuffles in fragment shaders map to quad operations
    if (c_is_imm && c_imm == QUAD_MASK && lane_is_imm && env.ShaderStage() == Stage::Fragment) {
        if (mode == ShuffleMode::Idx && lane_imm <= 3) {
            X(dest, ir.QuadBroadcast(value, ir.Imm32(lane_imm)));
            if (in_bounds_pred != PT) {
                SetP(in_bounds_pred, ir.Imm1(true));
            }
            return;
        }
        if (mode == ShuffleMode::Bfly && lane_imm >= 1 && lane_imm <= 3) {
            X(dest, ir.QuadSwap(value, ir.Imm32(lane_imm - 1)));
            if (in_bounds_pred != PT) {
                SetP(in_bounds_pred, ir.Imm1(true));
            }
            return;
        }
    }
    const IR::U32 result{Shuffle(ir, value, lane, c, mode)};
    if (in_bounds_pred != PT) {
        SetP(in_bounds_pred, ir.GetInBoundsFromOp(result));
    }
    X(dest, result);
}

namespace {
// Vote on the predicate at [87..91) with the operation at [72..74); the ballot is returned too
std::pair<IR::U1, IR::U32> Vote(TranslatorVisitor& v, Instruction insn) {
    const IR::U1 pred{v.P(insn.Bits(87, 3), insn.Bit(90))};
    IR::U1 vote;
    switch (insn.Bits(72, 2)) {
    case 0:
        vote = v.ir.VoteAll(pred);
        break;
    case 1:
        vote = v.ir.VoteAny(pred);
        break;
    case 2:
        vote = v.ir.VoteEqual(pred);
        break;
    default:
        throw NotImplementedException("VOTE operation {}", insn.Bits(72, 2));
    }
    return {vote, v.ir.SubgroupBallot(pred)};
}
} // Anonymous namespace

void TranslatorVisitor::VOTE(Instruction insn) {
    const auto [vote, ballot]{Vote(*this, insn)};
    if (insn.Bits(81, 3) != PT) {
        SetP(insn.Bits(81, 3), vote);
    }
    X(Reg(insn.Dest()), ballot);
}

void TranslatorVisitor::VOTEU(Instruction insn) {
    const auto [vote, ballot]{Vote(*this, insn)};
    if (insn.Bits(81, 3) != PT) {
        SetUP(insn.Bits(81, 3), vote);
    }
    X(UReg(insn.Bits(16, 6)), ballot);
}

} // namespace Shader::Ampere
