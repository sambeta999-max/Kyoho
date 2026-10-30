// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <optional>
#include <vector>

#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ampere/decode.h"
#include "shader_recompiler/frontend/ampere/instruction.h"
#include "shader_recompiler/frontend/ampere/location.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"

namespace Shader::Maxwell::Flow {
namespace {

using Ampere::Instruction;
using Ampere::Opcode;

constexpr u32 PT{7};

// A predicate source that is never true yields no condition
std::optional<IR::Condition> GuardCondition(const Instruction& insn) {
    if (insn.Pred() == PT) {
        return insn.PredNegated() ? std::nullopt : std::optional{IR::Condition{true}};
    }
    return IR::Condition{static_cast<IR::Pred>(insn.Pred()), insn.PredNegated()};
}

// Branch condition at [87..90], or a uniform predicate at [24..28] for the .U form
std::optional<IR::Condition> BranchCondition(const Instruction& insn) {
    if (insn.Opcode() == 0x947 && insn.Bit(91)) {
        // BRA.CONV (mode 3) and BRA.DIV (mode 2) test the convergence of the warp against a
        // uniform register mask; they guard fast and slow paths of warp-wide operations.
        // Invocations are emulated as converged: CONV is always taken, DIV never.
        switch (insn.Bits(32, 2)) {
        case 3:
            return IR::Condition{true};
        case 2:
            return std::nullopt;
        default:
            throw NotImplementedException("BRA convergence mode {}", insn.Bits(32, 2));
        }
    }
    if (insn.Bit(32)) {
        const u32 index{static_cast<u32>(insn.Bits(24, 3))};
        const bool negated{insn.Bit(27)};
        if (index == PT) {
            return negated ? std::nullopt : std::optional{IR::Condition{true}};
        }
        return IR::Condition{IR::UniformPred(index), negated};
    }
    const u32 index{static_cast<u32>(insn.Bits(87, 3))};
    const bool negated{insn.Bit(90)};
    if (index == PT) {
        return negated ? std::nullopt : std::optional{IR::Condition{true}};
    }
    return IR::Condition{static_cast<IR::Pred>(index), negated};
}

bool IsTrue(const IR::Condition& cond) {
    return cond == IR::Condition{true};
}

// Combines the guard with the instruction's own condition, nullopt when never taken
std::optional<IR::Condition> CombinedCondition(const Instruction& insn) {
    const std::optional guard{GuardCondition(insn)};
    const std::optional cond{BranchCondition(insn)};
    if (!guard || !cond) {
        return std::nullopt;
    }
    if (IsTrue(*guard)) {
        return cond;
    }
    if (IsTrue(*cond)) {
        return guard;
    }
    const auto [guard_pred, guard_negated]{guard->GetPred()};
    const auto [cond_pred, cond_negated]{cond->GetPred()};
    return IR::Condition{guard_pred, guard_negated, cond_pred, cond_negated};
}

// Relative targets are encoded in 32-bit words from the next instruction
u32 RelativeTarget(u32 offset, const Instruction& insn) {
    const u64 raw{insn.Bits(34, 48)};
    const s64 words{static_cast<s64>(raw << 16) >> 16};
    return static_cast<u32>(static_cast<s64>(offset) + 16 + words * 4);
}

constexpr u32 RZ{255};

// Jump table of an indirect branch: entries are 32-bit words in a constant buffer
struct BranchTable {
    u32 cbuf_index{};
    u32 cbuf_offset{};
    u32 num_entries{};
};

// Finds the jump table of BRX Ra by walking back to the start of its block. The compiler emits
//   IMNMX.U32 Ri, Rx, max ; SHF.L.U32 Rj, Ri, 2 ; LDC Ra, c[bank][Rj + offset] ; BRX Ra
// where the clamp bounds the table to max + 1 entries.
std::optional<BranchTable> TrackBranchTable(Environment& env, u32 brx_offset, u32 begin_offset) {
    const auto read{[&](u32 offset) {
        return Instruction{env.ReadInstruction(offset), env.ReadInstruction(offset + 8)};
    }};
    // Last unconditional writer of reg before offset, or nullopt when it isn't in the block
    const auto writer{[&](u32 reg, u32 offset) -> std::optional<std::pair<u32, Instruction>> {
        while (offset > begin_offset) {
            offset -= 16;
            const Instruction insn{read(offset)};
            if (insn.Dest() == reg && Ampere::Decode(insn) != Opcode::Invalid) {
                if (insn.Pred() != PT || insn.PredNegated()) {
                    return std::nullopt;
                }
                return std::pair{offset, insn};
            }
        }
        return std::nullopt;
    }};
    const Instruction brx{read(brx_offset)};
    const auto load{writer(brx.SrcA(), brx_offset)};
    if (!load || Ampere::Decode(load->second) != Opcode::LDC || load->second.Bits(78, 2) != 0 ||
        load->second.Bit(91) || load->second.SrcA() == RZ) {
        return std::nullopt;
    }
    const Instruction ldc{load->second};
    const auto scale{writer(ldc.SrcA(), load->first)};
    if (!scale || Ampere::Decode(scale->second) != Opcode::SHF ||
        scale->second.Form() != Ampere::AluForm::ImmReg || scale->second.Bit(76) ||
        scale->second.Bits(32, 32) != 2 || scale->second.Bits(64, 8) != RZ) {
        return std::nullopt;
    }
    const auto clamp{writer(scale->second.SrcA(), scale->first)};
    if (!clamp || Ampere::Decode(clamp->second) != Opcode::IMNMX ||
        clamp->second.Form() != Ampere::AluForm::ImmReg || clamp->second.Bit(73) ||
        clamp->second.Bits(87, 3) != PT || clamp->second.Bit(90)) {
        return std::nullopt;
    }
    const u32 max_index{static_cast<u32>(clamp->second.Bits(32, 32))};
    if (max_index >= 256) {
        return std::nullopt;
    }
    return BranchTable{
        .cbuf_index = ldc.CbufIndex(),
        .cbuf_offset = static_cast<u32>(static_cast<s32>(static_cast<s16>(ldc.CbufOffset()))),
        .num_entries = max_index + 1,
    };
}

} // Anonymous namespace

CFG::AnalysisState CFG::AnalyzeInstAmpere(Block* block, FunctionId function_id, Location pc) {
    const u32 offset{Ampere::ToOffset(pc)};
    const Instruction insn{env.ReadInstruction(offset), env.ReadInstruction(offset + 8)};
    const Opcode opcode{Ampere::Decode(insn)};
    switch (opcode) {
    case Opcode::NOP:
    case Opcode::BSSY:
    case Opcode::BSYNC:
    case Opcode::BREAK:
    case Opcode::WARPSYNC:
    case Opcode::YIELD:
        // Convergence barriers have no effect when each invocation runs independently
        return AnalysisState::Continue;
    case Opcode::BRA:
    case Opcode::CALL_REL: {
        // CALL.REL.NOINC (bit 82) without a link setup is used as a forward branch (e.g. loop
        // exits and LUT shaders). When preceded by a link-register setup, treat it as a call.
        if (opcode == Opcode::CALL_REL && offset >= 16) {
            const Instruction prev{env.ReadInstruction(offset - 16),
                                   env.ReadInstruction(offset - 8)};
            if (Ampere::Decode(prev) == Opcode::MOV && prev.Form() == Ampere::AluForm::ImmReg &&
                prev.Imm32() == offset + 16 && insn.Pred() == PT && !insn.PredNegated()) {
                const Location cal_pc{Ampere::ToLocation(RelativeTarget(offset, insn))};
                const auto it{std::ranges::find(functions, cal_pc, &Function::entrypoint)};
                const bool exists{it != functions.end()};
                const FunctionId call_id{
                    exists ? static_cast<size_t>(std::distance(functions.begin(), it))
                           : functions.size()};
                if (!exists) {
                    functions.emplace_back(block_pool, cal_pc);
                }
                block->end_class = EndClass::Call;
                block->function_call = call_id;
                block->return_block = AddLabel(block, block->stack, pc + 1, function_id);
                block->end = pc;
                return AnalysisState::Branch;
            }
        }
        const std::optional cond{CombinedCondition(insn)};
        if (!cond) {
            return AnalysisState::Continue;
        }
        block->cond = *cond;
        if (!IsTrue(*cond)) {
            block->branch_false = AddLabel(block, block->stack, pc + 1, function_id);
        }
        block->branch_true = AddLabel(block, block->stack,
                                      Ampere::ToLocation(RelativeTarget(offset, insn)), function_id);
        block->end = pc;
        return AnalysisState::Branch;
    }
    case Opcode::RET: {
        const std::optional cond{CombinedCondition(insn)};
        if (!cond) {
            return AnalysisState::Continue;
        }
        if (!IsTrue(*cond)) {
            AnalyzeCondInst(block, function_id, pc, EndClass::Return, *cond);
            return AnalysisState::Branch;
        }
        block->end = pc;
        block->end_class = EndClass::Return;
        return AnalysisState::Branch;
    }
    case Opcode::EXIT: {
        const std::optional cond{CombinedCondition(insn)};
        if (!cond) {
            return AnalysisState::Continue;
        }
        if (exits_to_dispatcher) {
            throw NotImplementedException("Dispatch EXIT");
        }
        if (!IsTrue(*cond)) {
            AnalyzeCondInst(block, function_id, pc, EndClass::Exit, *cond);
            return AnalysisState::Branch;
        }
        block->end = pc + 1;
        block->end_class = EndClass::Exit;
        return AnalysisState::Branch;
    }
    case Opcode::KILL: {
        const std::optional cond{GuardCondition(insn)};
        if (!cond) {
            return AnalysisState::Continue;
        }
        if (!IsTrue(*cond)) {
            AnalyzeCondInst(block, function_id, pc, EndClass::Kill, *cond);
            return AnalysisState::Branch;
        }
        block->end = pc + 1;
        block->end_class = EndClass::Kill;
        return AnalysisState::Branch;
    }
    case Opcode::BRX: {
        // Target: the next instruction plus Ra plus the signed immediate, as BRA
        if (insn.Pred() != PT || insn.PredNegated()) {
            throw NotImplementedException("Conditional indirect branch");
        }
        const std::optional table{TrackBranchTable(env, offset, Ampere::ToOffset(block->begin))};
        if (!table) {
            throw NotImplementedException("Untracked indirect branch at {:#x}", offset);
        }
        const u32 base{RelativeTarget(offset, insn)};
        std::vector<u32> targets;
        for (u32 i = 0; i < table->num_entries; ++i) {
            targets.push_back(base + env.ReadCbufValue(table->cbuf_index,
                                                       table->cbuf_offset + i * 4));
        }
        std::ranges::sort(targets);
        targets.erase(std::unique(targets.begin(), targets.end()), targets.end());
        block->indirect_branches.reserve(targets.size());
        for (const u32 target : targets) {
            block->indirect_branches.push_back({
                .block = AddLabel(block, block->stack, Ampere::ToLocation(target), function_id),
                .address = target,
            });
        }
        block->cond = IR::Condition{true};
        block->end = pc + 1;
        block->end_class = EndClass::IndirectBranch;
        block->branch_reg = static_cast<IR::Reg>(insn.SrcA());
        block->branch_offset = static_cast<s32>(base);
        return AnalysisState::Branch;
    }
    case Opcode::BRXU:
        throw NotImplementedException("Indirect branch {}", opcode);
    case Opcode::CALL_ABS:
        throw NotImplementedException("Function call {}", opcode);
    case Opcode::Invalid:
        throw NotImplementedException("Invalid instruction {:016x} {:016x} at {:#x}", insn.hi,
                                      insn.lo, offset);
    default:
        break;
    }
    const std::optional guard{GuardCondition(insn)};
    if (!guard || IsTrue(*guard)) {
        // Instructions that never execute are dropped by the translator
        return AnalysisState::Continue;
    }
    AnalyzeCondInst(block, function_id, pc, EndClass::Branch, *guard);
    return AnalysisState::Branch;
}

} // namespace Shader::Maxwell::Flow
