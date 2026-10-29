// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <optional>

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
    throw NotImplementedException("Branch with both guard and condition predicates");
}

// Relative targets are encoded in 32-bit words from the next instruction
u32 RelativeTarget(u32 offset, const Instruction& insn) {
    const u64 raw{insn.Bits(34, 48)};
    const s64 words{static_cast<s64>(raw << 16) >> 16};
    return static_cast<u32>(static_cast<s64>(offset) + 16 + words * 4);
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
    case Opcode::BRA: {
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
    case Opcode::BRX:
    case Opcode::BRXU:
        throw NotImplementedException("Indirect branch {}", opcode);
    case Opcode::CALL_ABS:
    case Opcode::CALL_REL:
    case Opcode::RET:
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
