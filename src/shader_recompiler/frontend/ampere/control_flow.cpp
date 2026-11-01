// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <optional>
#include <string_view>
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

Instruction ReadInsn(Environment& env, u32 offset) {
    return Instruction{env.ReadInstruction(offset), env.ReadInstruction(offset + 8)};
}

// Whether [16..24) of the instruction is a general purpose destination register. Barriers,
// stores, control flow, predicate writers and uniform datapath instructions use those bits for
// something else (DEPBAR.LE SB5, 0x1 reads as a write to R0).
bool WritesGpr(Opcode opcode) {
    switch (opcode) {
    case Opcode::Invalid:
    case Opcode::BAR:
    case Opcode::BRA:
    case Opcode::BRX:
    case Opcode::BRXU:
    case Opcode::BREAK:
    case Opcode::BSSY:
    case Opcode::BSYNC:
    case Opcode::CALL_ABS:
    case Opcode::CALL_REL:
    case Opcode::CCTL:
    case Opcode::CCTLL:
    case Opcode::CCTLT:
    case Opcode::DEPBAR:
    case Opcode::ERRBAR:
    case Opcode::EXIT:
    case Opcode::KILL:
    case Opcode::MEMBAR:
    case Opcode::NOP:
    case Opcode::RED:
    case Opcode::RET:
    case Opcode::ST:
    case Opcode::STG:
    case Opcode::STL:
    case Opcode::STS:
    case Opcode::WARPSYNC:
    case Opcode::YIELD:
        return false;
    default:
        break;
    }
    const std::string_view name{Ampere::NameOf(opcode)};
    return !name.starts_with('U') && name.find("SETP") == std::string_view::npos &&
           name.find("PLOP") == std::string_view::npos;
}

// Last unconditional writer of reg before offset, or nullopt when it isn't in the block
std::optional<std::pair<u32, Instruction>> LastWriter(Environment& env, u32 reg, u32 offset,
                                                      u32 begin_offset) {
    while (offset > begin_offset) {
        offset -= 16;
        const Instruction insn{ReadInsn(env, offset)};
        if (insn.Dest() == reg && WritesGpr(Ampere::Decode(insn))) {
            if (insn.Pred() != PT || insn.PredNegated()) {
                return std::nullopt;
            }
            return std::pair{offset, insn};
        }
    }
    return std::nullopt;
}

bool IsUnconditionalJump(Opcode opcode, Instruction insn) {
    switch (opcode) {
    case Opcode::BRA:
    case Opcode::BRX:
    case Opcode::BRXU:
    case Opcode::EXIT:
    case Opcode::RET:
    case Opcode::CALL_REL:
    case Opcode::CALL_ABS:
        return insn.Pred() == PT && !insn.PredNegated();
    default:
        return false;
    }
}

// Like LastWriter, but walks across the instructions guarded by a predicate (they split blocks)
// up to the function entry or the previous unconditional jump. A guarded writer of reg is not
// tracked.
std::optional<std::pair<u32, Instruction>> LastWriterInFunction(Environment& env, u32 reg,
                                                                u32 offset, u32 begin_offset) {
    while (offset > begin_offset) {
        offset -= 16;
        const Instruction insn{ReadInsn(env, offset)};
        const Opcode opcode{Ampere::Decode(insn)};
        if (IsUnconditionalJump(opcode, insn)) {
            return std::nullopt;
        }
        if (insn.Dest() == reg && WritesGpr(opcode)) {
            if (insn.Pred() != PT || insn.PredNegated()) {
                return std::nullopt;
            }
            return std::pair{offset, insn};
        }
    }
    return std::nullopt;
}

// A subroutine call loads a link register before CALL.REL.NOINC, and the callee returns with
// RET.REL.NODEC Rlink, which jumps to Rlink plus its own relative target. Graphics programs count
// addresses from the start of the program header, so the link value is not the return offset.
// The callee is the code from the target up to its first RET.
bool IsSubroutineCall(Environment& env, u32 call_offset, u32 begin_offset, u32 target) {
    constexpr u32 max_callee_size{0x10000};
    for (u32 offset = target; offset < target + max_callee_size; offset += 16) {
        const Instruction insn{ReadInsn(env, offset)};
        const Opcode opcode{Ampere::Decode(insn)};
        if (opcode == Opcode::Invalid ||
            (opcode == Opcode::BRA && RelativeTarget(offset, insn) == offset)) {
            return false;
        }
        if (opcode != Opcode::RET) {
            continue;
        }
        // Library functions may copy the link to another register before returning
        // (MOV R0, R15; MOV R1, 0x0; RET.REL.NODEC R0)
        u32 link_reg{insn.SrcA()};
        for (u32 copy_offset = offset, depth = 0; depth < 4; ++depth) {
            const auto copy{LastWriter(env, link_reg, copy_offset, target)};
            if (!copy || Ampere::Decode(copy->second) != Opcode::MOV ||
                copy->second.Form() != Ampere::AluForm::RegReg) {
                break;
            }
            link_reg = copy->second.SrcBReg();
            copy_offset = copy->first;
        }
        const auto link{LastWriterInFunction(env, link_reg, call_offset, begin_offset)};
        return link && Ampere::Decode(link->second) == Opcode::MOV &&
               link->second.Form() == Ampere::AluForm::ImmReg &&
               link->second.Imm32() + RelativeTarget(offset, insn) == call_offset + 16;
    }
    return false;
}

// Finds the jump table of BRX Ra by walking back to the start of its block. The compiler emits
//   IMNMX.U32 Ri, Rx, max ; SHF.L.U32 Rj, Ri, 2 ; LDC Ra, c[bank][Rj + offset] ; BRX Ra
// where the clamp bounds the table to max + 1 entries.
std::optional<BranchTable> TrackBranchTable(Environment& env, u32 brx_offset, u32 begin_offset) {
    const auto writer{[&](u32 reg, u32 offset) {
        return LastWriter(env, reg, offset, begin_offset);
    }};
    const Instruction brx{ReadInsn(env, brx_offset)};
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

// Last unconditional writer of uniform register ur before offset. Uniform datapath instructions
// (U* and R2UR) keep their destination at [16..22); predicate writers (UISETP, UPLOP3, ...) are
// skipped. The walk may cross block boundaries: predicated vector instructions split blocks but
// cannot write uniform registers, so it goes back to the function entry or the previous
// unconditional jump.
std::optional<std::pair<u32, Instruction>> LastUniformWriter(Environment& env, u32 ur, u32 offset,
                                                             u32 begin_offset) {
    while (offset > begin_offset) {
        offset -= 16;
        const Instruction insn{ReadInsn(env, offset)};
        const Opcode opcode{Ampere::Decode(insn)};
        if (opcode == Opcode::Invalid) {
            continue;
        }
        if (IsUnconditionalJump(opcode, insn)) {
            return std::nullopt;
        }
        const std::string_view name{Ampere::NameOf(opcode)};
        const bool uniform{name.starts_with('U') || name.starts_with("R2UR")};
        if (!uniform || name.find("SETP") != std::string_view::npos ||
            name.find("PLOP") != std::string_view::npos || insn.Bits(16, 6) != ur) {
            continue;
        }
        if (insn.Pred() != PT || insn.PredNegated()) {
            return std::nullopt;
        }
        return std::pair{offset, insn};
    }
    return std::nullopt;
}

// BRXU URa takes its target from a uniform register. The compiler emits
//   UISETP.LT.U32 UP, URx, max ; USEL URi, URx, max, UP ; USHF.L.U32 URj, URi, 2 ;
//   ULDC URa, c[bank][URj + offset] ; BRXU URa
// so the table has max + 1 entries.
std::optional<BranchTable> TrackUniformBranchTable(Environment& env, u32 brx_offset,
                                                   u32 begin_offset) {
    const auto writer{[&](u32 ur, u32 offset) {
        return LastUniformWriter(env, ur, offset, begin_offset);
    }};
    const Instruction brx{ReadInsn(env, brx_offset)};
    const auto load{writer(static_cast<u32>(brx.Bits(24, 6)), brx_offset)};
    if (!load || Ampere::Decode(load->second) != Opcode::ULDC_INDEXED ||
        load->second.Bits(73, 3) != 4) {
        return std::nullopt;
    }
    const Instruction uldc{load->second};
    const auto scale{writer(static_cast<u32>(uldc.Bits(24, 6)), load->first)};
    if (!scale || Ampere::Decode(scale->second) != Opcode::USHF ||
        scale->second.Form() != Ampere::AluForm::ImmReg || scale->second.Bit(76) ||
        scale->second.Bits(32, 32) != 2 || scale->second.Bits(64, 6) != 63) {
        return std::nullopt;
    }
    const auto clamp{writer(static_cast<u32>(scale->second.Bits(24, 6)), scale->first)};
    if (!clamp || Ampere::Decode(clamp->second) != Opcode::USEL ||
        clamp->second.Form() != Ampere::AluForm::ImmReg) {
        return std::nullopt;
    }
    // The selected value is URx while the predicate (URx < max) holds and max otherwise
    const u32 max_index{static_cast<u32>(clamp->second.Bits(32, 32))};
    if (max_index >= 256) {
        return std::nullopt;
    }
    return BranchTable{
        .cbuf_index = static_cast<u32>(uldc.Bits(54, 5)),
        .cbuf_offset = static_cast<u32>(static_cast<s32>(static_cast<s16>(uldc.Bits(38, 16)))),
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
        // exits and LUT shaders). When its callee returns through the link, treat it as a call.
        if (opcode == Opcode::CALL_REL && insn.Pred() == PT && !insn.PredNegated()) {
            const u32 target{RelativeTarget(offset, insn)};
            if (IsSubroutineCall(env, offset,
                                 Ampere::ToOffset(functions[function_id].entrypoint), target)) {
                const Location cal_pc{Ampere::ToLocation(target)};
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
        if (function_id == 0) {
            throw NotImplementedException("Return without a tracked call at {:#x}", offset);
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
    case Opcode::BRX:
    case Opcode::BRXU: {
        // Target: the next instruction plus Ra (URa for BRXU) plus the signed immediate, as BRA
        if (insn.Pred() != PT || insn.PredNegated()) {
            throw NotImplementedException("Conditional indirect branch");
        }
        const bool uniform{opcode == Opcode::BRXU};
        const std::optional table{
            uniform ? TrackUniformBranchTable(env, offset,
                                              Ampere::ToOffset(functions[function_id].entrypoint))
                    : TrackBranchTable(env, offset, Ampere::ToOffset(block->begin))};
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
        block->branch_reg = uniform ? IR::UniformReg(static_cast<u32>(insn.Bits(24, 6)))
                                    : static_cast<IR::Reg>(insn.SrcA());
        block->branch_offset = static_cast<s32>(base);
        return AnalysisState::Branch;
    }
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
