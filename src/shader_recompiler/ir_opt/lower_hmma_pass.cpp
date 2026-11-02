// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <deque>
#include <optional>

#include "common/div_ceil.h"
#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/ir_opt/passes.h"

// Lowers HmmaF16, one warp-wide HMMA.{1688,16816}.F16, as mma.m16n8k{8,16}.row.col.f16 in PTX.
// Lane t (group g = t / 4, q = t % 4) holds
//   A: a0 (g, 2q..2q+1), a1 (g+8, 2q..), and with K=16 a2 (g, 2q+8..), a3 (g+8, 2q+8..)
//   B: b0 rows 2q..2q+1 of column g, and with K=16 b1 rows 2q+8..2q+9
//   C and D: c0 (g, 2q..2q+1) and c1 (g+8, 2q..2q+1)
// Each word holds two F16 elements, the lower column (or row of B) in its low half.
//
// Compute shaders on hosts with subgroup cooperative matrices of these shapes stage the fragments
// in shared memory, if there is room, and run the product on the host's tensor cores,
// accumulating in F16 like the guest. The invocations' elements of a cooperative matrix are in an
// implementation-defined order (on NVIDIA not the order of these fragments), so the registers
// cannot go into a matrix directly. Otherwise the fragments are gathered with warp shuffles and
// summed in F32.
namespace Shader::Optimization {
namespace {

constexpr u32 WARP_SCRATCH_BYTES{1024};
constexpr u32 A_OFFSET{0};   // 16x16 F16 row-major, 512 bytes
constexpr u32 B_OFFSET{512}; // 16x8 F16 column-major, 256 bytes
constexpr u32 C_OFFSET{768}; // 16x8 F16 row-major, 256 bytes; D replaces it
constexpr u32 SIGN_BITS{0x80008000};

bool UsesCooperativeMatrix(const HostTranslateInfo& host_info, IR::HmmaInfo info) {
    return info.k16 != 0 ? host_info.support_cooperative_matrix_f16_k16
                         : host_info.support_cooperative_matrix_f16_k8;
}

struct Lanes {
    IR::U32 lane;
    IR::U32 group;
    IR::U32 thread_in_group;
};

Lanes MakeLanes(IR::IREmitter& ir) {
    const IR::U32 lane{ir.LaneId()};
    return {lane, ir.ShiftRightLogical(lane, ir.Imm32(2)), ir.BitwiseAnd(lane, ir.Imm32(3))};
}

IR::U32 Word(IR::IREmitter& ir, const IR::Value& composite, size_t index) {
    return IR::U32{ir.CompositeExtract(composite, index)};
}

class CooperativeLowering {
public:
    explicit CooperativeLowering(const IR::Program& program_, u32 scratch_base_)
        : program{program_}, scratch_base{scratch_base_} {}

    IR::Value Lower(IR::Block& block, IR::Inst& inst, IR::IREmitter& ir) {
        const auto info{inst.Flags<IR::HmmaInfo>()};
        const bool k16{info.k16 != 0};
        if (cached_block != &block) {
            cached_block = &block;
            lanes = MakeLanes(ir);
            warp_base = WarpBase(ir);
        }
        const IR::U32 g{lanes->group};
        const IR::U32 q{lanes->thread_in_group};
        // Byte offset of word `row * row_words + column_word + q` of a matrix
        const auto at{[&](u32 matrix, const IR::U32& row, u32 row_words, u32 column_word) {
            const IR::U32 word{ir.IAdd(ir.IMul(row, ir.Imm32(row_words)),
                                       ir.IAdd(q, ir.Imm32(column_word)))};
            return ir.IAdd(*warp_base, ir.IAdd(ir.ShiftLeftLogical(word, ir.Imm32(2)),
                                               ir.Imm32(matrix)));
        }};
        const auto negate{[&](IR::U32 value, bool enabled) {
            return enabled ? ir.BitwiseXor(value, ir.Imm32(SIGN_BITS)) : value;
        }};
        const IR::U32 g8{ir.IAdd(g, ir.Imm32(8))};
        const u32 k_words{k16 ? 8U : 4U};
        const IR::Value a{inst.Arg(0)};
        const IR::Value b{inst.Arg(1)};
        const IR::Value c{inst.Arg(2)};

        // The lanes may still be reading the previous product
        ir.SubgroupBarrier();
        ir.WriteShared(32, at(A_OFFSET, g, k_words, 0), negate(Word(ir, a, 0), info.negate_a != 0));
        ir.WriteShared(32, at(A_OFFSET, g8, k_words, 0), negate(Word(ir, a, 1), info.negate_a != 0));
        if (k16) {
            ir.WriteShared(32, at(A_OFFSET, g, k_words, 4), negate(Word(ir, a, 2), info.negate_a != 0));
            ir.WriteShared(32, at(A_OFFSET, g8, k_words, 4),
                           negate(Word(ir, a, 3), info.negate_a != 0));
        }
        ir.WriteShared(32, at(B_OFFSET, g, k_words, 0), negate(Word(ir, b, 0), info.negate_b != 0));
        if (k16) {
            ir.WriteShared(32, at(B_OFFSET, g, k_words, 4), negate(Word(ir, b, 1), info.negate_b != 0));
        }
        const IR::U32 d0_offset{at(C_OFFSET, g, 4, 0)};
        const IR::U32 d1_offset{at(C_OFFSET, g8, 4, 0)};
        ir.WriteShared(32, d0_offset, Word(ir, c, 0));
        ir.WriteShared(32, d1_offset, Word(ir, c, 1));
        ir.SubgroupBarrier();
        ir.CooperativeMatrixMulAddF16(ir.IAdd(*warp_base, ir.Imm32(A_OFFSET)),
                                      ir.IAdd(*warp_base, ir.Imm32(B_OFFSET)),
                                      ir.IAdd(*warp_base, ir.Imm32(C_OFFSET)), info);
        ir.SubgroupBarrier();
        return ir.CompositeConstruct(ir.LoadShared(32, false, d0_offset),
                                     ir.LoadShared(32, false, d1_offset));
    }

private:
    /// Shared memory of this warp: warps are consecutive groups of 32 linear invocation indices
    IR::U32 WarpBase(IR::IREmitter& ir) const {
        const auto& size{program.workgroup_size};
        const IR::U32 index{ir.IAdd(
            ir.LocalInvocationIdX(),
            ir.IMul(ir.IAdd(ir.LocalInvocationIdY(),
                            ir.IMul(ir.LocalInvocationIdZ(), ir.Imm32(size[1]))),
                    ir.Imm32(size[0])))};
        const IR::U32 warp{ir.ShiftRightLogical(index, ir.Imm32(5))};
        return ir.IAdd(ir.Imm32(scratch_base), ir.IMul(warp, ir.Imm32(WARP_SCRATCH_BYTES)));
    }

    const IR::Program& program;
    u32 scratch_base;
    const IR::Block* cached_block{};
    std::optional<Lanes> lanes;
    std::optional<IR::U32> warp_base;
};

class ShuffleLowering {
public:
    IR::Value Lower(IR::Block& block, IR::Inst& inst, IR::IREmitter& ir) {
        const auto info{inst.Flags<IR::HmmaInfo>()};
        const u32 k_count{info.k16 != 0 ? 16U : 8U};
        if (cached_block != &block) {
            cached_block = &block;
            lanes = MakeLanes(ir);
            fragments.clear();
        }
        const auto& rows{Fragment(ir, inst.Arg(0), k_count, info.negate_a != 0, true)};
        const auto& columns{Fragment(ir, inst.Arg(1), k_count, info.negate_b != 0, false)};
        const IR::Value c{inst.Arg(2)};
        std::array<IR::U32, 2> results;
        for (u32 half_row = 0; half_row < 2; ++half_row) {
            std::array<IR::F32, 2> sums{Unpack(ir, Word(ir, c, half_row), false)};
            for (u32 column = 0; column < 2; ++column) {
                for (u32 k = 0; k < k_count; ++k) {
                    sums[column] = IR::F32{
                        ir.FPFma(rows[16 * half_row + k], columns[16 * column + k], sums[column])};
                }
            }
            results[half_row] = ir.PackHalf2x16(ir.CompositeConstruct(sums[0], sums[1]));
        }
        return ir.CompositeConstruct(results[0], results[1]);
    }

private:
    using Elements = std::array<IR::F32, 32>;

    static std::array<IR::F32, 2> Unpack(IR::IREmitter& ir, const IR::U32& value, bool negate) {
        const IR::Value halves{ir.UnpackHalf2x16(value)};
        IR::F32 low{ir.CompositeExtract(halves, 0)};
        IR::F32 high{ir.CompositeExtract(halves, 1)};
        if (negate) {
            low = IR::F32{ir.FPNeg(low)};
            high = IR::F32{ir.FPNeg(high)};
        }
        return {low, high};
    }

    IR::U32 ShuffleFrom(IR::IREmitter& ir, const IR::U32& value, const IR::U32& lane) {
        return ir.ShuffleIndex(value, lane, ir.Imm32(31), ir.Imm32(0));
    }

    // Lane holding row `row` and column pair `column_pair` of an 8x8 block
    IR::U32 FragmentLane(IR::IREmitter& ir, const IR::U32& row, const IR::U32& column_pair) {
        return ir.IAdd(ir.ShiftLeftLogical(row, ir.Imm32(2)), column_pair);
    }

    /// The words of a composite, from its construction when possible so equal fragments match
    static std::array<IR::Value, 4> Words(IR::IREmitter& ir, const IR::Value& composite,
                                          size_t count) {
        std::array<IR::Value, 4> words{};
        const IR::Inst* const construct{composite.IsImmediate() ? nullptr
                                                                : composite.InstRecursive()};
        const bool constructed{construct != nullptr &&
                               (construct->GetOpcode() == IR::Opcode::CompositeConstructU32x2 ||
                                construct->GetOpcode() == IR::Opcode::CompositeConstructU32x4)};
        for (size_t index = 0; index < count; ++index) {
            words[index] = constructed ? construct->Arg(index) : Word(ir, composite, index);
        }
        return words;
    }

    /// For A, rows g and g + 8 (element 16r + k); for B, columns 2q and 2q + 1 (element 16c + k).
    /// Products with the same A or B words in a block share the shuffles.
    const Elements& Fragment(IR::IREmitter& ir, const IR::Value& composite, u32 k_count,
                             bool negate, bool is_a) {
        const std::array<IR::Value, 4> words{Words(ir, composite, is_a ? 4 : 2)};
        for (const Cached& cached : fragments) {
            if (cached.words == words && cached.k_count == k_count && cached.negate == negate &&
                cached.is_a == is_a) {
                return cached.elements;
            }
        }
        Elements elements;
        if (is_a) {
            const std::array<IR::U32, 4> regs{IR::U32{words[0]}, IR::U32{words[1]},
                                              IR::U32{words[2]}, IR::U32{words[3]}};
            for (u32 q = 0; q < 4; ++q) {
                const IR::U32 source{FragmentLane(ir, lanes->group, ir.Imm32(q))};
                for (u32 half_row = 0; half_row < 2; ++half_row) {
                    const auto low_k{Unpack(ir, ShuffleFrom(ir, regs[half_row], source), negate)};
                    elements[16 * half_row + 2 * q] = low_k[0];
                    elements[16 * half_row + 2 * q + 1] = low_k[1];
                    if (k_count == 16) {
                        const auto high_k{
                            Unpack(ir, ShuffleFrom(ir, regs[half_row + 2], source), negate)};
                        elements[16 * half_row + 2 * q + 8] = high_k[0];
                        elements[16 * half_row + 2 * q + 9] = high_k[1];
                    }
                }
            }
        } else {
            const std::array<IR::U32, 2> regs{IR::U32{words[0]}, IR::U32{words[1]}};
            for (u32 column = 0; column < 2; ++column) {
                const IR::U32 n{ir.IAdd(ir.ShiftLeftLogical(lanes->thread_in_group, ir.Imm32(1)),
                                        ir.Imm32(column))};
                for (u32 q = 0; q < 4; ++q) {
                    const IR::U32 source{FragmentLane(ir, n, ir.Imm32(q))};
                    const auto low_k{Unpack(ir, ShuffleFrom(ir, regs[0], source), negate)};
                    elements[16 * column + 2 * q] = low_k[0];
                    elements[16 * column + 2 * q + 1] = low_k[1];
                    if (k_count == 16) {
                        const auto high_k{Unpack(ir, ShuffleFrom(ir, regs[1], source), negate)};
                        elements[16 * column + 2 * q + 8] = high_k[0];
                        elements[16 * column + 2 * q + 9] = high_k[1];
                    }
                }
            }
        }
        return fragments.emplace_back(words, k_count, negate, is_a, elements).elements;
    }

    const IR::Block* cached_block{};
    std::optional<Lanes> lanes;
    struct Cached {
        std::array<IR::Value, 4> words;
        u32 k_count;
        bool negate;
        bool is_a;
        Elements elements;
    };
    std::deque<Cached> fragments;
};

} // Anonymous namespace

void LowerHmmaPass(IR::Program& program, const HostTranslateInfo& host_info) {
    // Shared memory for the staging areas, if the host has room for them
    bool cooperative{program.stage == Stage::Compute};
    bool any_cooperative{};
    for (IR::Block* const block : program.blocks) {
        for (IR::Inst& inst : block->Instructions()) {
            if (inst.GetOpcode() == IR::Opcode::HmmaF16) {
                any_cooperative |= UsesCooperativeMatrix(host_info, inst.Flags<IR::HmmaInfo>());
            }
        }
    }
    cooperative &= any_cooperative;
    const auto& size{program.workgroup_size};
    const u32 scratch_base{Common::DivCeil(program.shared_memory_size, 16U) * 16U};
    const u32 warps{Common::DivCeil(size[0] * size[1] * size[2], 32U)};
    if (cooperative && scratch_base + warps * WARP_SCRATCH_BYTES >
                           host_info.max_compute_shared_memory_size) {
        cooperative = false;
    }
    if (cooperative) {
        program.shared_memory_size = scratch_base + warps * WARP_SCRATCH_BYTES;
    }
    CooperativeLowering cooperative_lowering{program, scratch_base};
    ShuffleLowering shuffle_lowering;
    for (IR::Block* const block : program.blocks) {
        for (IR::Inst& inst : block->Instructions()) {
            if (inst.GetOpcode() != IR::Opcode::HmmaF16) {
                continue;
            }
            IR::IREmitter ir{*block, IR::Block::InstructionList::s_iterator_to(inst)};
            const bool use_cooperative{
                cooperative && UsesCooperativeMatrix(host_info, inst.Flags<IR::HmmaInfo>())};
            const IR::Value result{use_cooperative
                                       ? cooperative_lowering.Lower(*block, inst, ir)
                                       : shuffle_lowering.Lower(*block, inst, ir)};
            inst.ReplaceUsesWith(result);
        }
    }
}

} // namespace Shader::Optimization
