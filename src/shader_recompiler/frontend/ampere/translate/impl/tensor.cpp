// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>

#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ampere/translate/impl/impl.h"

// Tensor core instructions, emulated with warp shuffles. Matrix fragments are spread over the 32
// lanes of a warp as PTX documents for ldmatrix, movmatrix and mma.m16n8k16: lane t holds row
// t / 4 ("group") and the column pair 2 * (t % 4) of each 8x8 block of 16-bit elements. Every
// lane of the warp must execute these instructions together, as on hardware.
namespace Shader::Ampere {
namespace {

constexpr u32 RZ{255};

struct Lanes {
    IR::U32 lane;
    IR::U32 group;
    IR::U32 thread_in_group;
};

Lanes MakeLanes(IR::IREmitter& ir) {
    const IR::U32 lane{ir.LaneId()};
    return {lane, ir.ShiftRightLogical(lane, ir.Imm32(2)), ir.BitwiseAnd(lane, ir.Imm32(3))};
}

IR::U32 ShuffleFrom(IR::IREmitter& ir, const IR::U32& value, const IR::U32& lane) {
    return ir.ShuffleIndex(value, lane, ir.Imm32(31), ir.Imm32(0));
}

// Lane holding row `row` and column pair `column_pair` of an 8x8 block
IR::U32 FragmentLane(IR::IREmitter& ir, const IR::U32& row, const IR::U32& column_pair) {
    return ir.IAdd(ir.ShiftLeftLogical(row, ir.Imm32(2)), column_pair);
}

std::array<IR::F32, 2> Unpack(IR::IREmitter& ir, const IR::U32& value, bool negate) {
    const IR::Value halves{ir.UnpackHalf2x16(value)};
    IR::F32 low{ir.CompositeExtract(halves, 0)};
    IR::F32 high{ir.CompositeExtract(halves, 1)};
    if (negate) {
        low = IR::F32{ir.FPNeg(low)};
        high = IR::F32{ir.FPNeg(high)};
    }
    return {low, high};
}

} // Anonymous namespace

// LDSM.16.{M88,MT88}.{1,2,4} Rd, [Ra + imm]: lane i * 8 + r gives the shared memory address of row
// r of matrix i, and register i of each lane receives its fragment of matrix i. MT88 transposes.
// Count at [72..74), shape at [78..80), signed 24-bit offset at [40..64).
void TranslatorVisitor::LDSM(Instruction insn) {
    const u64 shape{insn.Bits(78, 2)};
    if (insn.Bits(72, 2) > 2 || shape > 1 || insn.Bit(91)) {
        throw NotImplementedException("LDSM count {} shape {} or uniform offset", insn.Bits(72, 2),
                                      shape);
    }
    const u32 count{1U << insn.Bits(72, 2)};
    const bool transpose{shape == 1};
    const s32 offset{static_cast<s32>(static_cast<u32>(insn.Bits(40, 24)) << 8) >> 8};
    const IR::U32 address{ir.IAdd(X(Reg(insn.SrcA())), ir.Imm32(offset))};
    const Lanes lanes{MakeLanes(ir)};
    std::array<IR::U32, 4> results;
    for (u32 matrix = 0; matrix < count; ++matrix) {
        const IR::U32 first_row{ir.Imm32(matrix * 8)};
        if (!transpose) {
            const IR::U32 row_address{
                ShuffleFrom(ir, address, ir.IAdd(first_row, lanes.group))};
            const IR::U32 element{ir.IAdd(
                row_address, ir.ShiftLeftLogical(lanes.thread_in_group, ir.Imm32(2)))};
            results[matrix] = IR::U32{ir.LoadShared(32, false, element)};
            continue;
        }
        // Rows 2 * (t % 4) and 2 * (t % 4) + 1 at column t / 4
        const IR::U32 row{ir.ShiftLeftLogical(lanes.thread_in_group, ir.Imm32(1))};
        const IR::U32 column{ir.ShiftLeftLogical(lanes.group, ir.Imm32(1))};
        const IR::U32 low_row{ShuffleFrom(ir, address, ir.IAdd(first_row, row))};
        const IR::U32 high_row{
            ShuffleFrom(ir, address, ir.IAdd(first_row, ir.IAdd(row, ir.Imm32(1))))};
        const IR::U32 low{IR::U32{ir.LoadShared(16, false, ir.IAdd(low_row, column))}};
        const IR::U32 high{IR::U32{ir.LoadShared(16, false, ir.IAdd(high_row, column))}};
        results[matrix] = ir.BitFieldInsert(low, high, ir.Imm32(16), ir.Imm32(16));
    }
    const IR::Reg dest{Reg(insn.Dest())};
    for (u32 matrix = 0; matrix < count; ++matrix) {
        X(dest + static_cast<int>(matrix), results[matrix]);
    }
}

// MOVM.16.MT88 Rd, Ra: transposes the 8x8 matrix of 16-bit elements whose fragments are in Ra.
// Element (r, c) is in lane r * 4 + c / 2, half c % 2; lane t receives elements (2 * (t % 4), t / 4)
// and (2 * (t % 4) + 1, t / 4). The other shapes at [78..80) are not implemented.
void TranslatorVisitor::MOVM(Instruction insn) {
    if (insn.Bits(78, 2) != 0) {
        throw NotImplementedException("MOVM shape {}", insn.Bits(78, 2));
    }
    const IR::U32 source{X(Reg(insn.SrcA()))};
    const Lanes lanes{MakeLanes(ir)};
    const IR::U32 row{ir.ShiftLeftLogical(lanes.thread_in_group, ir.Imm32(1))};
    const IR::U32 column_pair{ir.ShiftRightLogical(lanes.group, ir.Imm32(1))};
    const IR::U32 shift{ir.ShiftLeftLogical(ir.BitwiseAnd(lanes.group, ir.Imm32(1)), ir.Imm32(4))};
    const IR::U32 low_word{ShuffleFrom(ir, source, FragmentLane(ir, row, column_pair))};
    const IR::U32 high_word{ShuffleFrom(
        ir, source, FragmentLane(ir, ir.IAdd(row, ir.Imm32(1)), column_pair))};
    const IR::U32 low{ir.BitFieldExtract(low_word, shift, ir.Imm32(16))};
    const IR::U32 high{ir.BitFieldExtract(high_word, shift, ir.Imm32(16))};
    X(Reg(insn.Dest()), ir.BitFieldInsert(low, high, ir.Imm32(16), ir.Imm32(16)));
}

// HMMA.16816.F16 Rd, Ra, Rb, Rc: D (16x8) = A (16x16) * B (16x8) + C with F16 elements, as
// mma.m16n8k16.row.col.f16.f16.f16.f16. Lane t (group g, thread t % 4 = q) holds
//   A: Ra (g, 2q..2q+1), Ra+1 (g+8, 2q..), Ra+2 (g, 2q+8..), Ra+3 (g+8, 2q+8..)
//   B: Rb rows 2q..2q+1 and Rb+1 rows 2q+8..2q+9 of column g
//   C and D: Rc (g, 2q..2q+1) and Rc+1 (g+8, 2q..2q+1)
// Products are summed in F32 and rounded to F16 once. Negation of A and B is at bits 72 and 63;
// the sparse (73), m16n8k8 (75 clear), F32 accumulator (76) and BF16/TF32 (82..84) forms and a
// uniform predicate at [87..91) (stored XOR 7, so 0 is UPT) are not
// implemented.
void TranslatorVisitor::HMMA(Instruction insn) {
    if (insn.Bit(73) || !insn.Bit(75) || insn.Bit(76) || insn.Bits(82, 2) != 0 ||
        insn.Bits(87, 3) != 0 || insn.Bit(90)) {
        throw NotImplementedException("HMMA form {:#x}", insn.Bits(72, 20));
    }
    const IR::Reg a{Reg(insn.SrcA())};
    const IR::Reg b{Reg(insn.SrcBReg())};
    const u32 c_index{static_cast<u32>(insn.Bits(64, 8))};
    const bool negate_a{insn.Bit(72)};
    const bool negate_b{insn.Bit(63)};
    if (!hmma_lanes) {
        const Lanes made{MakeLanes(ir)};
        hmma_lanes = {made.lane, made.group, made.thread_in_group};
    }
    const Lanes lanes{(*hmma_lanes)[0], (*hmma_lanes)[1], (*hmma_lanes)[2]};
    const auto writes{[&](IR::Reg reg) { return register_writes[static_cast<u32>(reg)]; }};

    // Rows g and g + 8 of A, k = 0..15, from lanes 4g..4g+3: rows[r][k] is element 16r + k
    const std::array<u32, 7> a_key{0, static_cast<u32>(a), writes(a), writes(a + 1),
                                   writes(a + 2), writes(a + 3), negate_a};
    auto a_it{hmma_fragments.find(a_key)};
    if (a_it == hmma_fragments.end()) {
        std::array<IR::F32, 32> rows;
        const std::array<IR::U32, 4> a_regs{X(a), X(a + 1), X(a + 2), X(a + 3)};
        for (u32 q = 0; q < 4; ++q) {
            const IR::U32 source{FragmentLane(ir, lanes.group, ir.Imm32(q))};
            for (u32 half_row = 0; half_row < 2; ++half_row) {
                const auto low_k{Unpack(ir, ShuffleFrom(ir, a_regs[half_row], source), negate_a)};
                const auto high_k{
                    Unpack(ir, ShuffleFrom(ir, a_regs[half_row + 2], source), negate_a)};
                rows[16 * half_row + 2 * q] = low_k[0];
                rows[16 * half_row + 2 * q + 1] = low_k[1];
                rows[16 * half_row + 2 * q + 8] = high_k[0];
                rows[16 * half_row + 2 * q + 9] = high_k[1];
            }
        }
        a_it = hmma_fragments.emplace(a_key, rows).first;
    }
    // Columns 2q and 2q + 1 of B, k = 0..15, from lanes 4c..4c+3: element 16c + k
    const std::array<u32, 7> b_key{1, static_cast<u32>(b), writes(b), writes(b + 1), 0, 0,
                                   negate_b};
    auto b_it{hmma_fragments.find(b_key)};
    if (b_it == hmma_fragments.end()) {
        std::array<IR::F32, 32> columns;
        const std::array<IR::U32, 2> b_regs{X(b), X(b + 1)};
        for (u32 column = 0; column < 2; ++column) {
            const IR::U32 n{ir.IAdd(ir.ShiftLeftLogical(lanes.thread_in_group, ir.Imm32(1)),
                                    ir.Imm32(column))};
            for (u32 q = 0; q < 4; ++q) {
                const IR::U32 source{FragmentLane(ir, n, ir.Imm32(q))};
                const auto low_k{Unpack(ir, ShuffleFrom(ir, b_regs[0], source), negate_b)};
                const auto high_k{Unpack(ir, ShuffleFrom(ir, b_regs[1], source), negate_b)};
                columns[16 * column + 2 * q] = low_k[0];
                columns[16 * column + 2 * q + 1] = low_k[1];
                columns[16 * column + 2 * q + 8] = high_k[0];
                columns[16 * column + 2 * q + 9] = high_k[1];
            }
        }
        b_it = hmma_fragments.emplace(b_key, columns).first;
    }
    const std::array<IR::F32, 32>& rows{a_it->second};
    const std::array<IR::F32, 32>& columns{b_it->second};
    std::array<IR::U32, 2> results;
    for (u32 half_row = 0; half_row < 2; ++half_row) {
        std::array<IR::F32, 2> sums;
        if (c_index == RZ) {
            sums = {ir.Imm32(0.0f), ir.Imm32(0.0f)};
        } else {
            sums = Unpack(ir, X(Reg(c_index) + static_cast<int>(half_row)), false);
        }
        for (u32 column = 0; column < 2; ++column) {
            for (u32 k = 0; k < 16; ++k) {
                sums[column] = IR::F32{
                    ir.FPFma(rows[16 * half_row + k], columns[16 * column + k], sums[column])};
            }
        }
        results[half_row] = ir.PackHalf2x16(ir.CompositeConstruct(sums[0], sums[1]));
    }
    const IR::Reg dest{Reg(insn.Dest())};
    X(dest, results[0]);
    X(dest + 1, results[1]);
}

} // namespace Shader::Ampere
