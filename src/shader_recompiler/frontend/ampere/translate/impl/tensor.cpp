// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>

#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ampere/translate/impl/impl.h"

// Tensor core instructions. LDSM and MOVM are emulated with warp shuffles; matrix fragments are
// spread over the 32 lanes of a warp as PTX documents for ldmatrix, movmatrix and mma.m16n8k16:
// lane t holds row t / 4 ("group") and the column pair 2 * (t % 4) of each 8x8 block of 16-bit elements. Every
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

} // Anonymous namespace

// LDSM.16.{M88,MT88}.{1,2,4} Rd, [Ra + imm]: lane i * 8 + r gives the shared memory address of row
// r of matrix i, and register i of each lane receives its fragment of matrix i. MT88 transposes.
// Count at [72..74) (1, 2 or 4 matrices), shape at [78..80), signed 24-bit offset at [40..64);
// with bit 91 a uniform register at [32..38) is added ([Ra+URb+imm]).
void TranslatorVisitor::LDSM(Instruction insn) {
    const u64 shape{insn.Bits(78, 2)};
    if (insn.Bits(72, 2) > 2 || shape > 1) {
        throw NotImplementedException("LDSM count {} shape {}", insn.Bits(72, 2), shape);
    }
    const u32 count{1U << insn.Bits(72, 2)};
    const bool transpose{shape == 1};
    const s32 offset{static_cast<s32>(static_cast<u32>(insn.Bits(40, 24)) << 8) >> 8};
    IR::U32 address{ir.IAdd(X(Reg(insn.SrcA())), ir.Imm32(offset))};
    if (insn.Bit(91) && !IR::IsZero(UReg(insn.Bits(32, 6)))) {
        address = ir.IAdd(X(UReg(insn.Bits(32, 6))), address);
    }
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
// mma.m16n8k16.row.col.f16.f16.f16.f16; HMMA.1688.F16 (bit 75 clear) is mma.m16n8k8 and reads
// only Ra, Ra+1 and Rb. LowerHmmaPass documents the fragments and lowers the product.
// Negation of A and B is at bits 72 and 63; the sparse (73), F32 accumulator (76) and BF16/TF32
// (82..84) forms and a uniform predicate at [87..91) (stored XOR 7, so 0 is UPT) are not
// implemented.
void TranslatorVisitor::HMMA(Instruction insn) {
    if (insn.Bit(73) || insn.Bit(76) || insn.Bits(82, 2) != 0 ||
        insn.Bits(87, 3) != 0 || insn.Bit(90)) {
        throw NotImplementedException("HMMA form {:#x}", insn.Bits(72, 20));
    }
    const IR::Reg a{Reg(insn.SrcA())};
    const IR::Reg b{Reg(insn.SrcBReg())};
    const u32 c_index{static_cast<u32>(insn.Bits(64, 8))};
    const bool k16{insn.Bit(75)};
    IR::HmmaInfo info{};
    info.k16.Assign(k16 ? 1 : 0);
    info.negate_a.Assign(insn.Bit(72) ? 1 : 0);
    info.negate_b.Assign(insn.Bit(63) ? 1 : 0);
    const IR::U32 a0{X(a)};
    const IR::U32 a1{X(a + 1)};
    const IR::Value a_words{k16 ? ir.CompositeConstruct(a0, a1, X(a + 2), X(a + 3))
                                : ir.CompositeConstruct(a0, a1, a0, a1)};
    const IR::U32 b0{X(b)};
    const IR::Value b_words{ir.CompositeConstruct(b0, k16 ? X(b + 1) : b0)};
    const IR::Value c_words{c_index == RZ
                                ? ir.CompositeConstruct(ir.Imm32(0), ir.Imm32(0))
                                : ir.CompositeConstruct(X(Reg(c_index)), X(Reg(c_index) + 1))};
    const IR::Value d_words{ir.HmmaF16(a_words, b_words, c_words, info)};
    const IR::Reg dest{Reg(insn.Dest())};
    X(dest, IR::U32{ir.CompositeExtract(d_words, 0)});
    X(dest + 1, IR::U32{ir.CompositeExtract(d_words, 1)});
}

} // namespace Shader::Ampere
