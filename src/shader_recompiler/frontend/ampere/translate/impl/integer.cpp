// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <vector>

#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ampere/translate/impl/impl.h"
#include "shader_recompiler/frontend/maxwell/translate/impl/common_funcs.h"

namespace Shader::Ampere {
namespace {

constexpr u64 PT{7};

// Full 32x32 multiply split in 16-bit halves, so no 64-bit host integers are required
std::pair<IR::U32, IR::U32> MulWide(IR::IREmitter& ir, const IR::U32& a, const IR::U32& b,
                                    bool is_signed) {
    const IR::U32 mask{ir.Imm32(0xffff)};
    const IR::U32 sixteen{ir.Imm32(16)};
    const IR::U32 a_lo{ir.BitwiseAnd(a, mask)};
    const IR::U32 a_hi{ir.ShiftRightLogical(a, sixteen)};
    const IR::U32 b_lo{ir.BitwiseAnd(b, mask)};
    const IR::U32 b_hi{ir.ShiftRightLogical(b, sixteen)};
    const IR::U32 p0{ir.IMul(a_lo, b_lo)};
    const IR::U32 p1{ir.IMul(a_lo, b_hi)};
    const IR::U32 p2{ir.IMul(a_hi, b_lo)};
    const IR::U32 p3{ir.IMul(a_hi, b_hi)};
    const IR::U32 mid{ir.IAdd(ir.IAdd(ir.ShiftRightLogical(p0, sixteen), ir.BitwiseAnd(p1, mask)),
                              ir.BitwiseAnd(p2, mask))};
    IR::U32 hi{ir.IAdd(ir.IAdd(p3, ir.ShiftRightLogical(p1, sixteen)),
                       ir.IAdd(ir.ShiftRightLogical(p2, sixteen), ir.ShiftRightLogical(mid, sixteen)))};
    if (is_signed) {
        const IR::U32 zero{ir.Imm32(0)};
        hi = ir.ISub(hi, IR::U32{ir.Select(ir.ILessThan(a, zero, true), b, zero)});
        hi = ir.ISub(hi, IR::U32{ir.Select(ir.ILessThan(b, zero, true), a, zero)});
    }
    return {ir.IMul(a, b), hi};
}

IR::U32 CarryOf(IR::IREmitter& ir, const IR::U32& sum, const IR::U32& addend) {
    return IR::U32{ir.Select(ir.ILessThan(sum, addend, false), ir.Imm32(1), ir.Imm32(0))};
}

IR::U32 PredicateBit(IR::IREmitter& ir, const IR::U1& pred) {
    return IR::U32{ir.Select(pred, ir.Imm32(1), ir.Imm32(0))};
}

// Three-input add with optional operand negation, carry inputs and carry outputs.
// Carry outputs receive bits 32 and 33 of the full-width sum.
void Add3(TranslatorVisitor& v, Instruction insn, bool uniform) {
    const bool extended{insn.Bit(74)};
    std::array<IR::U32, 3> operands{v.SrcA(insn, uniform), v.SrcB(insn, uniform),
                                    v.SrcC(insn, uniform)};
    const std::array<bool, 3> negate{v.NegA(insn), v.NegB(insn), v.NegC(insn)};
    const u64 carry_out_0{insn.Bits(81, 3)};
    const u64 carry_out_1{insn.Bits(84, 3)};
    const bool wants_carry{carry_out_0 != PT || carry_out_1 != PT};

    std::vector<IR::U32> addends;
    for (size_t i = 0; i < operands.size(); ++i) {
        if (!negate[i]) {
            addends.push_back(operands[i]);
            continue;
        }
        // Two's complement, or one's complement when the borrow comes from the carry inputs
        addends.push_back(v.ir.BitwiseNot(operands[i]));
        if (!extended) {
            addends.push_back(v.ir.Imm32(1));
        }
    }
    if (extended) {
        const auto carry_in{[&](u32 index_bit, u32 not_bit) {
            const u64 index{insn.Bits(index_bit, 3)};
            const bool negated{insn.Bit(not_bit)};
            return uniform ? v.UP(index, negated) : v.P(index, negated);
        }};
        addends.push_back(PredicateBit(v.ir, carry_in(87, 90)));
        addends.push_back(PredicateBit(v.ir, carry_in(77, 80)));
    }
    IR::U32 sum{addends[0]};
    IR::U32 carries{v.ir.Imm32(0)};
    for (size_t i = 1; i < addends.size(); ++i) {
        const IR::U32 next{v.ir.IAdd(sum, addends[i])};
        if (wants_carry) {
            carries = v.ir.IAdd(carries, CarryOf(v.ir, next, addends[i]));
        }
        sum = next;
    }
    const auto set_pred{[&](u64 index, const IR::U1& value) {
        uniform ? v.SetUP(index, value) : v.SetP(index, value);
    }};
    if (carry_out_0 != PT) {
        set_pred(carry_out_0, v.ir.INotEqual(v.ir.BitwiseAnd(carries, v.ir.Imm32(1)), v.ir.Imm32(0)));
    }
    if (carry_out_1 != PT) {
        set_pred(carry_out_1, v.ir.INotEqual(v.ir.BitwiseAnd(carries, v.ir.Imm32(2)), v.ir.Imm32(0)));
    }
    v.X(uniform ? UReg(insn.Bits(16, 6)) : Reg(insn.Dest()), sum);
}

// The predicate output at [81..84) is set when the result is nonzero, combined with the predicate
// at [87..90) (negated by bit 90) by OR, or by AND with .PAND (bit 80)
void Lop3(TranslatorVisitor& v, Instruction insn, bool uniform) {
    const IR::U32 result{Maxwell::ApplyLUT(v.ir, v.SrcA(insn, uniform), v.SrcB(insn, uniform),
                                           v.SrcC(insn, uniform), insn.Bits(72, 8))};
    const u64 pred_dest{insn.Bits(81, 3)};
    if (pred_dest != PT) {
        const IR::U1 nonzero{v.ir.INotEqual(result, v.ir.Imm32(0))};
        const u64 pred_src{insn.Bits(87, 3)};
        const bool pred_negated{insn.Bit(90)};
        const IR::U1 input{uniform ? v.UP(pred_src, pred_negated) : v.P(pred_src, pred_negated)};
        const IR::U1 value{insn.Bit(80) ? v.ir.LogicalAnd(nonzero, input)
                                        : v.ir.LogicalOr(nonzero, input)};
        uniform ? v.SetUP(pred_dest, value) : v.SetP(pred_dest, value);
    }
    v.X(uniform ? UReg(insn.Bits(16, 6)) : Reg(insn.Dest()), result);
}

enum class ShiftType : u64 {
    S64,
    U64,
    S32,
    U32,
};

// Funnel shift of {c, a} by b; .HI selects the upper word of the shifted pair
void Shf(TranslatorVisitor& v, Instruction insn, bool uniform) {
    const auto type{static_cast<ShiftType>(insn.Bits(73, 2))};
    const bool wrap{insn.Bit(75)};
    const bool right{insn.Bit(76)};
    const bool high{insn.Bit(80)};
    const IR::U32 low_bits{v.SrcA(insn, uniform)};
    const IR::U32 shift{v.SrcB(insn, uniform)};
    const IR::U32 high_bits{v.SrcC(insn, uniform)};
    const bool is_64{type == ShiftType::S64 || type == ShiftType::U64};
    const IR::U32 max_shift{v.ir.Imm32(is_64 ? 64 : 32)};
    const IR::U32 safe_shift{wrap ? v.ir.BitwiseAnd(shift, v.ir.Imm32(is_64 ? 63 : 31))
                                  : v.ir.UMin(shift, max_shift)};
    // Host shifts by the full 64-bit width are undefined, those are resolved below
    const bool can_overflow{is_64 && !wrap};
    const IR::U32 amount{can_overflow ? v.ir.UMin(safe_shift, v.ir.Imm32(63)) : safe_shift};
    const IR::U64 packed{v.ir.PackUint2x32(v.ir.CompositeConstruct(low_bits, high_bits))};
    IR::U64 shifted;
    if (!right) {
        shifted = v.ir.ShiftLeftLogical(packed, amount);
    } else if (type == ShiftType::S64 || type == ShiftType::S32) {
        shifted = v.ir.ShiftRightArithmetic(packed, amount);
    } else {
        shifted = v.ir.ShiftRightLogical(packed, amount);
    }
    const IR::Value words{v.ir.UnpackUint2x32(shifted)};
    IR::U32 result{v.ir.CompositeExtract(words, high ? 1 : 0)};
    if (can_overflow) {
        // A shift by the full width leaves only zeros or sign bits
        const IR::U1 overflow{v.ir.IEqual(safe_shift, v.ir.Imm32(64))};
        const IR::U32 fill{right && type == ShiftType::S64
                               ? IR::U32{v.ir.ShiftRightArithmetic(high_bits, v.ir.Imm32(31))}
                               : v.ir.Imm32(0)};
        result = IR::U32{v.ir.Select(overflow, fill, result)};
    }
    v.X(uniform ? UReg(insn.Bits(16, 6)) : Reg(insn.Dest()), result);
}

void Isetp(TranslatorVisitor& v, Instruction insn, bool uniform) {
    if (insn.Bit(72)) {
        throw NotImplementedException("ISETP.EX");
    }
    const auto compare_op{static_cast<Maxwell::CompareOp>(insn.Bits(76, 3))};
    const auto bop{static_cast<Maxwell::BooleanOp>(insn.Bits(74, 2))};
    const IR::U1 comparison{Maxwell::IntegerCompare(v.ir, v.SrcA(insn, uniform),
                                                    v.SrcB(insn, uniform), compare_op,
                                                    insn.Bit(73))};
    const IR::U1 accumulator{uniform ? v.UP(insn.Bits(87, 3), insn.Bit(90))
                                     : v.P(insn.Bits(87, 3), insn.Bit(90))};
    const IR::U1 first{Maxwell::PredicateCombine(v.ir, comparison, accumulator, bop)};
    const IR::U1 second{
        Maxwell::PredicateCombine(v.ir, v.ir.LogicalNot(comparison), accumulator, bop)};
    if (uniform) {
        v.SetUP(insn.Bits(81, 3), first);
        v.SetUP(insn.Bits(84, 3), second);
    } else {
        v.SetP(insn.Bits(81, 3), first);
        v.SetP(insn.Bits(84, 3), second);
    }
}

void Plop3(TranslatorVisitor& v, Instruction insn, bool uniform) {
    const auto apply_lut{[&](const IR::U1& a, const IR::U1& b, const IR::U1& c, u64 lut) {
        const auto mask{[&](const IR::U1& p) {
            return IR::U32{v.ir.Select(p, v.ir.Imm32(~0u), v.ir.Imm32(0u))};
        }};
        return v.ir.INotEqual(Maxwell::ApplyLUT(v.ir, mask(a), mask(b), mask(c), lut),
                              v.ir.Imm32(0u));
    }};
    const IR::U1 pa{uniform ? v.UP(insn.Bits(87, 3), insn.Bit(90))
                            : v.P(insn.Bits(87, 3), insn.Bit(90))};
    const IR::U1 pb{uniform ? v.UP(insn.Bits(77, 3), insn.Bit(80))
                            : v.P(insn.Bits(77, 3), insn.Bit(80))};
    const IR::U1 pc{(uniform || insn.Bit(67)) ? v.UP(insn.Bits(68, 3), insn.Bit(71))
                                              : v.P(insn.Bits(68, 3), insn.Bit(71))};
    const u64 lut_0{insn.Bits(64, 3) | (insn.Bits(72, 5) << 3)};
    const u64 lut_1{insn.Bits(16, 8)};
    const IR::U1 out_0{apply_lut(pa, pb, pc, lut_0)};
    const IR::U1 out_1{apply_lut(pa, pb, pc, lut_1)};
    if (uniform) {
        v.SetUP(insn.Bits(81, 3), out_0);
        v.SetUP(insn.Bits(84, 3), out_1);
    } else {
        v.SetP(insn.Bits(81, 3), out_0);
        v.SetP(insn.Bits(84, 3), out_1);
    }
}

} // Anonymous namespace

void TranslatorVisitor::IABS(Instruction insn) {
    X(Reg(insn.Dest()), ir.IAbs(SrcB(insn)));
}

void TranslatorVisitor::IADD3(Instruction insn) {
    Add3(*this, insn, false);
}

void TranslatorVisitor::UIADD3(Instruction insn) {
    Add3(*this, insn, true);
}

void TranslatorVisitor::IMAD(Instruction insn) {
    if (insn.Bits(81, 3) != PT) {
        throw NotImplementedException("IMAD with carry output");
    }
    // .X (bit 74) adds the carry predicate at bits 87-89, negated by bit 90, as in IADD3.
    // The compiler pairs it with an IADD3 carry output for the high word of 64-bit sums.
    const bool extended{insn.Bit(74)};
    if (extended && NegC(insn)) {
        throw NotImplementedException("IMAD.X with a negated addend");
    }
    const IR::U32 product{ir.IMul(SrcA(insn), SrcB(insn))};
    const IR::U32 addend{NegC(insn) ? IR::U32{ir.INeg(SrcC(insn))} : SrcC(insn)};
    IR::U32 result{ir.IAdd(product, addend)};
    if (extended) {
        result = ir.IAdd(result, PredicateBit(ir, P(insn.Bits(87, 3), insn.Bit(90))));
    }
    X(Reg(insn.Dest()), result);
}

void TranslatorVisitor::IMAD_WIDE(Instruction insn) {
    if (insn.Bit(74) || insn.Bits(81, 3) != PT) {
        throw NotImplementedException("IMAD.WIDE with carry");
    }
    if (NegC(insn)) {
        throw NotImplementedException("IMAD.WIDE with a negated addend");
    }
    // The 64-bit addend is a register pair or a pair of constant buffer words
    IR::U32 c_lo;
    IR::U32 c_hi;
    if (BIsWide(insn)) {
        const IR::Reg addend_reg{Reg(insn.Bits(64, 8))};
        c_lo = X(addend_reg);
        c_hi = IR::IsZero(addend_reg) ? ir.Imm32(0) : X(addend_reg + 1);
    } else if (insn.Form() == AluForm::RegCbuf && !insn.CbufBindless()) {
        const IR::U32 index{ir.Imm32(insn.CbufIndex())};
        c_lo = ir.GetCbuf(index, ir.Imm32(insn.CbufOffset()));
        c_hi = ir.GetCbuf(index, ir.Imm32(insn.CbufOffset() + 4));
    } else {
        throw NotImplementedException("IMAD.WIDE addend form {}", static_cast<u32>(insn.Form()));
    }
    const auto [lo, hi]{MulWide(ir, SrcA(insn), SrcB(insn), insn.Bit(73))};
    const IR::U32 sum_lo{ir.IAdd(lo, c_lo)};
    const IR::U32 sum_hi{ir.IAdd(ir.IAdd(hi, c_hi), CarryOf(ir, sum_lo, c_lo))};
    const IR::Reg dest{Reg(insn.Dest())};
    X(dest, sum_lo);
    X(dest + 1, sum_hi);
}

void TranslatorVisitor::IMNMX(Instruction insn) {
    const bool is_signed{insn.Bit(73)};
    const IR::U32 a{SrcA(insn)};
    const IR::U32 b{SrcB(insn)};
    const IR::U1 pick_min{P(insn.Bits(87, 3), insn.Bit(90))};
    X(Reg(insn.Dest()),
      IR::U32{ir.Select(pick_min, ir.IMin(a, b, is_signed), ir.IMax(a, b, is_signed))});
}

void TranslatorVisitor::ISETP(Instruction insn) {
    Isetp(*this, insn, false);
}

void TranslatorVisitor::UISETP(Instruction insn) {
    Isetp(*this, insn, true);
}

void TranslatorVisitor::LOP3(Instruction insn) {
    Lop3(*this, insn, false);
}

void TranslatorVisitor::ULOP3(Instruction insn) {
    Lop3(*this, insn, true);
}

void TranslatorVisitor::PLOP3(Instruction insn) {
    Plop3(*this, insn, false);
}

void TranslatorVisitor::UPLOP3(Instruction insn) {
    Plop3(*this, insn, true);
}

// R2P PR, Ra.Bn, mask: predicates P0-P6 selected by the mask take the bits of byte n of Ra
void TranslatorVisitor::R2P(Instruction insn) {
    const IR::U32 value{SrcA(insn)};
    const IR::U32 mask{SrcB(insn)};
    const u32 byte{static_cast<u32>(insn.Bits(76, 2))};
    for (u32 index = 0; index < 7; ++index) {
        const auto bit{[&](const IR::U32& word, u32 position) {
            return ir.INotEqual(ir.BitwiseAnd(word, ir.Imm32(1U << position)), ir.Imm32(0));
        }};
        SetP(index, IR::U1{ir.Select(bit(mask, index), bit(value, byte * 8 + index), P(index))});
    }
}

void TranslatorVisitor::SHF(Instruction insn) {
    Shf(*this, insn, false);
}

void TranslatorVisitor::USHF(Instruction insn) {
    Shf(*this, insn, true);
}

void TranslatorVisitor::SEL(Instruction insn) {
    const IR::U1 cond{P(insn.Bits(87, 3), insn.Bit(90))};
    X(Reg(insn.Dest()), IR::U32{ir.Select(cond, SrcA(insn), SrcB(insn))});
}

void TranslatorVisitor::USEL(Instruction insn) {
    const IR::U1 cond{UP(insn.Bits(87, 3), insn.Bit(90))};
    X(UReg(insn.Bits(16, 6)), IR::U32{ir.Select(cond, SrcA(insn, true), SrcB(insn, true))});
}

namespace {
// Single-operand bit instructions read b from [32..64) with a bitwise-not modifier at 63
IR::U32 BitSource(TranslatorVisitor& v, Instruction insn) {
    const IR::U32 src{v.SrcB(insn)};
    const bool invert{insn.Form() != AluForm::ImmReg && insn.Bit(63)};
    return invert ? v.ir.BitwiseNot(src) : src;
}
} // Anonymous namespace

void TranslatorVisitor::POPC(Instruction insn) {
    X(Reg(insn.Dest()), ir.BitCount(BitSource(*this, insn)));
}

void TranslatorVisitor::FLO(Instruction insn) {
    if (insn.Bits(81, 3) != PT) {
        throw NotImplementedException("FLO predicate output");
    }
    const IR::U32 src{BitSource(*this, insn)};
    IR::U32 result{insn.Bit(73) ? ir.FindSMsb(src) : ir.FindUMsb(src)};
    if (insn.Bit(74)) {
        // .SH returns the shift amount that brings the bit to the top
        const IR::U32 none{ir.Imm32(0xffffffff)};
        result = IR::U32{ir.Select(ir.IEqual(result, none), none, ir.ISub(ir.Imm32(31), result))};
    }
    X(Reg(insn.Dest()), result);
}

void TranslatorVisitor::BREV(Instruction insn) {
    X(Reg(insn.Dest()), ir.BitReverse(SrcB(insn)));
}

namespace {
// Byte permute in index mode: each selector nibble picks a byte of {c, a}, bit 3 replicates its
// sign bit
IR::U32 Permute(TranslatorVisitor& v, Instruction insn, bool uniform) {
    if (insn.Bits(72, 3) != 0) {
        throw NotImplementedException("PRMT mode {}", insn.Bits(72, 3));
    }
    const IR::U32 a{v.SrcA(insn, uniform)};
    const IR::U32 c{v.SrcC(insn, uniform)};
    const IR::U32 selector{v.SrcB(insn, uniform)};
    const auto replicate_sign{[&](const IR::U32& byte) {
        return v.ir.BitFieldExtract(byte, v.ir.Imm32(7), v.ir.Imm32(1), true);
    }};
    IR::U32 result{v.ir.Imm32(0)};
    if (selector.IsImmediate()) {
        for (u32 i = 0; i < 4; ++i) {
            const u32 nibble{(selector.U32() >> (i * 4)) & 0xf};
            const u32 index{nibble & 7};
            IR::U32 byte{v.ir.BitFieldExtract(index < 4 ? a : c, v.ir.Imm32((index & 3) * 8),
                                              v.ir.Imm32(8))};
            if ((nibble & 8) != 0) {
                byte = replicate_sign(byte);
            }
            result = v.ir.BitFieldInsert(result, byte, v.ir.Imm32(i * 8), v.ir.Imm32(8));
        }
        return result;
    }
    const IR::U64 source{v.ir.PackUint2x32(v.ir.CompositeConstruct(a, c))};
    for (u32 i = 0; i < 4; ++i) {
        const IR::U32 nibble{v.ir.BitFieldExtract(selector, v.ir.Imm32(i * 4), v.ir.Imm32(4))};
        const IR::U32 shift{
            v.ir.ShiftLeftLogical(v.ir.BitwiseAnd(nibble, v.ir.Imm32(7)), v.ir.Imm32(3))};
        const IR::U64 shifted{v.ir.ShiftRightLogical(source, shift)};
        const IR::U32 low{v.ir.CompositeExtract(v.ir.UnpackUint2x32(shifted), 0)};
        const IR::U32 byte{v.ir.BitwiseAnd(low, v.ir.Imm32(0xff))};
        const IR::U1 sign{v.ir.INotEqual(v.ir.BitwiseAnd(nibble, v.ir.Imm32(8)), v.ir.Imm32(0))};
        const IR::U32 value{v.ir.Select(sign, replicate_sign(byte), byte)};
        result = v.ir.BitFieldInsert(result, value, v.ir.Imm32(i * 8), v.ir.Imm32(8));
    }
    return result;
}
} // Anonymous namespace

void TranslatorVisitor::PRMT(Instruction insn) {
    X(Reg(insn.Dest()), Permute(*this, insn, false));
}

void TranslatorVisitor::UPRMT(Instruction insn) {
    X(UReg(insn.Bits(16, 6)), Permute(*this, insn, true));
}

// a shifted left by [75..80) plus b; .HI shifts the pair {c, a} and keeps the upper word
void TranslatorVisitor::LEA(Instruction insn) {
    if (NegA(insn)) {
        throw NotImplementedException("LEA with negated operand");
    }
    const u32 shift{static_cast<u32>(insn.Bits(75, 5))};
    const bool high{insn.Bit(80)};
    const IR::U32 a{SrcA(insn)};
    IR::U32 shifted;
    if (!high) {
        shifted = ir.ShiftLeftLogical(a, ir.Imm32(shift));
    } else if (shift == 0) {
        shifted = SrcC(insn);
    } else {
        shifted = ir.BitwiseOr(ir.ShiftLeftLogical(SrcC(insn), ir.Imm32(shift)),
                               ir.ShiftRightLogical(a, ir.Imm32(32 - shift)));
    }
    const IR::U32 b{SrcB(insn)};
    IR::U32 sum{ir.IAdd(shifted, b)};
    IR::U32 carries{CarryOf(ir, sum, b)};
    if (insn.Bit(74)) {
        const IR::U32 carry_in{PredicateBit(ir, P(insn.Bits(87, 3), insn.Bit(90)))};
        const IR::U32 next{ir.IAdd(sum, carry_in)};
        carries = ir.IAdd(carries, CarryOf(ir, next, carry_in));
        sum = next;
    }
    if (insn.Bits(81, 3) != PT) {
        SetP(insn.Bits(81, 3), ir.INotEqual(carries, ir.Imm32(0)));
    }
    X(Reg(insn.Dest()), sum);
}

} // namespace Shader::Ampere
