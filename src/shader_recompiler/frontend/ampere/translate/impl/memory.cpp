// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <algorithm>
#include <optional>

#include "common/logging.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ampere/translate/impl/impl.h"

namespace Shader::Ampere {
namespace {

constexpr u64 PT{7};

enum class Size : u64 {
    U8,
    S8,
    U16,
    S16,
    B32,
    B64,
    B128,
};

std::pair<int, bool> BitSize(Instruction insn) {
    switch (static_cast<Size>(insn.Bits(73, 3))) {
    case Size::U8:
        return {8, false};
    case Size::S8:
        return {8, true};
    case Size::U16:
        return {16, false};
    case Size::S16:
        return {16, true};
    case Size::B32:
        return {32, false};
    case Size::B64:
        return {64, false};
    case Size::B128:
        return {128, false};
    }
    throw NotImplementedException("Memory access size {}", insn.Bits(73, 3));
}

// Forms with bit 11 of the opcode and bit 91 set take a uniform register operand
bool HasUniformAddress(Instruction insn) {
    return (insn.Opcode() & 0x800) != 0 && insn.Bit(91);
}

} // Anonymous namespace

s32 ImmediateOffset(Instruction insn) {
    return static_cast<s32>(static_cast<u32>(insn.Bits(40, 24)) << 8) >> 8;
}

IR::U64 GlobalAddress(TranslatorVisitor& v, Instruction insn, u32 ureg_offset, u32 reg_64_bit) {
    const bool uniform_form{HasUniformAddress(insn)};
    const bool reg_is_64{uniform_form ? insn.Bit(reg_64_bit) : insn.Bit(72)};
    // With a 64-bit register address, a 64-bit uniform register is a memory descriptor
    // (desc[URx]) and does not take part in the address; checked on GA102 with nvcc's
    // STG.E [R2.64] (desc[UR4]). With Ra = RZ the uniform register is the address ([URx], the
    // only uniform form in the NS2 titles' programs besides [Rx.U32+URy]). nvdisasm prints some
    // 64-bit forms as [Rx.64+URy] (ATOMG; STG with bit 76); none was seen in titles yet.
    const bool is_descriptor{uniform_form && reg_is_64 && insn.Bit(72) &&
                             !IR::IsZero(Reg(insn.SrcA()))};
    const bool has_ureg{uniform_form && !is_descriptor};
    const bool ureg_is_64{has_ureg && insn.Bit(72)};
    // Built as IAdd64(PackUint2x32(CompositeConstruct(lo, hi)), immediate), the form storage
    // buffer tracking in GlobalMemoryToStorageBufferPass expects
    std::optional<std::pair<IR::U32, IR::U32>> parts;
    const auto add_register{[&](IR::Reg reg, bool is_64) {
        if (IR::IsZero(reg)) {
            return;
        }
        if (is_64 && !IR::IsAligned(reg, 2)) {
            throw NotImplementedException("Unaligned source register {}", reg);
        }
        const IR::U32 lo{v.X(reg)};
        const IR::U32 hi{is_64 ? v.X(reg + 1) : v.ir.Imm32(0)};
        if (!parts) {
            parts.emplace(lo, hi);
            return;
        }
        const IR::U32 sum_lo{v.ir.IAdd(parts->first, lo)};
        const IR::U32 carry{
            v.ir.Select(v.ir.ILessThan(sum_lo, lo, false), v.ir.Imm32(1), v.ir.Imm32(0))};
        const IR::U32 sum_hi{v.ir.IAdd(v.ir.IAdd(parts->second, hi), carry)};
        parts = {sum_lo, sum_hi};
    }};
    add_register(Reg(insn.SrcA()), reg_is_64);
    if (has_ureg) {
        add_register(UReg(insn.Bits(ureg_offset, 6)), ureg_is_64);
    }
    const s64 offset{ImmediateOffset(insn)};
    if (!parts) {
        return v.ir.Imm64(static_cast<u64>(offset));
    }
    const IR::U64 base{
        v.ir.PackUint2x32(v.ir.CompositeConstruct(parts->first, parts->second))};
    return offset == 0 ? base : IR::U64{v.ir.IAdd(base, v.ir.Imm64(static_cast<u64>(offset)))};
}

IR::U32 WindowAddress(TranslatorVisitor& v, Instruction insn, u32 ureg_offset, bool has_stride) {
    IR::U32 address{v.ir.Imm32(ImmediateOffset(insn))};
    const IR::Reg reg{Reg(insn.SrcA())};
    if (!IR::IsZero(reg)) {
        IR::U32 value{v.X(reg)};
        const u64 stride{has_stride ? insn.Bits(78, 2) : 0};
        if (stride != 0) {
            value = v.ir.ShiftLeftLogical(value, v.ir.Imm32(static_cast<u32>(stride + 1)));
        }
        address = v.ir.IAdd(value, address);
    }
    if (insn.Bit(91)) {
        const IR::Reg ureg{UReg(insn.Bits(ureg_offset, 6))};
        if (!IR::IsZero(ureg)) {
            address = v.ir.IAdd(v.X(ureg), address);
        }
    }
    return address;
}

namespace {

void WriteVector(TranslatorVisitor& v, IR::Reg dest, const IR::Value& vector, int bit_size) {
    if (!IR::IsAligned(dest, static_cast<size_t>(bit_size / 32))) {
        throw NotImplementedException("Unaligned destination register {}", dest);
    }
    for (int i = 0; i < bit_size / 32; ++i) {
        v.X(dest + i, IR::U32{v.ir.CompositeExtract(vector, static_cast<size_t>(i))});
    }
}

IR::Value ReadVector(TranslatorVisitor& v, IR::Reg src, int bit_size) {
    if (!IR::IsAligned(src, static_cast<size_t>(bit_size / 32))) {
        throw NotImplementedException("Unaligned source register {}", src);
    }
    if (bit_size == 64) {
        return v.ir.CompositeConstruct(v.X(src), v.X(src + 1));
    }
    return v.ir.CompositeConstruct(v.X(src), v.X(src + 1), v.X(src + 2), v.X(src + 3));
}

IR::U32 ByteBit(IR::IREmitter& ir, const IR::U32& offset, u32 mask) {
    return ir.BitwiseAnd(ir.ShiftLeftLogical(offset, ir.Imm32(3)), ir.Imm32(mask));
}

IR::U32 LoadLocalWord(TranslatorVisitor& v, const IR::U32& offset) {
    const IR::U32 word_offset{v.ir.ShiftRightLogical(offset, v.ir.Imm32(2))};
    const IR::U1 in_bounds{v.ir.ILessThan(offset, v.ir.Imm32(v.env.LocalMemorySize()), false)};
    return IR::U32{v.ir.Select(in_bounds, v.ir.LoadLocal(word_offset), v.ir.Imm32(0))};
}

} // Anonymous namespace

void TranslatorVisitor::LDG(Instruction insn) {
    if (insn.Bits(81, 3) != PT) {
        throw NotImplementedException("LDG with a predicate destination");
    }
    // The source predicate at [64..67] is stored XOR 7, so 0 is PT. Compilers guard robust
    // buffer reads with it ("LDG R, [UR], !P0" after "P0 = size <= 0"); when it is false the
    // destination is zero.
    const u64 pred_index{insn.Bits(64, 3) ^ PT};
    const bool pred_negated{insn.Bit(67)};
    const IR::U64 address{GlobalAddress(*this, insn, 32, 90)};
    const IR::Reg dest{Reg(insn.Dest())};
    const auto [bit_size, is_signed]{BitSize(insn)};
    switch (bit_size) {
    case 8:
        X(dest, is_signed ? ir.LoadGlobalS8(address) : ir.LoadGlobalU8(address));
        break;
    case 16:
        X(dest, is_signed ? ir.LoadGlobalS16(address) : ir.LoadGlobalU16(address));
        break;
    case 32:
        X(dest, ir.LoadGlobal32(address));
        break;
    case 64:
        WriteVector(*this, dest, ir.LoadGlobal64(address), 64);
        break;
    case 128:
        WriteVector(*this, dest, ir.LoadGlobal128(address), 128);
        break;
    }
    if (pred_index != PT || pred_negated) {
        const IR::U1 condition{P(pred_index, pred_negated)};
        for (int i = 0; i < std::max(bit_size / 32, 1); ++i) {
            X(dest + i, IR::U32{ir.Select(condition, X(dest + i), ir.Imm32(0))});
        }
    }
}

void TranslatorVisitor::STG(Instruction insn) {
    const IR::U64 address{GlobalAddress(*this, insn, 64, 90)};
    const IR::Reg data{Reg(insn.SrcBReg())};
    const auto [bit_size, is_signed]{BitSize(insn)};
    switch (bit_size) {
    case 8:
        is_signed ? ir.WriteGlobalS8(address, X(data)) : ir.WriteGlobalU8(address, X(data));
        break;
    case 16:
        is_signed ? ir.WriteGlobalS16(address, X(data)) : ir.WriteGlobalU16(address, X(data));
        break;
    case 32:
        ir.WriteGlobal32(address, X(data));
        break;
    case 64:
        ir.WriteGlobal64(address, ReadVector(*this, data, 64));
        break;
    case 128:
        ir.WriteGlobal128(address, ReadVector(*this, data, 128));
        break;
    }
}

void TranslatorVisitor::LDS(Instruction insn) {
    const IR::U32 offset{WindowAddress(*this, insn, 32, true)};
    const IR::Reg dest{Reg(insn.Dest())};
    const auto [bit_size, is_signed]{BitSize(insn)};
    const IR::Value value{ir.LoadShared(bit_size, is_signed, offset)};
    if (bit_size <= 32) {
        X(dest, IR::U32{value});
    } else {
        WriteVector(*this, dest, value, bit_size);
    }
}

void TranslatorVisitor::STS(Instruction insn) {
    const IR::U32 offset{WindowAddress(*this, insn, 64, true)};
    const IR::Reg data{Reg(insn.SrcBReg())};
    const int bit_size{BitSize(insn).first};
    if (bit_size <= 32) {
        ir.WriteShared(bit_size, offset, X(data));
    } else {
        ir.WriteShared(bit_size, offset, ReadVector(*this, data, bit_size));
    }
}

// LDGSTS [Rd + imm20], [Ra.64 + imm12], P: copies global memory to shared memory (cp.async).
// The copy is done at once, so LDGDEPBAR and DEPBAR have nothing to wait for. Shared offset at
// [44..64), global offset at [32..44), size at [73..76) as in LDG, a uniform register at
// [64..70) is added to the global address when bit 76 is clear (URZ otherwise), and the
// predicate at [87..90) (negated by bit 90) selects between the copy and zeros (GA102).
// .ZFILL (bit 82), which takes the number of bytes to zero from the address, is not
// implemented.
void TranslatorVisitor::LDGSTS(Instruction insn) {
    if (insn.Bit(82)) {
        throw NotImplementedException("LDGSTS.ZFILL");
    }
    const int bit_size{BitSize(insn).first};
    const s32 shared_offset{static_cast<s32>(static_cast<u32>(insn.Bits(44, 20)) << 12) >> 12};
    const s64 global_offset{static_cast<s32>(static_cast<u32>(insn.Bits(32, 12)) << 20) >> 20};
    const IR::U32 shared{ir.IAdd(X(Reg(insn.Dest())), ir.Imm32(shared_offset))};
    const IR::Reg base{Reg(insn.SrcA())};
    IR::U32 lo{IR::IsZero(base) ? ir.Imm32(0) : X(base)};
    IR::U32 hi{IR::IsZero(base) ? ir.Imm32(0) : X(base + 1)};
    if (!insn.Bit(76) && !IR::IsZero(UReg(insn.Bits(64, 6)))) {
        const IR::U32 ureg{X(UReg(insn.Bits(64, 6)))};
        const IR::U32 sum{ir.IAdd(lo, ureg)};
        hi = ir.IAdd(hi, IR::U32{ir.Select(ir.ILessThan(sum, ureg, false), ir.Imm32(1),
                                           ir.Imm32(0))});
        lo = sum;
    }
    IR::U64 address{ir.PackUint2x32(ir.CompositeConstruct(lo, hi))};
    if (global_offset != 0) {
        address = ir.IAdd(address, ir.Imm64(static_cast<u64>(global_offset)));
    }
    const IR::U1 condition{P(insn.Bits(87, 3), insn.Bit(90))};
    const bool guarded{insn.Bits(87, 3) != PT || insn.Bit(90)};
    const auto select{[&](const IR::U32& value) {
        return guarded ? IR::U32{ir.Select(condition, value, ir.Imm32(0))} : value;
    }};
    switch (bit_size) {
    case 32:
        ir.WriteShared(32, shared, select(ir.LoadGlobal32(address)));
        break;
    case 64:
    case 128: {
        const IR::Value value{bit_size == 64 ? ir.LoadGlobal64(address)
                                             : ir.LoadGlobal128(address)};
        std::array<IR::U32, 4> words;
        for (int i = 0; i < bit_size / 32; ++i) {
            words[i] = select(IR::U32{ir.CompositeExtract(value, static_cast<size_t>(i))});
        }
        ir.WriteShared(bit_size, shared,
                       bit_size == 64 ? ir.CompositeConstruct(words[0], words[1])
                                      : ir.CompositeConstruct(words[0], words[1], words[2],
                                                              words[3]));
        break;
    }
    default:
        throw NotImplementedException("LDGSTS size {}", bit_size);
    }
}

void TranslatorVisitor::LDL(Instruction insn) {
    const IR::U32 offset{WindowAddress(*this, insn, 32, false)};
    const IR::Reg dest{Reg(insn.Dest())};
    const auto [bit_size, is_signed]{BitSize(insn)};
    const IR::U32 word{LoadLocalWord(*this, offset)};
    switch (bit_size) {
    case 8:
        X(dest, ir.BitFieldExtract(word, ByteBit(ir, offset, 24), ir.Imm32(8), is_signed));
        return;
    case 16:
        X(dest, ir.BitFieldExtract(word, ByteBit(ir, offset, 16), ir.Imm32(16), is_signed));
        return;
    default:
        if (!IR::IsAligned(dest, static_cast<size_t>(bit_size / 32))) {
            throw NotImplementedException("Unaligned destination register {}", dest);
        }
        X(dest, word);
        for (int i = 1; i < bit_size / 32; ++i) {
            X(dest + i, LoadLocalWord(*this, ir.IAdd(offset, ir.Imm32(i * 4))));
        }
        return;
    }
}

void TranslatorVisitor::STL(Instruction insn) {
    const IR::U32 offset{WindowAddress(*this, insn, 64, false)};
    if (offset.IsImmediate() && offset.U32() >= env.LocalMemorySize()) {
        LOG_WARNING(Shader, "Storing local memory at {:#x} with a size of {:#x}, dropping",
                    offset.U32(), env.LocalMemorySize());
        return;
    }
    const IR::U32 word_offset{ir.ShiftRightLogical(offset, ir.Imm32(2))};
    const IR::Reg data{Reg(insn.SrcBReg())};
    const int bit_size{BitSize(insn).first};
    switch (bit_size) {
    case 8:
    case 16: {
        const IR::U32 bit{ByteBit(ir, offset, bit_size == 8 ? 24 : 16)};
        const IR::U32 merged{ir.BitFieldInsert(ir.LoadLocal(word_offset), X(data), bit,
                                               ir.Imm32(static_cast<u32>(bit_size)))};
        ir.WriteLocal(word_offset, merged);
        return;
    }
    default:
        if (!IR::IsAligned(data, static_cast<size_t>(bit_size / 32))) {
            throw NotImplementedException("Unaligned source register {}", data);
        }
        for (int i = 0; i < bit_size / 32; ++i) {
            ir.WriteLocal(ir.IAdd(word_offset, ir.Imm32(i)), X(data + i));
        }
        return;
    }
}

} // namespace Shader::Ampere
