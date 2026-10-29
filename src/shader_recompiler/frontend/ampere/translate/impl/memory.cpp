// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

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

// Signed 24-bit immediate offset at [40..64)
s32 ImmediateOffset(Instruction insn) {
    return static_cast<s32>(static_cast<u32>(insn.Bits(40, 24)) << 8) >> 8;
}

// Forms with bit 11 of the opcode set add a uniform register to the address
bool HasUniformAddress(Instruction insn) {
    return (insn.Opcode() & 0x800) != 0;
}

IR::U64 GlobalAddress(TranslatorVisitor& v, Instruction insn, u32 ureg_offset) {
    const bool reg_is_64{HasUniformAddress(insn) ? insn.Bit(90) : insn.Bit(72)};
    // With a 64-bit register address, a 64-bit uniform register is a memory descriptor
    // (desc[URx]) and does not take part in the address; checked on GA102
    const bool is_descriptor{HasUniformAddress(insn) && insn.Bit(90) && insn.Bit(72)};
    const bool has_ureg{HasUniformAddress(insn) && !is_descriptor};
    const bool ureg_is_64{has_ureg && insn.Bit(72)};
    // Built as IAdd64(base, immediate), the form storage buffer tracking expects
    std::optional<IR::U64> base;
    const auto add_register{[&](IR::Reg reg, bool is_64) {
        if (IR::IsZero(reg)) {
            return;
        }
        const IR::U64 value{is_64 ? v.L(reg) : IR::U64{v.ir.UConvert(64, v.X(reg))}};
        base = base ? IR::U64{v.ir.IAdd(*base, value)} : value;
    }};
    add_register(Reg(insn.SrcA()), reg_is_64);
    if (has_ureg) {
        add_register(UReg(insn.Bits(ureg_offset, 6)), ureg_is_64);
    }
    const s64 offset{ImmediateOffset(insn)};
    if (!base) {
        return v.ir.Imm64(static_cast<u64>(offset));
    }
    return offset == 0 ? *base : IR::U64{v.ir.IAdd(*base, v.ir.Imm64(static_cast<u64>(offset)))};
}

// Shared and local addresses; shared accesses may scale the register
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
    if (insn.Bits(64, 3) != 0 || insn.Bit(67) || insn.Bits(81, 3) != PT) {
        throw NotImplementedException("LDG with predicates");
    }
    const IR::U64 address{GlobalAddress(*this, insn, 32)};
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
}

void TranslatorVisitor::STG(Instruction insn) {
    const IR::U64 address{GlobalAddress(*this, insn, 64)};
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
