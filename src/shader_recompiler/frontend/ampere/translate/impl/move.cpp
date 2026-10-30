// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ampere/decode.h"
#include "shader_recompiler/frontend/ampere/translate/impl/impl.h"
#include "shader_recompiler/frontend/maxwell/translate/impl/common_funcs.h"

namespace Shader::Ampere {
namespace {

constexpr u64 PT{7};
constexpr u32 SRZ{0xff};

enum class LoadSize : u64 {
    U8,
    S8,
    U16,
    S16,
    B32,
    B64,
};

// Loads a constant buffer value of the given size into one or two registers
void LoadConstant(TranslatorVisitor& v, IR::Reg dest, LoadSize size, const IR::U32& index,
                  const IR::U32& offset) {
    switch (size) {
    case LoadSize::U8:
        v.X(dest, IR::U32{v.ir.GetCbuf(index, offset, 8, false)});
        return;
    case LoadSize::S8:
        v.X(dest, IR::U32{v.ir.GetCbuf(index, offset, 8, true)});
        return;
    case LoadSize::U16:
        v.X(dest, IR::U32{v.ir.GetCbuf(index, offset, 16, false)});
        return;
    case LoadSize::S16:
        v.X(dest, IR::U32{v.ir.GetCbuf(index, offset, 16, true)});
        return;
    case LoadSize::B32:
        v.X(dest, IR::U32{v.ir.GetCbuf(index, offset, 32, false)});
        return;
    case LoadSize::B64: {
        if (!IR::IsAligned(dest, 2)) {
            throw NotImplementedException("Unaligned destination register");
        }
        const IR::Value vector{v.ir.GetCbuf(index, offset, 64, false)};
        for (int i = 0; i < 2; ++i) {
            v.X(dest + i, IR::U32{v.ir.CompositeExtract(vector, static_cast<size_t>(i))});
        }
        return;
    }
    }
    throw NotImplementedException("Constant load size {}", static_cast<u64>(size));
}

} // Anonymous namespace

void TranslatorVisitor::MOV(Instruction insn) {
    if (insn.Bits(72, 4) != 0xf) {
        throw NotImplementedException("MOV with quad lane mask {:x}", insn.Bits(72, 4));
    }
    X(Reg(insn.Dest()), SrcB(insn));
}

void TranslatorVisitor::UMOV(Instruction insn) {
    X(UReg(insn.Bits(16, 6)), SrcB(insn, true));
}

void TranslatorVisitor::R2UR(Instruction insn) {
    // Every invocation keeps its own copy of uniform registers, so the value is moved as is
    X(UReg(insn.Bits(16, 6)), SrcA(insn));
    if (insn.Bits(81, 3) != PT) {
        if (insn.Bit(84)) {
            throw NotImplementedException("R2UR.OR");
        }
        SetP(insn.Bits(81, 3), ir.Imm1(true));
    }
}

void TranslatorVisitor::S2R(Instruction insn) {
    X(Reg(insn.Dest()), Maxwell::ReadSpecialRegister(ir, static_cast<u32>(insn.Bits(72, 8))));
}

void TranslatorVisitor::S2UR(Instruction insn) {
    X(UReg(insn.Bits(16, 6)),
      Maxwell::ReadSpecialRegister(ir, static_cast<u32>(insn.Bits(72, 8))));
}

void TranslatorVisitor::CS2R(Instruction insn) {
    const u32 index{static_cast<u32>(insn.Bits(72, 8))};
    const bool is_64{insn.Bit(80)};
    const IR::Reg dest{Reg(insn.Dest())};
    if (index == SRZ) {
        X(dest, ir.Imm32(0));
        if (is_64) {
            X(dest + 1, ir.Imm32(0));
        }
        return;
    }
    X(dest, Maxwell::ReadSpecialRegister(ir, index));
    if (is_64) {
        X(dest + 1, Maxwell::ReadSpecialRegister(ir, index + 1));
    }
}

void TranslatorVisitor::LDC(Instruction insn) {
    if (insn.Bit(91)) {
        throw NotImplementedException("Bindless LDC");
    }
    const u64 mode{insn.Bits(78, 2)};
    const IR::U32 imm_index{ir.Imm32(insn.CbufIndex())};
    const IR::U32 imm_offset{ir.Imm32(static_cast<s32>(static_cast<s16>(insn.CbufOffset())))};
    const IR::U32 address{ir.IAdd(SrcA(insn), imm_offset)};
    IR::U32 index{imm_index};
    IR::U32 offset{address};
    switch (mode) {
    case 0:
        break;
    case 2:
        // Segmented: the upper half of the address selects the constant buffer
        index = ir.IAdd(ir.BitFieldExtract(address, ir.Imm32(16), ir.Imm32(16)), imm_index);
        offset = ir.BitFieldExtract(address, ir.Imm32(0), ir.Imm32(16));
        break;
    default:
        throw NotImplementedException("LDC mode {}", mode);
    }
    LoadConstant(*this, Reg(insn.Dest()), static_cast<LoadSize>(insn.Bits(73, 3)), index, offset);
}

void TranslatorVisitor::ULDC(Instruction insn) {
    const bool indexed{Decode(insn) == Opcode::ULDC_INDEXED};
    if (!indexed && insn.Bit(91)) {
        throw NotImplementedException("Bindless ULDC");
    }
    const u64 mode{insn.Bits(78, 2)};
    const IR::U32 imm_index{ir.Imm32(insn.CbufIndex())};
    const IR::U32 imm_offset{ir.Imm32(static_cast<s32>(static_cast<s16>(insn.CbufOffset())))};
    const IR::U32 address{indexed ? IR::U32{ir.IAdd(SrcA(insn, true), imm_offset)} : imm_offset};
    IR::U32 index{imm_index};
    IR::U32 offset{address};
    switch (mode) {
    case 0:
        break;
    case 2:
        index = ir.IAdd(ir.BitFieldExtract(address, ir.Imm32(16), ir.Imm32(16)), imm_index);
        offset = ir.BitFieldExtract(address, ir.Imm32(0), ir.Imm32(16));
        break;
    default:
        throw NotImplementedException("ULDC mode {}", mode);
    }
    LoadConstant(*this, UReg(insn.Bits(16, 6)), static_cast<LoadSize>(insn.Bits(73, 3)), index,
                 offset);
}

} // namespace Shader::Ampere
