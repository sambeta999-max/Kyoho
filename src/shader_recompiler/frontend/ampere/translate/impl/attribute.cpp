// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>

#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ampere/translate/impl/impl.h"
#include "shader_recompiler/frontend/ir/attribute.h"
#include "shader_recompiler/frontend/ir/patch.h"
#include "shader_recompiler/program_header.h"

namespace Shader::Ampere {
namespace {

constexpr u64 PT{7};

// Byte address at [40..50) and component count minus one at [74..76)
u32 AttributeIndex(Instruction insn) {
    const u32 address{static_cast<u32>(insn.Bits(40, 10))};
    if (address % 4 != 0) {
        throw NotImplementedException("Unaligned attribute address {:#x}", address);
    }
    return address / 4;
}

u32 NumComponents(Instruction insn) {
    return static_cast<u32>(insn.Bits(74, 2)) + 1;
}

} // Anonymous namespace

void TranslatorVisitor::ALD(Instruction insn) {
    if (!IR::IsZero(Reg(insn.SrcA())) || insn.Bit(77) || insn.Bit(79)) {
        throw NotImplementedException("Indexed, physical or output ALD");
    }
    const bool patch{insn.Bit(76)};
    const IR::U32 vertex{X(Reg(insn.SrcBReg()))};
    const u32 index{AttributeIndex(insn)};
    const IR::Reg dest{Reg(insn.Dest())};
    for (u32 element = 0; element < NumComponents(insn); ++element) {
        const int offset{static_cast<int>(element)};
        if (patch) {
            F(dest + offset, ir.GetPatch(IR::Patch{index + element}));
        } else {
            F(dest + offset, ir.GetAttribute(IR::Attribute{index + element}, vertex));
        }
    }
}

void TranslatorVisitor::AST(Instruction insn) {
    if (!IR::IsZero(Reg(insn.SrcA())) || insn.Bit(77)) {
        throw NotImplementedException("Indexed or physical AST");
    }
    const bool patch{insn.Bit(76)};
    const IR::U32 vertex{X(Reg(insn.SrcCReg()))};
    const u32 index{AttributeIndex(insn)};
    const IR::Reg src{Reg(insn.SrcBReg())};
    for (u32 element = 0; element < NumComponents(insn); ++element) {
        const int offset{static_cast<int>(element)};
        if (patch) {
            ir.SetPatch(IR::Patch{index + element}, F(src + offset));
        } else {
            ir.SetAttribute(IR::Attribute{index + element}, F(src + offset), vertex);
        }
    }
}

// Perspective-interpolated inputs are returned premultiplied by W; the shader divides by W
// itself, as on earlier generations
void TranslatorVisitor::IPA(Instruction insn) {
    if (insn.Bits(76, 2) == 2) {
        throw NotImplementedException("IPA at an offset");
    }
    const IR::Attribute attribute{static_cast<u64>(insn.Bits(64, 8))};
    IR::F32 value{ir.GetAttribute(attribute)};
    const bool is_legacy{IR::IsLegacyAttribute(attribute)};
    const bool is_pass{insn.Bits(78, 2) == 0};
    if (is_pass && (IR::IsGeneric(attribute) || is_legacy)) {
        bool is_perspective{is_legacy};
        if (!is_legacy) {
            const std::array input_map{
                env.SPH().ps.GenericInputMap(IR::GenericAttributeIndex(attribute))};
            Shader::PixelImap effective_imap{Shader::PixelImap::Unused};
            for (const Shader::PixelImap component : input_map) {
                if (component != Shader::PixelImap::Unused) {
                    effective_imap = component;
                    break;
                }
            }
            is_perspective = effective_imap == Shader::PixelImap::Perspective ||
                             effective_imap == Shader::PixelImap::Unused;
        }
        if (is_perspective) {
            value = ir.FPMul(value, ir.GetAttribute(IR::Attribute::PositionW));
        }
    }
    F(Reg(insn.Dest()), value);
    if (insn.Bits(81, 3) != PT) {
        SetP(insn.Bits(81, 3), ir.Imm1(true));
    }
}

} // namespace Shader::Ampere
