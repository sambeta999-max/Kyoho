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

// ISBERD Rd, [Ra + imm]: reads the internal stage buffer entry that maps a geometry program's
// input index (from SR_INVOCATION_INFO) to the vertex handle ALD takes. The host takes vertex
// indices directly, so the entry is the index itself, as on Maxwell. Sizes and the patch,
// primitive, skew and output modes are at [74..80).
void TranslatorVisitor::ISBERD(Instruction insn) {
    if (env.ShaderStage() == Stage::Task || env.ShaderStage() == Stage::Mesh) {
        IsbeAccess(insn, false);
        return;
    }
    if (insn.Bits(74, 6) != 0) {
        throw NotImplementedException("ISBERD mode {:#x}", insn.Bits(74, 6));
    }
    const u32 offset{static_cast<u32>(insn.Bits(40, 16))};
    const IR::U32 index{X(Reg(insn.SrcA()))};
    X(Reg(insn.Dest()), offset == 0 ? index : IR::U32{ir.IAdd(index, ir.Imm32(offset))});
}

// ISBEWR [Ra + imm], Rb: task and mesh programs' stage buffer writes (see IsbeAccess)
void TranslatorVisitor::ISBEWR(Instruction insn) {
    IsbeAccess(insn, true);
}

// ISBERD/ISBEWR in task and mesh programs. The stage buffer holds a task program's workgroup index
// and its payload, and a mesh program's index, the payload it reads and its outputs: Ra at
// [24..32), the byte offset at [40..64), ISBERD's Rd at [16..24), ISBEWR's data at [32..40), .32
// at 75 (U8 otherwise), the region at [76..78) (3 attributes, 0 the primitive map), .SKEW at 78
// and .O (the output buffer) at 79. Addresses are linear bytes in the region.
void TranslatorVisitor::IsbeAccess(Instruction insn, bool write) {
    const u64 region{insn.Bits(76, 2)};
    const bool output{insn.Bits(79, 1) != 0};
    if (insn.Bits(64, 11) != 0 || (region != 0 && region != 3) || write != output ||
        (!write && region != 3)) {
        throw NotImplementedException("Stage buffer access mode {:#x}", insn.Bits(64, 16));
    }
    const u32 offset{static_cast<u32>(insn.Bits(40, 24))};
    const IR::U32 address{ir.IAdd(X(Reg(insn.Bits(24, 8))), ir.Imm32(offset))};
    if (!write) {
        X(Reg(insn.Dest()), ir.IsbeRead(address));
        return;
    }
    const IR::U32 value{X(Reg(insn.Bits(32, 8)))};
    const bool is_32{insn.Bits(75, 1) != 0};
    if (region == 3) {
        if (!is_32) {
            throw NotImplementedException("U8 stage buffer attribute write");
        }
        ir.IsbeWriteAttribute(address, value);
    } else if (is_32) {
        ir.IsbeWriteMapU32(address, value);
    } else {
        ir.IsbeWriteMapU8(address, value);
    }
}

// OUT.{FINAL,EMIT,CUT,EMIT_THEN_CUT} Rd, Ra, Rb: [78..80) selects the operation, Ra is the
// output handle and Rb the stream. The host has no handle; Rd is written with zero.
void TranslatorVisitor::OUT(Instruction insn) {
    const IR::U32 stream{ir.BitwiseAnd(SrcB(insn), ir.Imm32(3))};
    const u64 operation{insn.Bits(78, 2)};
    if ((operation & 1) != 0) {
        ir.EmitVertex(stream);
    }
    if ((operation & 2) != 0) {
        ir.EndPrimitive(stream);
    }
    X(Reg(insn.Dest()), ir.Imm32(0));
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
