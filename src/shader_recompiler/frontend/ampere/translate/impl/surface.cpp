// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <bit>
#include <optional>

#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ampere/translate/impl/impl.h"
#include "shader_recompiler/frontend/ir/modifiers.h"

// Surface loads and stores. .P accesses are formatted with a channel mask at [72..76), .D
// accesses are raw with a size at [73..76) and byte addressing (.BA) at bit 72.
namespace Shader::Ampere {
namespace {

enum class Dim : u64 {
    _1D,
    Buffer1D,
    Array1D,
    _2D,
    Array2D,
    _3D,
};

enum class Size : u64 {
    U8,
    S8,
    U16,
    S16,
    B32,
    B64,
    B128,
};

TextureType GetType(Dim dim) {
    switch (dim) {
    case Dim::_1D:
        return TextureType::Color1D;
    case Dim::Buffer1D:
        return TextureType::Buffer;
    case Dim::Array1D:
        return TextureType::ColorArray1D;
    case Dim::_2D:
        return TextureType::Color2D;
    case Dim::Array2D:
        return TextureType::ColorArray2D;
    case Dim::_3D:
        return TextureType::Color3D;
    }
    throw NotImplementedException("Surface dimension {}", static_cast<u64>(dim));
}

ImageFormat Format(Size size) {
    switch (size) {
    case Size::U8:
        return ImageFormat::R8_UINT;
    case Size::S8:
        return ImageFormat::R8_SINT;
    case Size::U16:
        return ImageFormat::R16_UINT;
    case Size::S16:
        return ImageFormat::R16_SINT;
    case Size::B32:
        return ImageFormat::R32_UINT;
    case Size::B64:
        return ImageFormat::R32G32_UINT;
    case Size::B128:
        return ImageFormat::R32G32B32A32_UINT;
    }
    throw NotImplementedException("Surface size {}", static_cast<u64>(size));
}

u32 SizeInBytes(Size size) {
    switch (size) {
    case Size::U8:
    case Size::S8:
        return 1;
    case Size::U16:
    case Size::S16:
        return 2;
    case Size::B32:
        return 4;
    case Size::B64:
        return 8;
    case Size::B128:
        return 16;
    }
    throw NotImplementedException("Surface size {}", static_cast<u64>(size));
}

struct Access {
    IR::U32 handle;
    IR::Value coords;
    IR::TextureInstInfo info;
    std::optional<Size> raw_size;
};

// Coordinates start at [24..32); array layers are 16-bit integers after the coordinates.
// Bindless forms hold the handle at [64..72), constant buffer forms reference it like textures.
Access MakeAccess(TranslatorVisitor& v, Instruction insn, bool is_raw,
                  AluForm cbuf_form = AluForm::CbufReg) {
    const auto dim{static_cast<Dim>(insn.Bits(61, 3))};
    const IR::Reg reg{Reg(insn.SrcA())};
    std::optional<Size> raw_size;
    if (is_raw) {
        raw_size = static_cast<Size>(insn.Bits(73, 3));
    }
    IR::U32 x{v.X(reg)};
    if (is_raw && insn.Bit(72)) {
        // .BA: the first coordinate is in bytes
        x = v.ir.ShiftRightLogical(x, v.ir.Imm32(std::countr_zero(SizeInBytes(*raw_size))));
    }
    const auto layer{[&](int index) {
        return v.ir.BitFieldExtract(v.X(reg + index), v.ir.Imm32(0), v.ir.Imm32(16));
    }};
    IR::Value coords;
    switch (dim) {
    case Dim::_1D:
    case Dim::Buffer1D:
        coords = x;
        break;
    case Dim::Array1D:
        coords = v.ir.CompositeConstruct(x, layer(1));
        break;
    case Dim::_2D:
        coords = v.ir.CompositeConstruct(x, v.X(reg + 1));
        break;
    case Dim::Array2D:
        coords = v.ir.CompositeConstruct(x, v.X(reg + 1), layer(2));
        break;
    case Dim::_3D:
        coords = v.ir.CompositeConstruct(x, v.X(reg + 1), v.X(reg + 2));
        break;
    default:
        throw NotImplementedException("Surface dimension {}", insn.Bits(61, 3));
    }
    IR::U32 handle;
    if (insn.Form() == cbuf_form) {
        const u32 index{static_cast<u32>(insn.Bits(54, 5))};
        const u32 offset{static_cast<u32>(insn.Bits(40, 14)) * 4};
        handle = v.ir.GetCbuf(v.ir.Imm32(index), v.ir.Imm32(offset));
    } else if (insn.Form() == AluForm::ImmReg) {
        handle = v.X(Reg(insn.SrcCReg()));
    } else {
        throw NotImplementedException("Surface handle form {}", static_cast<u32>(insn.Form()));
    }
    IR::TextureInstInfo info{};
    info.type.Assign(GetType(dim));
    info.image_format.Assign(raw_size ? Format(*raw_size) : ImageFormat::Typeless);
    return {handle, coords, info, raw_size};
}

u32 RegistersOf(Size size) {
    return size == Size::B128 ? 4 : size == Size::B64 ? 2 : 1;
}

} // Anonymous namespace

void TranslatorVisitor::SULD(Instruction insn) {
    const bool is_raw{insn.Opcode() % 0x200 == 0x199 || insn.Opcode() % 0x200 == 0x19a};
    const Access access{MakeAccess(*this, insn, is_raw)};
    const IR::Value result{ir.ImageRead(access.handle, access.coords, access.info)};
    IR::Reg dest{Reg(insn.Dest())};
    if (access.raw_size) {
        for (u32 i = 0; i < RegistersOf(*access.raw_size); ++i) {
            X(dest + static_cast<int>(i), IR::U32{ir.CompositeExtract(result, i)});
        }
    } else {
        const u32 mask{static_cast<u32>(insn.Bits(72, 4))};
        for (u32 component = 0; component < 4; ++component) {
            if (((mask >> component) & 1) != 0) {
                X(dest, IR::U32{ir.CompositeExtract(result, component)});
                ++dest;
            }
        }
    }
    if (insn.Bits(81, 3) != 7) {
        SetP(insn.Bits(81, 3), ir.Imm1(false));
    }
}

void TranslatorVisitor::SUST(Instruction insn) {
    const bool is_raw{insn.Opcode() % 0x200 == 0x19d || insn.Opcode() % 0x200 == 0x19e};
    const Access access{MakeAccess(*this, insn, is_raw)};
    const IR::Reg data{Reg(insn.SrcBReg())};
    const u32 count{access.raw_size ? RegistersOf(*access.raw_size) : 4};
    if (!access.raw_size && insn.Bits(72, 4) != 0xf) {
        throw NotImplementedException("SUST with channel mask {:x}", insn.Bits(72, 4));
    }
    std::array<IR::U32, 4> color{ir.Imm32(0), ir.Imm32(0), ir.Imm32(0), ir.Imm32(0)};
    for (u32 i = 0; i < count; ++i) {
        color[i] = X(data + static_cast<int>(i));
    }
    ir.ImageWrite(access.handle, access.coords,
                  ir.CompositeConstruct(color[0], color[1], color[2], color[3]), access.info);
}

namespace {

// SUATOM.D Rd, [Ra], Rb, cbuf and SURED.D [Ra], Rb, cbuf: raw 32-bit image atomics. The
// operation is at [87..91) and the type at [73..76) as in ATOMG. The constant buffer form is 2
// here, not 5 as for SULD and SUST.
// SUATOM can also write a predicate at [81..84); SURED has no such field.
IR::Value SurfaceAtomic(TranslatorVisitor& v, Instruction insn, const char* name,
                        bool has_predicate) {
    IR::IREmitter& ir{v.ir};
    if (insn.Form() != AluForm::RegImm || insn.Bit(72) ||
        (has_predicate && insn.Bits(81, 3) != 7)) {
        throw NotImplementedException("{} form {} with .BA {} or predicate output", name,
                                      static_cast<u32>(insn.Form()), insn.Bit(72));
    }
    const u64 type{insn.Bits(73, 3)};
    if (type > 1) {
        throw NotImplementedException("{} type {}", name, type);
    }
    const bool is_signed{type == 1};
    Access access{MakeAccess(v, insn, false, AluForm::RegImm)};
    access.info.image_format.Assign(ImageFormat::R32_UINT);
    const IR::U32 value{v.X(Reg(insn.SrcBReg()))};
    const IR::Value& handle{access.handle};
    const IR::Value& coords{access.coords};
    IR::Value result;
    switch (insn.Bits(87, 4)) {
    case 0:
        result = ir.ImageAtomicIAdd(handle, coords, value, access.info);
        break;
    case 1:
        result = ir.ImageAtomicIMin(handle, coords, value, is_signed, access.info);
        break;
    case 2:
        result = ir.ImageAtomicIMax(handle, coords, value, is_signed, access.info);
        break;
    case 3:
        result = ir.ImageAtomicInc(handle, coords, value, access.info);
        break;
    case 4:
        result = ir.ImageAtomicDec(handle, coords, value, access.info);
        break;
    case 5:
        result = ir.ImageAtomicAnd(handle, coords, value, access.info);
        break;
    case 6:
        result = ir.ImageAtomicOr(handle, coords, value, access.info);
        break;
    case 7:
        result = ir.ImageAtomicXor(handle, coords, value, access.info);
        break;
    case 8:
        result = ir.ImageAtomicExchange(handle, coords, value, access.info);
        break;
    default:
        throw NotImplementedException("{} operation {}", name, insn.Bits(87, 4));
    }
    return result;
}

} // Anonymous namespace

void TranslatorVisitor::SUATOM(Instruction insn) {
    X(Reg(insn.Dest()), IR::U32{SurfaceAtomic(*this, insn, "SUATOM", true)});
}

void TranslatorVisitor::SURED(Instruction insn) {
    static_cast<void>(SurfaceAtomic(*this, insn, "SURED", false));
}

} // namespace Shader::Ampere
