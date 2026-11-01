// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <bit>
#include <optional>
#include <vector>

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

// Bytes per texel of the uncompressed formats raw accesses can cover several texels of, 0 otherwise
u32 TexelBytes(TexturePixelFormat format) {
    switch (format) {
    case TexturePixelFormat::R8_UNORM:
    case TexturePixelFormat::R8_SNORM:
    case TexturePixelFormat::R8_SINT:
    case TexturePixelFormat::R8_UINT:
        return 1;
    case TexturePixelFormat::R5G6B5_UNORM:
    case TexturePixelFormat::B5G6R5_UNORM:
    case TexturePixelFormat::A1R5G5B5_UNORM:
    case TexturePixelFormat::A1B5G5R5_UNORM:
    case TexturePixelFormat::A5B5G5R1_UNORM:
    case TexturePixelFormat::A4B4G4R4_UNORM:
    case TexturePixelFormat::R16_FLOAT:
    case TexturePixelFormat::R16_UNORM:
    case TexturePixelFormat::R16_SNORM:
    case TexturePixelFormat::R16_UINT:
    case TexturePixelFormat::R16_SINT:
    case TexturePixelFormat::R8G8_UNORM:
    case TexturePixelFormat::R8G8_SNORM:
    case TexturePixelFormat::R8G8_SINT:
    case TexturePixelFormat::R8G8_UINT:
        return 2;
    case TexturePixelFormat::A8B8G8R8_UNORM:
    case TexturePixelFormat::A8B8G8R8_SNORM:
    case TexturePixelFormat::A8B8G8R8_SINT:
    case TexturePixelFormat::A8B8G8R8_UINT:
    case TexturePixelFormat::A8B8G8R8_SRGB:
    case TexturePixelFormat::B8G8R8A8_UNORM:
    case TexturePixelFormat::B8G8R8A8_SRGB:
    case TexturePixelFormat::A2B10G10R10_UNORM:
    case TexturePixelFormat::A2B10G10R10_UINT:
    case TexturePixelFormat::A2R10G10B10_UNORM:
    case TexturePixelFormat::B10G11R11_FLOAT:
    case TexturePixelFormat::E5B9G9R9_FLOAT:
    case TexturePixelFormat::R32_FLOAT:
    case TexturePixelFormat::R32_UINT:
    case TexturePixelFormat::R32_SINT:
    case TexturePixelFormat::R16G16_UNORM:
    case TexturePixelFormat::R16G16_SNORM:
    case TexturePixelFormat::R16G16_FLOAT:
    case TexturePixelFormat::R16G16_UINT:
    case TexturePixelFormat::R16G16_SINT:
        return 4;
    case TexturePixelFormat::R16G16B16A16_FLOAT:
    case TexturePixelFormat::R16G16B16A16_UNORM:
    case TexturePixelFormat::R16G16B16A16_SNORM:
    case TexturePixelFormat::R16G16B16A16_SINT:
    case TexturePixelFormat::R16G16B16A16_UINT:
    case TexturePixelFormat::R16G16B16X16_FLOAT:
    case TexturePixelFormat::R32G32_FLOAT:
    case TexturePixelFormat::R32G32_SINT:
    case TexturePixelFormat::R32G32_UINT:
        return 8;
    default:
        return 0;
    }
}

Size UnsignedSizeOfBytes(u32 bytes) {
    switch (bytes) {
    case 1:
        return Size::U8;
    case 2:
        return Size::U16;
    case 4:
        return Size::B32;
    default:
        return Size::B64;
    }
}

// A raw access reads or writes whole texels of the view, so an access larger than the texels of
// the surface (SUST.D.BA of 32 bits on R16_FLOAT, two texels at once in Fast Fusion) becomes one
// access of each texel it covers. Their coordinates are in `coords`.
struct Access {
    IR::U32 handle;
    std::vector<IR::Value> coords;
    IR::TextureInstInfo info;
    std::optional<Size> raw_size;
    u32 texel_bytes{};
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
    IR::U32 handle;
    u32 surface_texel_bytes{};
    if (insn.Form() == cbuf_form) {
        const u32 index{static_cast<u32>(insn.Bits(54, 5))};
        const u32 offset{static_cast<u32>(insn.Bits(40, 14)) * 4};
        handle = v.ir.GetCbuf(v.ir.Imm32(index), v.ir.Imm32(offset));
        if (is_raw) {
            surface_texel_bytes =
                TexelBytes(v.env.ReadTexturePixelFormat(v.env.ReadCbufValue(index, offset)));
        }
    } else if (insn.Form() == AluForm::ImmReg) {
        handle = v.X(Reg(insn.SrcCReg()));
    } else {
        throw NotImplementedException("Surface handle form {}", static_cast<u32>(insn.Form()));
    }
    // Sign extended raw sizes keep their own format
    const bool can_split{raw_size && *raw_size != Size::S8 && *raw_size != Size::S16};
    const u32 access_bytes{raw_size ? SizeInBytes(*raw_size) : 0};
    const u32 texel_bytes{can_split && surface_texel_bytes != 0 &&
                                  surface_texel_bytes < access_bytes
                              ? surface_texel_bytes
                              : access_bytes};
    IR::U32 x{v.X(reg)};
    if (is_raw && insn.Bit(72)) {
        // .BA: the first coordinate is in bytes
        x = v.ir.ShiftRightLogical(x, v.ir.Imm32(std::countr_zero(texel_bytes)));
    }
    const auto layer{[&](int index) {
        return v.ir.BitFieldExtract(v.X(reg + index), v.ir.Imm32(0), v.ir.Imm32(16));
    }};
    const auto make_coords{[&](const IR::U32& first) -> IR::Value {
        switch (dim) {
        case Dim::_1D:
        case Dim::Buffer1D:
            return first;
        case Dim::Array1D:
            return v.ir.CompositeConstruct(first, layer(1));
        case Dim::_2D:
            return v.ir.CompositeConstruct(first, v.X(reg + 1));
        case Dim::Array2D:
            return v.ir.CompositeConstruct(first, v.X(reg + 1), layer(2));
        case Dim::_3D:
            return v.ir.CompositeConstruct(first, v.X(reg + 1), v.X(reg + 2));
        default:
            throw NotImplementedException("Surface dimension {}", insn.Bits(61, 3));
        }
    }};
    std::vector<IR::Value> coords{make_coords(x)};
    for (u32 i = 1; raw_size && i < access_bytes / texel_bytes; ++i) {
        coords.push_back(make_coords(v.ir.IAdd(x, v.ir.Imm32(i))));
    }
    if (raw_size && texel_bytes != access_bytes) {
        raw_size = UnsignedSizeOfBytes(texel_bytes);
    }
    IR::TextureInstInfo info{};
    info.type.Assign(GetType(dim));
    info.image_format.Assign(raw_size ? Format(*raw_size) : ImageFormat::Typeless);
    return {handle, std::move(coords), info, raw_size, texel_bytes};
}

u32 RegistersOf(Size size) {
    return size == Size::B128 ? 4 : size == Size::B64 ? 2 : 1;
}

} // Anonymous namespace

void TranslatorVisitor::SULD(Instruction insn) {
    const bool is_raw{insn.Opcode() % 0x200 == 0x199 || insn.Opcode() % 0x200 == 0x19a};
    const Access access{MakeAccess(*this, insn, is_raw)};
    IR::Reg dest{Reg(insn.Dest())};
    if (access.coords.size() > 1) {
        // Several texels make up the registers, the first one in the lowest bytes
        const u32 bytes{access.texel_bytes};
        std::array<IR::U32, 4> words{ir.Imm32(0), ir.Imm32(0), ir.Imm32(0), ir.Imm32(0)};
        for (u32 i = 0; i < access.coords.size(); ++i) {
            const IR::Value texel{ir.ImageRead(access.handle, access.coords[i], access.info)};
            const u32 byte{i * bytes};
            IR::U32& word{words[byte / 4]};
            const IR::U32 low{ir.CompositeExtract(texel, 0)};
            if (bytes >= 4) {
                word = low;
                if (bytes == 8) {
                    words[byte / 4 + 1] = IR::U32{ir.CompositeExtract(texel, 1)};
                }
            } else {
                word = ir.BitFieldInsert(word, low, ir.Imm32((byte % 4) * 8), ir.Imm32(bytes * 8));
            }
        }
        const u32 total{static_cast<u32>(access.coords.size()) * bytes};
        for (u32 i = 0; i < total / 4; ++i) {
            X(dest + static_cast<int>(i), words[i]);
        }
        if (insn.Bits(81, 3) != 7) {
            SetP(insn.Bits(81, 3), ir.Imm1(false));
        }
        return;
    }
    const IR::Value result{ir.ImageRead(access.handle, access.coords[0], access.info)};
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
    if (access.coords.size() > 1) {
        const u32 bytes{access.texel_bytes};
        for (u32 i = 0; i < access.coords.size(); ++i) {
            const u32 byte{i * bytes};
            const IR::U32 word{X(data + static_cast<int>(byte / 4))};
            std::array<IR::U32, 2> texel{ir.Imm32(0), ir.Imm32(0)};
            if (bytes >= 4) {
                texel[0] = word;
                if (bytes == 8) {
                    texel[1] = X(data + static_cast<int>(byte / 4 + 1));
                }
            } else {
                texel[0] = ir.BitFieldExtract(word, ir.Imm32((byte % 4) * 8), ir.Imm32(bytes * 8));
            }
            ir.ImageWrite(access.handle, access.coords[i],
                          ir.CompositeConstruct(texel[0], texel[1], ir.Imm32(0), ir.Imm32(0)),
                          access.info);
        }
        return;
    }
    std::array<IR::U32, 4> color{ir.Imm32(0), ir.Imm32(0), ir.Imm32(0), ir.Imm32(0)};
    const u32 mask{access.raw_size ? 0xfU : static_cast<u32>(insn.Bits(72, 4))};
    if (access.raw_size) {
        for (u32 i = 0; i < RegistersOf(*access.raw_size); ++i) {
            color[i] = X(data + static_cast<int>(i));
        }
    } else {
        // The enabled channels take consecutive registers, as for SULD. A formatted store
        // writes every channel, so the others keep what the surface holds.
        std::optional<IR::Value> old;
        if (mask != 0xf) {
            old = ir.ImageRead(access.handle, access.coords[0], access.info);
        }
        IR::Reg reg{data};
        for (u32 component = 0; component < 4; ++component) {
            if (((mask >> component) & 1) != 0) {
                color[component] = X(reg);
                ++reg;
            } else if (old) {
                color[component] = IR::U32{ir.CompositeExtract(*old, component)};
            }
        }
    }
    ir.ImageWrite(access.handle, access.coords[0],
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
    const IR::Value& coords{access.coords[0]};
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
