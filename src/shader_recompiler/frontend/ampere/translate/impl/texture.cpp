// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <optional>

#include <boost/container/static_vector.hpp>

#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ampere/translate/impl/impl.h"
#include "shader_recompiler/frontend/ir/modifiers.h"

// Texture instructions. Source and destination layouts follow Mesa NAK (nak_nir_lower_tex.c and
// from_nir.rs) and were checked against nvcc output.
namespace Shader::Ampere {
namespace {

constexpr u64 PT{7};

enum class Dim : u64 {
    _1D,
    _2D,
    _3D,
    Cube,
    Array1D,
    Array2D,
    Array3D,
    ArrayCube,
};

enum class LodMode : u64 {
    Auto,
    Zero,
    Bias,
    Lod,
    Clamp,
    BiasClamp,
};

Shader::TextureType GetType(Dim dim) {
    switch (dim) {
    case Dim::_1D:
        return Shader::TextureType::Color1D;
    case Dim::_2D:
        return Shader::TextureType::Color2D;
    case Dim::_3D:
        return Shader::TextureType::Color3D;
    case Dim::Cube:
        return Shader::TextureType::ColorCube;
    case Dim::Array1D:
        return Shader::TextureType::ColorArray1D;
    case Dim::Array2D:
        return Shader::TextureType::ColorArray2D;
    case Dim::ArrayCube:
        return Shader::TextureType::ColorArrayCube;
    case Dim::Array3D:
        break;
    }
    throw NotImplementedException("Texture dimension {}", static_cast<u64>(dim));
}

bool IsArray(Dim dim) {
    return dim >= Dim::Array1D;
}

u32 NumCoords(Dim dim) {
    switch (dim) {
    case Dim::_1D:
    case Dim::Array1D:
        return 1;
    case Dim::_2D:
    case Dim::Array2D:
        return 2;
    default:
        return 3;
    }
}

Dim GetDim(Instruction insn) {
    return static_cast<Dim>(insn.Bits(61, 3));
}

// Bindless forms take the handle from a register, cbuf forms read it from a constant buffer
bool IsBindless(Instruction insn) {
    return insn.Bit(59);
}

IR::U32 CbufHandle(TranslatorVisitor& v, Instruction insn) {
    if (insn.Form() != AluForm::CbufReg) {
        throw NotImplementedException("Texture handle form {}", static_cast<u32>(insn.Form()));
    }
    const u32 index{static_cast<u32>(insn.Bits(54, 5))};
    const u32 offset{static_cast<u32>(insn.Bits(40, 14)) * 4};
    return v.ir.GetCbuf(v.ir.Imm32(index), v.ir.Imm32(offset));
}

// Registers holding the source list [src0..., src1...]. Separate vectors start at Ra and Rb;
// the scalar form spreads up to four values over the register pairs at Ra and Rb.
class Sources {
public:
    explicit Sources(Instruction insn, u32 src0_count, u32 src1_count, bool scalar) {
        const IR::Reg ra{Reg(insn.SrcA())};
        const IR::Reg rb{Reg(insn.SrcBReg())};
        if (!scalar) {
            for (u32 i = 0; i < src0_count; ++i) {
                regs.push_back(ra + static_cast<int>(i));
            }
            for (u32 i = 0; i < src1_count; ++i) {
                regs.push_back(rb + static_cast<int>(i));
            }
            return;
        }
        switch (src0_count + src1_count) {
        case 1:
            regs = {ra};
            return;
        case 2:
            regs = {ra, rb};
            return;
        case 3:
            regs = {ra, ra + 1, rb};
            return;
        case 4:
            regs = {ra, ra + 1, rb, rb + 1};
            return;
        default:
            throw NotImplementedException("Scalar texture with {} sources",
                                          src0_count + src1_count);
        }
    }

    IR::Reg Next() {
        return regs.at(index++);
    }

private:
    boost::container::static_vector<IR::Reg, 12> regs;
    size_t index{};
};

// Array layers are stored as 16-bit integers
IR::Value MakeCoords(TranslatorVisitor& v, Sources& src, Dim dim, bool integer) {
    std::optional<IR::Value> layer;
    if (IsArray(dim)) {
        const IR::U32 raw{v.X(src.Next())};
        layer = integer ? IR::Value{v.ir.BitFieldExtract(raw, v.ir.Imm32(0), v.ir.Imm32(16))}
                        : IR::Value{v.ir.ConvertUToF(32, 16, raw)};
    }
    boost::container::static_vector<IR::Value, 4> components;
    for (u32 i = 0; i < NumCoords(dim); ++i) {
        const IR::Reg reg{src.Next()};
        components.push_back(integer ? IR::Value{v.X(reg)} : IR::Value{v.F(reg)});
    }
    if (layer) {
        components.push_back(*layer);
    }
    switch (components.size()) {
    case 1:
        return components[0];
    case 2:
        return v.ir.CompositeConstruct(components[0], components[1]);
    case 3:
        return v.ir.CompositeConstruct(components[0], components[1], components[2]);
    default:
        return v.ir.CompositeConstruct(components[0], components[1], components[2],
                                       components[3]);
    }
}

// Signed offsets packed with the given bit stride per component
IR::Value MakeOffset(TranslatorVisitor& v, const IR::U32& packed, Dim dim, u32 stride, u32 bits) {
    const auto component{[&](u32 i) {
        return v.ir.BitFieldExtract(packed, v.ir.Imm32(i * stride), v.ir.Imm32(bits), true);
    }};
    switch (dim) {
    case Dim::_1D:
    case Dim::Array1D:
        return component(0);
    case Dim::_2D:
    case Dim::Array2D:
        return v.ir.CompositeConstruct(component(0), component(1));
    case Dim::_3D:
        return v.ir.CompositeConstruct(component(0), component(1), component(2));
    default:
        throw NotImplementedException("Offset on texture dimension {}", static_cast<u64>(dim));
    }
}

// Enabled channels are written in order; on Volta and later the first two go to [16..24) and the
// rest to [64..72)
void WriteChannels(TranslatorVisitor& v, Instruction insn, const IR::Value& result,
                   bool depth_result, bool integer_result) {
    const u32 mask{static_cast<u32>(insn.Bits(72, 4))};
    IR::Reg dest{Reg(insn.Dest())};
    IR::Reg dest2{Reg(insn.Bits(64, 8))};
    u32 written{};
    for (u32 element = 0; element < 4; ++element) {
        if (((mask >> element) & 1) == 0) {
            continue;
        }
        const IR::Reg reg{written < 2 ? dest + static_cast<int>(written)
                                      : dest2 + static_cast<int>(written - 2)};
        ++written;
        if (depth_result) {
            v.F(reg, element < 3 ? IR::F32{result} : v.ir.Imm32(1.0f));
        } else if (integer_result) {
            v.X(reg, IR::U32{v.ir.CompositeExtract(result, element)});
        } else {
            v.F(reg, IR::F32{v.ir.CompositeExtract(result, element)});
        }
    }
}

void WriteSparse(TranslatorVisitor& v, Instruction insn, const IR::Value& result) {
    if (insn.Bits(81, 3) != PT) {
        v.SetP(insn.Bits(81, 3), v.ir.LogicalNot(v.ir.GetSparseFromOp(result)));
    }
}

} // Anonymous namespace

void TranslatorVisitor::TEX(Instruction insn) {
    const Dim dim{GetDim(insn)};
    const auto lod_mode{static_cast<LodMode>(insn.Bits(87, 3))};
    if (lod_mode == LodMode::Clamp || lod_mode == LodMode::BiasClamp) {
        throw NotImplementedException("TEX with LOD clamp");
    }
    const bool bindless{IsBindless(insn)};
    const bool has_lod{lod_mode == LodMode::Bias || lod_mode == LodMode::Lod};
    const bool aoffi{insn.Bit(76)};
    const bool dc{insn.Bit(78)};
    const u32 src0_count{(IsArray(dim) ? 1u : 0u) + NumCoords(dim)};
    const u32 src1_count{(bindless ? 1u : 0u) + (has_lod ? 1u : 0u) + (aoffi ? 1u : 0u) +
                         (dc ? 1u : 0u)};
    Sources src{insn, src0_count, src1_count, insn.Bit(60)};

    const IR::Value coords{MakeCoords(*this, src, dim, false)};
    const IR::U32 handle{bindless ? X(src.Next()) : CbufHandle(*this, insn)};
    const IR::F32 lod{has_lod ? F(src.Next()) : ir.Imm32(0.0f)};
    IR::Value offset;
    if (aoffi) {
        offset = MakeOffset(*this, X(src.Next()), dim, 4, 4);
    }
    IR::F32 dref;
    if (dc) {
        dref = F(src.Next());
    }
    IR::TextureInstInfo info{};
    info.type.Assign(GetType(dim));
    info.is_depth.Assign(dc ? 1 : 0);
    info.has_bias.Assign(lod_mode == LodMode::Bias ? 1 : 0);
    info.ndv_is_active.Assign(insn.Bit(77) ? 1 : 0);

    const bool explicit_lod{lod_mode == LodMode::Zero || lod_mode == LodMode::Lod};
    IR::Value result;
    if (!dc) {
        result = explicit_lod ? ir.ImageSampleExplicitLod(handle, coords, lod, offset, info)
                              : ir.ImageSampleImplicitLod(handle, coords, lod, offset, {}, info);
    } else {
        result = explicit_lod
                     ? ir.ImageSampleDrefExplicitLod(handle, coords, dref, lod, offset, info)
                     : ir.ImageSampleDrefImplicitLod(handle, coords, dref, lod, offset, {}, info);
    }
    WriteChannels(*this, insn, result, dc, false);
    WriteSparse(*this, insn, result);
}

void TranslatorVisitor::TLD(Instruction insn) {
    const Dim dim{GetDim(insn)};
    const auto lod_mode{static_cast<LodMode>(insn.Bits(87, 3))};
    if (lod_mode != LodMode::Zero && lod_mode != LodMode::Lod) {
        throw NotImplementedException("TLD LOD mode {}", insn.Bits(87, 3));
    }
    const bool bindless{IsBindless(insn)};
    const bool has_lod{lod_mode == LodMode::Lod};
    const bool aoffi{insn.Bit(76)};
    const bool multisample{insn.Bit(78)};
    const u32 src0_count{(IsArray(dim) ? 1u : 0u) + NumCoords(dim)};
    const u32 src1_count{(bindless ? 1u : 0u) + (has_lod ? 1u : 0u) + (aoffi ? 1u : 0u) +
                         (multisample ? 1u : 0u)};
    Sources src{insn, src0_count, src1_count, insn.Bit(60)};

    const IR::Value coords{MakeCoords(*this, src, dim, true)};
    const IR::U32 handle{bindless ? X(src.Next()) : CbufHandle(*this, insn)};
    const IR::U32 lod{has_lod ? X(src.Next()) : ir.Imm32(0U)};
    IR::Value offset;
    if (aoffi) {
        offset = MakeOffset(*this, X(src.Next()), dim, 4, 4);
    }
    IR::U32 sample_index;
    if (multisample) {
        sample_index = X(src.Next());
    }
    IR::TextureInstInfo info{};
    info.type.Assign(GetType(dim));
    const IR::Value result{ir.ImageFetch(handle, coords, offset, lod, sample_index, info)};
    WriteChannels(*this, insn, result, false, false);
    WriteSparse(*this, insn, result);
}

void TranslatorVisitor::TLD4(Instruction insn) {
    const Dim dim{GetDim(insn)};
    // 1 is one offset for the footprint (AOFFI), 2 one offset per gathered texel (PTP)
    const u64 offset_mode{insn.Bits(76, 2)};
    if (offset_mode > 2) {
        throw NotImplementedException("TLD4 offset mode {}", offset_mode);
    }
    const bool bindless{IsBindless(insn)};
    const bool aoffi{offset_mode == 1};
    const bool ptp{offset_mode == 2};
    const bool dc{insn.Bit(78)};
    const u32 src0_count{(IsArray(dim) ? 1u : 0u) + NumCoords(dim)};
    const u32 src1_count{(bindless ? 1u : 0u) + (aoffi ? 1u : 0u) + (ptp ? 2u : 0u) +
                         (dc ? 1u : 0u)};
    Sources src{insn, src0_count, src1_count, insn.Bit(60)};

    const IR::Value coords{MakeCoords(*this, src, dim, false)};
    const IR::U32 handle{bindless ? X(src.Next()) : CbufHandle(*this, insn)};
    IR::Value offset;
    IR::Value offset2;
    if (aoffi) {
        offset = MakeOffset(*this, X(src.Next()), dim, 8, 8);
    } else if (ptp) {
        // Two words of signed 6-bit x, y byte pairs: texels 0 and 1, then texels 2 and 3, as
        // in the Maxwell encoding
        const auto texel_offsets{[&](const IR::U32& packed) {
            const auto field{[&](u32 byte) {
                return ir.BitFieldExtract(packed, ir.Imm32(byte * 8), ir.Imm32(6), true);
            }};
            return ir.CompositeConstruct(field(0), field(1), field(2), field(3));
        }};
        offset = texel_offsets(X(src.Next()));
        offset2 = texel_offsets(X(src.Next()));
    }
    IR::TextureInstInfo info{};
    info.type.Assign(GetType(dim));
    info.is_depth.Assign(dc ? 1 : 0);
    info.gather_component.Assign(static_cast<u32>(insn.Bits(87, 2)));
    const IR::Value result{dc ? ir.ImageGatherDref(handle, coords, offset, offset2, F(src.Next()), info)
                              : ir.ImageGather(handle, coords, offset, offset2, info)};
    WriteChannels(*this, insn, result, false, false);
    WriteSparse(*this, insn, result);
}

void TranslatorVisitor::TXD(Instruction insn) {
    const Dim dim{GetDim(insn)};
    if (dim == Dim::Cube || dim == Dim::ArrayCube || dim == Dim::_3D) {
        throw NotImplementedException("TXD on texture dimension {}", insn.Bits(61, 3));
    }
    const bool bindless{IsBindless(insn)};
    const bool aoffi{insn.Bit(76)};
    const u32 num_coords{NumCoords(dim)};
    // The handle comes first; the layer, packed with the offset, follows the coordinates
    const bool has_layer_word{IsArray(dim) || aoffi};
    const u32 src0_count{(bindless ? 1u : 0u) + num_coords + (has_layer_word ? 1u : 0u)};
    Sources src{insn, src0_count, num_coords * 2, false};

    const IR::U32 handle{bindless ? X(src.Next()) : CbufHandle(*this, insn)};
    boost::container::static_vector<IR::Value, 3> components;
    for (u32 i = 0; i < num_coords; ++i) {
        components.push_back(F(src.Next()));
    }
    IR::Value offset;
    if (has_layer_word) {
        const IR::U32 word{X(src.Next())};
        if (IsArray(dim)) {
            components.push_back(ir.ConvertUToF(32, 16, ir.BitFieldExtract(word, ir.Imm32(0),
                                                                           ir.Imm32(16))));
        }
        if (aoffi) {
            offset = MakeOffset(*this, ir.ShiftRightLogical(word, ir.Imm32(16)), dim, 4, 4);
        }
    }
    const IR::Value coords{components.size() == 1 ? components[0]
                           : components.size() == 2
                               ? ir.CompositeConstruct(components[0], components[1])
                               : ir.CompositeConstruct(components[0], components[1],
                                                       components[2])};
    // Derivatives are interleaved: ddx.x, ddy.x, ddx.y, ddy.y
    boost::container::static_vector<IR::F32, 4> derivatives;
    for (u32 i = 0; i < num_coords * 2; ++i) {
        derivatives.push_back(F(src.Next()));
    }
    const IR::Value derivative_vector{
        num_coords == 1 ? ir.CompositeConstruct(derivatives[0], derivatives[1])
                        : ir.CompositeConstruct(derivatives[0], derivatives[1], derivatives[2],
                                                derivatives[3])};
    IR::TextureInstInfo info{};
    info.type.Assign(GetType(dim));
    info.num_derivatives.Assign(num_coords);
    const IR::Value result{ir.ImageGradient(handle, coords, derivative_vector, offset, {}, info)};
    WriteChannels(*this, insn, result, false, false);
    WriteSparse(*this, insn, result);
}

void TranslatorVisitor::TXQ(Instruction insn) {
    if (insn.Bits(62, 2) != 0) {
        throw NotImplementedException("TXQ query {}", insn.Bits(62, 2));
    }
    const bool bindless{IsBindless(insn)};
    Sources src{insn, (bindless ? 1u : 0u) + 1u, 0, false};
    const IR::U32 handle{bindless ? X(src.Next()) : CbufHandle(*this, insn)};
    const IR::U32 lod{X(src.Next())};
    const bool needs_mips{insn.Bit(75)};
    const IR::Value result{ir.ImageQueryDimension(handle, lod, ir.Imm1(!needs_mips))};
    WriteChannels(*this, insn, result, false, true);
}

} // namespace Shader::Ampere
