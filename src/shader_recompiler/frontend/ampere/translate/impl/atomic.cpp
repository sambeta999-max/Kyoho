// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ampere/translate/impl/impl.h"

namespace Shader::Ampere {
namespace {

enum class AtomOp : u64 {
    Add,
    Min,
    Max,
    Inc,
    Dec,
    And,
    Or,
    Xor,
    Exch,
};

enum class AtomType : u64 {
    U32,
    S32,
    U64,
    F32,
    F16x2,
    S64,
    F64,
};

bool Is64(AtomType type) {
    return type == AtomType::U64 || type == AtomType::S64;
}

bool IsSigned(AtomType type) {
    return type == AtomType::S32 || type == AtomType::S64;
}

IR::Value GlobalAtomic(TranslatorVisitor& v, AtomOp op, AtomType type, const IR::U64& address,
                       IR::Reg data) {
    IR::IREmitter& ir{v.ir};
    if (type == AtomType::F32 || type == AtomType::F16x2) {
        if (op != AtomOp::Add) {
            throw NotImplementedException("Floating point atomic {}", static_cast<u64>(op));
        }
        if (type == AtomType::F32) {
            return ir.GlobalAtomicF32Add(address, v.F(data), {});
        }
        return ir.GlobalAtomicF16x2Add(address, ir.UnpackFloat2x16(v.X(data)), {});
    }
    if (type == AtomType::F64) {
        throw NotImplementedException("F64 atomic");
    }
    const IR::U32U64 value{Is64(type) ? IR::U32U64{v.L(data)} : IR::U32U64{v.X(data)}};
    switch (op) {
    case AtomOp::Add:
        return ir.GlobalAtomicIAdd(address, value);
    case AtomOp::Min:
        return ir.GlobalAtomicIMin(address, value, IsSigned(type));
    case AtomOp::Max:
        return ir.GlobalAtomicIMax(address, value, IsSigned(type));
    case AtomOp::Inc:
    case AtomOp::Dec:
        if (Is64(type)) {
            throw NotImplementedException("64-bit wrapping atomic");
        }
        return op == AtomOp::Inc ? ir.GlobalAtomicInc(address, IR::U32{value})
                                 : ir.GlobalAtomicDec(address, IR::U32{value});
    case AtomOp::And:
        return ir.GlobalAtomicAnd(address, value);
    case AtomOp::Or:
        return ir.GlobalAtomicOr(address, value);
    case AtomOp::Xor:
        return ir.GlobalAtomicXor(address, value);
    case AtomOp::Exch:
        return ir.GlobalAtomicExchange(address, value);
    }
    throw NotImplementedException("Atomic operation {}", static_cast<u64>(op));
}

void WriteResult(TranslatorVisitor& v, IR::Reg dest, AtomType type, const IR::Value& result) {
    if (Is64(type)) {
        v.L(dest, IR::U64{result});
    } else if (type == AtomType::F32) {
        v.F(dest, IR::F32{result});
    } else if (type == AtomType::F16x2) {
        v.X(dest, v.ir.PackFloat2x16(result));
    } else {
        v.X(dest, IR::U32{result});
    }
}

} // Anonymous namespace

// Operation at [87..91), type at [73..76); the 64-bit register flag of uniform forms is bit 70
void TranslatorVisitor::ATOMG(Instruction insn) {
    if (insn.Bits(81, 3) != 7) {
        throw NotImplementedException("ATOMG predicate output");
    }
    const auto op{static_cast<AtomOp>(insn.Bits(87, 4))};
    const auto type{static_cast<AtomType>(insn.Bits(73, 3))};
    const IR::U64 address{GlobalAddress(*this, insn, 64, 70)};
    const IR::Value result{GlobalAtomic(*this, op, type, address, Reg(insn.SrcBReg()))};
    WriteResult(*this, Reg(insn.Dest()), type, result);
}

// Reduction without a result; the operation is at [87..90)
void TranslatorVisitor::RED(Instruction insn) {
    const auto op{static_cast<AtomOp>(insn.Bits(87, 3))};
    const auto type{static_cast<AtomType>(insn.Bits(73, 3))};
    const IR::U64 address{GlobalAddress(*this, insn, 64, 90)};
    static_cast<void>(GlobalAtomic(*this, op, type, address, Reg(insn.SrcBReg())));
}

void TranslatorVisitor::ATOMS(Instruction insn) {
    const auto op{static_cast<AtomOp>(insn.Bits(87, 4))};
    const auto type{static_cast<AtomType>(insn.Bits(73, 3))};
    if (type != AtomType::U32 && type != AtomType::S32) {
        throw NotImplementedException("Shared atomic type {}", insn.Bits(73, 3));
    }
    const IR::U32 offset{WindowAddress(*this, insn, 64, true)};
    const IR::U32 value{X(Reg(insn.SrcBReg()))};
    const bool is_signed{IsSigned(type)};
    IR::U32 result;
    switch (op) {
    case AtomOp::Add:
        result = ir.SharedAtomicIAdd(offset, value);
        break;
    case AtomOp::Min:
        result = ir.SharedAtomicIMin(offset, value, is_signed);
        break;
    case AtomOp::Max:
        result = ir.SharedAtomicIMax(offset, value, is_signed);
        break;
    case AtomOp::Inc:
        result = ir.SharedAtomicInc(offset, value);
        break;
    case AtomOp::Dec:
        result = ir.SharedAtomicDec(offset, value);
        break;
    case AtomOp::And:
        result = ir.SharedAtomicAnd(offset, value);
        break;
    case AtomOp::Or:
        result = ir.SharedAtomicOr(offset, value);
        break;
    case AtomOp::Xor:
        result = ir.SharedAtomicXor(offset, value);
        break;
    case AtomOp::Exch:
        result = IR::U32{ir.SharedAtomicExchange(offset, value)};
        break;
    default:
        throw NotImplementedException("Shared atomic operation {}", insn.Bits(87, 4));
    }
    X(Reg(insn.Dest()), result);
}

} // namespace Shader::Ampere
