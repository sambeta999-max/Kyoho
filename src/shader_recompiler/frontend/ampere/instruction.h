// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "common/common_types.h"

namespace Shader::Ampere {

// Operand layout of the 128-bit SM70+ encoding.
// Field positions follow Mesa NAK's sm70_encode.rs (MIT).
enum class AluForm : u32 {
    RegReg = 1,   // b = R[32..40], c = R[64..72]
    RegImm = 2,   // b = R[64..72], c = imm[32..64]
    RegCbuf = 3,  // b = R[64..72], c = c[32..59]
    ImmReg = 4,   // b = imm[32..64], c = R[64..72]
    CbufReg = 5,  // b = c[32..59], c = R[64..72]
    URegReg = 6,  // b = UR[32..38], c = R[64..72]
    RegUReg = 7,  // b = R[64..72], c = UR[32..38]
};

struct Instruction {
    u64 lo;
    u64 hi;

    [[nodiscard]] constexpr u64 Bits(u32 offset, u32 count) const noexcept {
        const u64 mask = count >= 64 ? ~u64{0} : (u64{1} << count) - 1;
        if (offset >= 64) {
            return (hi >> (offset - 64)) & mask;
        }
        u64 value = lo >> offset;
        if (offset + count > 64) {
            value |= hi << (64 - offset);
        }
        return value & mask;
    }

    [[nodiscard]] constexpr bool Bit(u32 offset) const noexcept {
        return Bits(offset, 1) != 0;
    }

    [[nodiscard]] constexpr u32 Opcode() const noexcept {
        return static_cast<u32>(Bits(0, 12));
    }

    [[nodiscard]] constexpr u32 BaseOpcode() const noexcept {
        return static_cast<u32>(Bits(0, 9));
    }

    [[nodiscard]] constexpr AluForm Form() const noexcept {
        return static_cast<AluForm>(Bits(9, 3));
    }

    // Guard predicate, 7 is PT
    [[nodiscard]] constexpr u32 Pred() const noexcept {
        return static_cast<u32>(Bits(12, 3));
    }

    [[nodiscard]] constexpr bool PredNegated() const noexcept {
        return Bit(15);
    }

    // 255 is RZ, 63 is URZ
    [[nodiscard]] constexpr u32 Dest() const noexcept {
        return static_cast<u32>(Bits(16, 8));
    }

    [[nodiscard]] constexpr u32 SrcA() const noexcept {
        return static_cast<u32>(Bits(24, 8));
    }

    [[nodiscard]] constexpr u32 SrcBReg() const noexcept {
        return static_cast<u32>(Bits(32, 8));
    }

    [[nodiscard]] constexpr u32 SrcBUReg() const noexcept {
        return static_cast<u32>(Bits(32, 6));
    }

    [[nodiscard]] constexpr u32 Imm32() const noexcept {
        return static_cast<u32>(Bits(32, 32));
    }

    [[nodiscard]] constexpr u32 SrcCReg() const noexcept {
        return static_cast<u32>(Bits(64, 8));
    }

    // Constant buffer operand at [32..59]; bindless form takes the index from a UR
    [[nodiscard]] constexpr bool CbufBindless() const noexcept {
        return Bit(91);
    }

    [[nodiscard]] constexpr u32 CbufUReg() const noexcept {
        return static_cast<u32>(Bits(32, 6));
    }

    // Byte offset
    [[nodiscard]] constexpr u32 CbufOffset() const noexcept {
        return static_cast<u32>(Bits(38, 16));
    }

    [[nodiscard]] constexpr u32 CbufIndex() const noexcept {
        return static_cast<u32>(Bits(54, 5));
    }

    // Set by uniform datapath ALU instructions and by forms taking a UR operand
    [[nodiscard]] constexpr bool HasUniformOperand() const noexcept {
        return Bit(91);
    }

    // Scheduling control
    [[nodiscard]] constexpr u32 StallCycles() const noexcept {
        return static_cast<u32>(Bits(105, 4));
    }

    [[nodiscard]] constexpr bool Yield() const noexcept {
        return Bit(109);
    }

    [[nodiscard]] constexpr u32 WriteBarrier() const noexcept {
        return static_cast<u32>(Bits(110, 3));
    }

    [[nodiscard]] constexpr u32 ReadBarrier() const noexcept {
        return static_cast<u32>(Bits(113, 3));
    }

    [[nodiscard]] constexpr u32 WaitMask() const noexcept {
        return static_cast<u32>(Bits(116, 6));
    }

    [[nodiscard]] constexpr u32 ReuseMask() const noexcept {
        return static_cast<u32>(Bits(122, 4));
    }
};
static_assert(sizeof(Instruction) == 16);

} // namespace Shader::Ampere
