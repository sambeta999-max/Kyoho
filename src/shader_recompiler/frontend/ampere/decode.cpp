// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>

#include "common/common_types.h"
#include "shader_recompiler/frontend/ampere/decode.h"

namespace Shader::Ampere {
namespace {

// Encodings in ampere.inc cover the 12 opcode bits [11..0], '-' marks a don't-care bit
consteval std::pair<u32, u32> MaskValueFromEncoding(const char* data) noexcept {
    u32 mask = 0, value = 0, bit = 1U << 11;
    for (; *data != '\0'; ++data) {
        switch (*data) {
        case '0':
            mask |= bit;
            bit >>= 1;
            break;
        case '1':
            mask |= bit;
            value |= bit;
            bit >>= 1;
            break;
        case '-':
            bit >>= 1;
            break;
        default:
            break;
        }
    }
    return {mask, value};
}

consteval std::array<Opcode, 0x1000> BuildDecodeTable() {
    std::array<Opcode, 0x1000> table{};
    table.fill(Opcode::Invalid);
    // Visit every opcode matching an encoding by enumerating its don't-care bits
    const auto fill = [&table](std::pair<u32, u32> p, Opcode opcode) {
        const u32 free_bits = ~p.first & 0xfff;
        u32 sub = 0;
        do {
            const u32 op = p.second | sub;
            if (table[op] == Opcode::Invalid) {
                table[op] = opcode;
            }
            sub = (sub - free_bits) & free_bits;
        } while (sub != 0);
    };
#define INST(name, cute, encode) fill(MaskValueFromEncoding(encode), Opcode::name);
#define INST_ALIAS(name, cute, encode) INST(name, cute, encode)
#include "ampere.inc"
#undef INST_ALIAS
#undef INST
    return table;
}

constexpr std::array<Opcode, 0x1000> DECODE_TABLE = BuildDecodeTable();

} // Anonymous namespace

Opcode Decode(const Instruction& insn) noexcept {
    return DECODE_TABLE[insn.Opcode()];
}

} // namespace Shader::Ampere
