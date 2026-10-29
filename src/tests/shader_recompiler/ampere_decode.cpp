// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <string_view>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include "common/common_types.h"
#include "shader_recompiler/frontend/ampere/decode.h"

using namespace Shader::Ampere;

namespace {

struct OpcodeVector {
    u64 lo;
    u64 hi;
    Opcode expected;
};

// One valid encoding of every SM86 opcode, as accepted by nvdisasm -b SM86
constexpr OpcodeVector OPCODE_VECTORS[] = {
#include "ampere_decode_vectors.inc"
};

struct CompilerVector {
    u64 lo;
    u64 hi;
    std::string_view mnemonic;
};

// Instructions emitted by nvcc -arch=sm_86, with the mnemonic printed by nvdisasm
constexpr CompilerVector COMPILER_VECTORS[] = {
#include "ampere_decode_compiler_vectors.inc"
};

std::string_view BaseMnemonic(std::string_view name) {
    return name.substr(0, name.find_first_of(". "));
}

} // Anonymous namespace

TEST_CASE("Ampere decode: every opcode", "[shader][ampere]") {
    for (const auto& v : OPCODE_VECTORS) {
        INFO(fmt::format("{:016x} {:016x}", v.lo, v.hi));
        REQUIRE(Decode(Instruction{v.lo, v.hi}) == v.expected);
    }
}

TEST_CASE("Ampere decode: compiler output", "[shader][ampere]") {
    for (const auto& v : COMPILER_VECTORS) {
        INFO(fmt::format("{:016x} {:016x} {}", v.lo, v.hi, v.mnemonic));
        REQUIRE(BaseMnemonic(NameOf(Decode(Instruction{v.lo, v.hi}))) == BaseMnemonic(v.mnemonic));
    }
}

TEST_CASE("Ampere decode: unknown opcode", "[shader][ampere]") {
    // 0x7ff is rejected by nvdisasm
    REQUIRE(Decode(Instruction{0x7fff, 0x000fc00000000000ULL}) == Opcode::Invalid);
}

TEST_CASE("Ampere instruction fields", "[shader][ampere]") {
    // @!P2 FFMA R1, R2, c[0x3][0x58], R4
    const Instruction cbuf{0xa23 | (2ULL << 12) | (1ULL << 15) | (1ULL << 16) | (2ULL << 24) |
                               (0x58ULL << 38) | (3ULL << 54),
                           0x000fc00000000004ULL};
    REQUIRE(Decode(cbuf) == Opcode::FFMA);
    REQUIRE(cbuf.Form() == AluForm::CbufReg);
    REQUIRE(cbuf.Pred() == 2);
    REQUIRE(cbuf.PredNegated());
    REQUIRE(cbuf.Dest() == 1);
    REQUIRE(cbuf.SrcA() == 2);
    REQUIRE(!cbuf.CbufBindless());
    REQUIRE(cbuf.CbufIndex() == 3);
    REQUIRE(cbuf.CbufOffset() == 0x58);
    REQUIRE(cbuf.SrcCReg() == 4);

    // FFMA R1, R2, R5, 1 (immediate in the third operand)
    const Instruction imm{0x7423 | (1ULL << 16) | (2ULL << 24) | (0x3f800000ULL << 32),
                          0x000fc00000000005ULL};
    REQUIRE(imm.Form() == AluForm::RegImm);
    REQUIRE(imm.Imm32() == 0x3f800000);
    REQUIRE(imm.SrcCReg() == 5);

    // FFMA R1, R2, UR9, R5
    const Instruction ureg{0x7c23 | (1ULL << 16) | (2ULL << 24) | (9ULL << 32),
                           0x000fc00008000005ULL};
    REQUIRE(ureg.Form() == AluForm::URegReg);
    REQUIRE(ureg.HasUniformOperand());
    REQUIRE(ureg.SrcBUReg() == 9);

    // Scheduling control of the encoding above: no stall, no barriers
    REQUIRE(ureg.StallCycles() == 0);
    REQUIRE(ureg.WriteBarrier() == 7);
    REQUIRE(ureg.ReadBarrier() == 7);
    REQUIRE(ureg.WaitMask() == 0);
}
