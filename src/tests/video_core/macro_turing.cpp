// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <map>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "common/common_types.h"
#include "video_core/macro_turing.h"

using Tegra::TuringMacroEngine;

namespace {

// Register and operation numbers of mme_tu104.h
constexpr u32 R0 = 0, R1 = 1, R2 = 2, ZERO = 24, IMM = 25, IMMPAIR = 26, LOAD0 = 28, LOAD1 = 29;
enum Op : u32 {
    ADD = 0, ADDC = 1, MUL = 4, MULH = 5, EXTENDED = 7, MERGE = 16, STATE = 22, LOOP = 23,
    BEQ = 29, DREAD = 30, DWRITE = 31,
};
enum Out : u32 { NONE = 0, ALU0 = 1, ALU1 = 2, LOAD_0 = 3, IMM0 = 5, IMM1 = 6, OUT_IMM32 = 10 };

struct Alu {
    u32 op = ADD;
    u32 dst = ZERO;
    u32 src0 = ZERO;
    u32 src1 = ZERO;
};

struct Inst {
    bool end_next = false;
    u32 pred_mode = 0;
    u32 pred = ZERO;
    u32 imm[2]{};
    Alu alu[2]{};
    u32 mthd[2]{NONE, NONE};
    u32 emit[2]{NONE, NONE};
};

// mme_tu104_encode: fields of a 96-bit word, stored most significant word first
void Pack(u32 (&b)[3], u32 lo, u32 hi, u32 value) {
    for (u32 bit = lo; bit <= hi; ++bit) {
        if ((value >> (bit - lo)) & 1) {
            b[bit / 32] |= 1U << (bit % 32);
        }
    }
}

std::vector<u32> Encode(const std::vector<Inst>& program) {
    std::vector<u32> words;
    for (const Inst& inst : program) {
        u32 b[3]{};
        Pack(b, 0, 0, inst.end_next ? 1 : 0);
        Pack(b, 1, 4, inst.pred_mode);
        Pack(b, 5, 9, inst.pred);
        Pack(b, 10, 14, inst.alu[0].op);
        Pack(b, 15, 19, inst.alu[0].dst);
        Pack(b, 20, 24, inst.alu[0].src0);
        Pack(b, 25, 29, inst.alu[0].src1);
        Pack(b, 30, 45, inst.imm[0]);
        Pack(b, 46, 50, inst.alu[1].op);
        Pack(b, 51, 55, inst.alu[1].dst);
        Pack(b, 56, 60, inst.alu[1].src0);
        Pack(b, 61, 65, inst.alu[1].src1);
        Pack(b, 66, 81, inst.imm[1]);
        Pack(b, 82, 84, inst.mthd[0]);
        Pack(b, 85, 88, inst.emit[0]);
        Pack(b, 89, 91, inst.mthd[1]);
        Pack(b, 92, 95, inst.emit[1]);
        words.push_back(b[2]);
        words.push_back(b[1]);
        words.push_back(b[0]);
    }
    return words;
}

// Ends like Mesa's builder: two no-ops, the first with END_NEXT
std::vector<Inst> Finish(std::vector<Inst> program) {
    program.push_back(Inst{.end_next = true});
    program.push_back(Inst{});
    return program;
}

// Emits an ALU result to a method given by imm[1]
Inst EmitAlu(Alu alu, u32 method, u32 imm0 = 0) {
    return Inst{.imm = {imm0, method}, .alu = {alu, Alu{}}, .mthd = {IMM1, NONE},
                .emit = {ALU0, NONE}};
}

class FakeHost final : public TuringMacroEngine::Host {
public:
    u32 ReadState(u32 method) override {
        return state[method];
    }
    void CallMethod(u32 method, u32 value) override {
        calls.emplace_back(method, value);
    }
    void ReadMemory(GPUVAddr address, std::span<u32> data) override {
        for (u32 i = 0; i < data.size(); ++i) {
            data[i] = memory[address + i * 4];
        }
    }
    void WriteMemory(GPUVAddr address, std::span<const u32> data) override {
        for (u32 i = 0; i < data.size(); ++i) {
            memory[address + i * 4] = data[i];
        }
    }

    std::map<u32, u32> state;
    std::map<GPUVAddr, u32> memory;
    std::vector<std::pair<u32, u32>> calls;
};

using Calls = std::vector<std::pair<u32, u32>>;

Calls Run(TuringMacroEngine& mme, const std::vector<Inst>& program, std::vector<u32> params,
          FakeHost& host, u32 start = 0) {
    mme.SetUploadPointer(start);
    for (const u32 word : Encode(Finish(program))) {
        mme.Upload(word);
    }
    REQUIRE(mme.Execute(start, params, host));
    return host.calls;
}

} // Anonymous namespace

TEST_CASE("Turing MME: arithmetic and outputs", "[video_core][mme]") {
    TuringMacroEngine mme;
    FakeHost host;
    const auto calls = Run(mme,
                           {
                               // R0 = p0 + p1, emitted to 0x123
                               EmitAlu({ADD, R0, LOAD0, LOAD1}, 0x123),
                               // 64-bit add with the carry: sign extended 0xFFFF plus 1, then
                               // 0 + 0 + carry. Method IMM32 = 0xFFFF0001 is method 1, increment 0.
                               Inst{.imm = {0xFFFF, 1},
                                    .alu = {Alu{ADD, R1, IMM, IMMPAIR}, Alu{ADDC, R2, ZERO, ZERO}},
                                    .mthd = {OUT_IMM32, NONE},
                                    .emit = {ALU0, ALU1}},
                               // MERGE the low 4 bits of R0 (12) to bit 4 of zero
                               EmitAlu({MERGE, ZERO, ZERO, R0}, 0x30, (4U << 10) | (4U << 5) | 0),
                           },
                           {5, 7}, host);
    REQUIRE(calls == Calls{{0x123, 12}, {1, 0}, {1, 1}, {0x30, 0xC0}});
}

TEST_CASE("Turing MME: both ALUs read registers before writing", "[video_core][mme]") {
    TuringMacroEngine mme;
    FakeHost host;
    const auto calls = Run(mme,
                           {
                               EmitAlu({ADD, R0, LOAD0, ZERO}, 0x10),
                               EmitAlu({ADD, R1, LOAD0, ZERO}, 0x10),
                               // Swap R0 and R1 in one instruction
                               Inst{.alu = {Alu{ADD, R0, R1, ZERO}, Alu{ADD, R1, R0, ZERO}}},
                               EmitAlu({ADD, ZERO, R0, ZERO}, 0x20),
                               EmitAlu({ADD, ZERO, R1, ZERO}, 0x21),
                           },
                           {1, 2}, host);
    REQUIRE(calls == Calls{{0x10, 1}, {0x10, 2}, {0x20, 2}, {0x21, 1}});
}

TEST_CASE("Turing MME: multiply", "[video_core][mme]") {
    TuringMacroEngine mme;
    FakeHost host;
    const auto calls = Run(mme,
                           {
                               // -2 * 3 as a signed 64-bit product; MULH pairs with MUL in
                               // ALU1, like ADDC with ADD
                               Inst{.imm = {0x40, 0},
                                    .alu = {Alu{MUL, R0, LOAD0, LOAD1}, Alu{MULH, R1, ZERO, ZERO}},
                                    .mthd = {IMM0, NONE},
                                    .emit = {ALU1, NONE}},
                               EmitAlu({ADD, ZERO, R0, ZERO}, 0x41),
                           },
                           {0xFFFFFFFE, 3}, host);
    REQUIRE(calls == Calls{{0x40, 0xFFFFFFFF}, {0x41, 0xFFFFFFFA}});
}

TEST_CASE("Turing MME: loops and branches", "[video_core][mme]") {
    SECTION("A loop runs its body count times") {
        TuringMacroEngine mme;
        FakeHost host;
        const auto calls = Run(mme,
                               {
                                   // LOOP p0, the body is the next instruction
                                   Inst{.imm = {2, 0}, .alu = {Alu{LOOP, ZERO, LOAD0, ZERO}, Alu{}}},
                                   Inst{.mthd = {LOAD_0, NONE}, .emit = {LOAD_0, NONE}},
                               },
                               {3, 0x11, 0x22, 0x33}, host);
        REQUIRE(calls == Calls{{0x11, 0x11}, {0x22, 0x22}, {0x33, 0x33}});
    }
    SECTION("A loop with no iterations skips its body") {
        TuringMacroEngine mme;
        FakeHost host;
        const auto calls =
            Run(mme,
                {
                    Inst{.imm = {2, 0}, .alu = {Alu{LOOP, ZERO, LOAD0, ZERO}, Alu{}}},
                    Inst{.mthd = {LOAD_0, NONE}, .emit = {LOAD_0, NONE}},
                    EmitAlu({ADD, ZERO, ZERO, ZERO}, 0x50),
                },
                {0}, host);
        REQUIRE(calls == Calls{{0x50, 0}});
    }
    SECTION("A branch is taken when the condition equals bit 15") {
        TuringMacroEngine mme;
        FakeHost host;
        const auto calls = Run(mme,
                               {
                                   // if (p0 == 0) skip the next instruction
                                   Inst{.imm = {0x8000 | 2, 0},
                                        .alu = {Alu{BEQ, ZERO, LOAD0, ZERO}, Alu{}}},
                                   EmitAlu({ADD, ZERO, ZERO, ZERO}, 0x60),
                                   EmitAlu({ADD, ZERO, ZERO, ZERO}, 0x61),
                               },
                               {0}, host);
        REQUIRE(calls == Calls{{0x61, 0}});
    }
    SECTION("A branch offset of -0x1000 exits the macro") {
        TuringMacroEngine mme;
        FakeHost host;
        const auto calls = Run(mme,
                               {
                                   EmitAlu({ADD, ZERO, ZERO, ZERO}, 0x70),
                                   Inst{.imm = {0x8000 | 0x1000, 0},
                                        .alu = {Alu{BEQ, ZERO, ZERO, ZERO}, Alu{}}},
                                   EmitAlu({ADD, ZERO, ZERO, ZERO}, 0x71),
                               },
                               {}, host);
        REQUIRE(calls == Calls{{0x70, 0}});
    }
}

TEST_CASE("Turing MME: predicates", "[video_core][mme]") {
    TuringMacroEngine mme;
    FakeHost host;
    const auto calls = Run(mme,
                           {
                               EmitAlu({ADD, R0, LOAD0, ZERO}, 0x10),
                               // TTTT: every lane is skipped while the predicate is true
                               Inst{.pred_mode = 1, .pred = R0, .imm = {0, 0x80},
                                    .mthd = {IMM1, NONE}, .emit = {IMM1, NONE}},
                               // FFFF: every lane runs while the predicate is true
                               Inst{.pred_mode = 2, .pred = R0, .imm = {0, 0x81},
                                    .mthd = {IMM1, NONE}, .emit = {IMM1, NONE}},
                           },
                           {1}, host);
    REQUIRE(calls == Calls{{0x10, 1}, {0x81, 0x81}});
}

TEST_CASE("Turing MME: state, data RAM and DMA", "[video_core][mme]") {
    TuringMacroEngine mme;
    FakeHost host;
    host.state[0x45] = 0x1234;
    host.memory[0x1000] = 0xAA;
    host.memory[0x1004] = 0xBB;

    // SET_MME_MEM_ADDRESS_B, SET_MME_DATA_RAM_ADDRESS, MME_DMA_READ of two words
    REQUIRE(mme.ProcessMethod(0x554 / 4, 0x1000, host));
    REQUIRE(mme.ProcessMethod(0x558 / 4, 0x10, host));
    REQUIRE(mme.ProcessMethod(0x55C / 4, 2, host));
    REQUIRE(mme.DataRam()[0x10] == 0xAA);
    REQUIRE(mme.DataRam()[0x11] == 0xBB);
    REQUIRE(!mme.ProcessMethod(0x100, 0, host));

    // MME_DMA_READ_FIFOED of two words, which the EXTENDED barrier makes the parameters
    REQUIRE(mme.ProcessMethod(0x560 / 4, 2, host));
    const auto calls = Run(mme,
                           {
                               EmitAlu({STATE, ZERO, IMM, ZERO}, 0x20, 0x45),
                               EmitAlu({DREAD, ZERO, IMM, ZERO}, 0x21, 0x11),
                               Inst{.imm = {7, 0}, .alu = {Alu{DWRITE, ZERO, IMM, LOAD0}, Alu{}}},
                               EmitAlu({DREAD, ZERO, IMM, ZERO}, 0x22, 7),
                               Inst{.imm = {0x1000, 1},
                                    .alu = {Alu{EXTENDED, ZERO, IMM, IMMPAIR}, Alu{}}},
                               Inst{.mthd = {LOAD_0, NONE}, .emit = {LOAD_0, NONE}},
                               Inst{.mthd = {LOAD_0, NONE}, .emit = {LOAD_0, NONE}},
                           },
                           {0x99}, host);
    REQUIRE(calls == Calls{{0x20, 0x1234}, {0x21, 0xBB}, {0x22, 0x99}, {0xAA, 0xAA}, {0xBB, 0xBB}});

    // MME_DMA_WRITE of the data RAM word written by the macro, MME_DMA_SYNC
    REQUIRE(mme.ProcessMethod(0x554 / 4, 0x2000, host));
    REQUIRE(mme.ProcessMethod(0x558 / 4, 7, host));
    REQUIRE(mme.ProcessMethod(0x564 / 4, 1, host));
    REQUIRE(host.memory[0x2000] == 0x99);
    REQUIRE(mme.ProcessMethod(0x570 / 4, 0x5A, host));
    REQUIRE(mme.DataRam()[7] == 0x5A);

    // MME_DMA_REDUCTION add of the data RAM word into memory
    host.memory[0x2000] = 10;
    REQUIRE(mme.ProcessMethod(0x568 / 4, 0, host));
    REQUIRE(host.memory[0x2000] == 10 + 0x5A);
}
