// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <bit>
#include <cstdlib>

#include "common/logging.h"
#include "video_core/engines/semaphore.h"
#include <fmt/ranges.h>
#include "video_core/draw_capture.h"
#include "video_core/macro_turing.h"
#ifdef ARCHITECTURE_x86_64
#include "video_core/macro_turing_jit.h"
#endif

namespace Tegra {
namespace {

enum class Reg : u32 {
    R23 = 23,
    Zero = 24,
    Imm = 25,
    ImmPair = 26,
    Imm32 = 27,
    Load0 = 28,
    Load1 = 29,
};

enum class AluOp : u32 {
    Add,
    AddC,
    Sub,
    SubB,
    Mul,
    MulH,
    MulU,
    Extended,
    Clz,
    Sll,
    Srl,
    Sra,
    And,
    Nand,
    Or,
    Xor,
    Merge,
    Slt,
    SltU,
    Sle,
    SleU,
    Seq,
    State,
    Loop,
    Jal,
    Blt,
    BltU,
    Ble,
    BleU,
    Beq,
    DRead,
    DWrite,
};

enum class OutOp : u32 {
    None,
    Alu0,
    Alu1,
    Load0,
    Load1,
    Imm0,
    Imm1,
    Reserved,
    ImmHigh0,
    ImmHigh1,
    Imm32,
};

/// Lanes enabled by each predicate mode for a true and a false predicate: ALU0, ALU1, OUT0, OUT1
/// are T(rue), F(alse) or U(nconditional)
constexpr std::array<const char*, 16> PREDICATE_MODES{
    "UUUU", "TTTT", "FFFF", "TTUU", "FFUU", "TFUU", "TUUU", "FUUU",
    "UUTT", "UUTF", "UUTU", "UUFT", "UUFF", "UUFU", "UUUT", "UUUF",
};

// MME DMA methods of the 3D class, as method indices
constexpr u32 SET_MME_MEM_ADDRESS_A = 0x0550 / 4;
constexpr u32 SET_MME_MEM_ADDRESS_B = 0x0554 / 4;
constexpr u32 SET_MME_DATA_RAM_ADDRESS = 0x0558 / 4;
constexpr u32 MME_DMA_READ = 0x055C / 4;
constexpr u32 MME_DMA_READ_FIFOED = 0x0560 / 4;
constexpr u32 MME_DMA_WRITE = 0x0564 / 4;
constexpr u32 MME_DMA_REDUCTION = 0x0568 / 4;
constexpr u32 MME_DMA_SYSMEMBAR = 0x056C / 4;
constexpr u32 MME_DMA_SYNC = 0x0570 / 4;
constexpr u32 SET_MME_DATA_FIFO_CONFIG = 0x0574 / 4;
constexpr u32 MME_DMA_WRITE_METHOD_BARRIER = 0x0DEC / 4;

/// One decoded instruction. The three words are stored most significant first.
struct Instruction {
    explicit Instruction(std::span<const u32, 3> words)
        : bits{(u64{words[1]} << 32) | words[2], words[0]} {}

    [[nodiscard]] u32 Field(u32 lo, u32 count) const {
        const u64 low = bits[0];
        const u64 high = bits[1];
        u64 value;
        if (lo >= 64) {
            value = high >> (lo - 64);
        } else if (lo + count <= 64) {
            value = low >> lo;
        } else {
            value = (low >> lo) | (high << (64 - lo));
        }
        return static_cast<u32>(value & ((u64{1} << count) - 1));
    }

    [[nodiscard]] bool EndNext() const {
        return Field(0, 1) != 0;
    }
    [[nodiscard]] u32 PredMode() const {
        return Field(1, 4);
    }
    [[nodiscard]] u32 Pred() const {
        return Field(5, 5);
    }
    [[nodiscard]] AluOp Op(u32 lane) const {
        return static_cast<AluOp>(Field(lane == 0 ? 10 : 46, 5));
    }
    [[nodiscard]] u32 Dst(u32 lane) const {
        return Field(lane == 0 ? 15 : 51, 5);
    }
    [[nodiscard]] u32 Src(u32 lane, u32 index) const {
        return Field((lane == 0 ? 20 : 56) + index * 5, 5);
    }
    [[nodiscard]] u32 Imm(u32 lane) const {
        return Field(lane == 0 ? 30 : 66, 16);
    }
    [[nodiscard]] OutOp Method(u32 lane) const {
        return static_cast<OutOp>(Field(lane == 0 ? 82 : 89, 3));
    }
    [[nodiscard]] OutOp Emit(u32 lane) const {
        return static_cast<OutOp>(Field(lane == 0 ? 85 : 92, 4));
    }

    [[nodiscard]] bool ReadsReg(Reg reg) const {
        const u32 r = static_cast<u32>(reg);
        return (PredMode() != 0 && Pred() == r) || Src(0, 0) == r || Src(0, 1) == r ||
               Src(1, 0) == r || Src(1, 1) == r;
    }
    [[nodiscard]] bool ReadsOut(OutOp op) const {
        return Method(0) == op || Emit(0) == op || Method(1) == op || Emit(1) == op;
    }

    std::array<u64, 2> bits;
};

bool Compare(AluOp op, u32 x, u32 y) {
    switch (op) {
    case AluOp::Slt:
    case AluOp::Blt:
        return static_cast<s32>(x) < static_cast<s32>(y);
    case AluOp::SltU:
    case AluOp::BltU:
        return x < y;
    case AluOp::Sle:
    case AluOp::Ble:
        return static_cast<s32>(x) <= static_cast<s32>(y);
    case AluOp::SleU:
    case AluOp::BleU:
        return x <= y;
    default:
        return x == y;
    }
}

} // Anonymous namespace

void TuringMacroEngine::SetUploadPointer(u32 instruction) {
    upload_word = instruction * INSTRUCTION_WORDS;
}

void TuringMacroEngine::Upload(u32 word) {
    if (upload_word >= code.size()) {
        LOG_ERROR(HW_GPU, "MME instruction RAM overflow");
        return;
    }
    // Titles upload the same macros again; only a changed word makes the memo and the compiled
    // code stale
    if (code[upload_word] != word) {
        code[upload_word] = word;
        ++code_generation;
    }
    ++upload_word;
}

namespace {
bool MemoEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_MME_MEMO");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

bool JitEnabled() {
#ifdef ARCHITECTURE_x86_64
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_MME_JIT");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
#else
    return false;
#endif
}

bool MemoVerify() {
    static const bool verify = std::getenv("EDEN_MME_MEMO_VERIFY") != nullptr;
    return verify;
}

/// Runs a macro without executing its method writes, for comparisons
class CaptureHost final : public TuringMacroEngine::Host {
public:
    explicit CaptureHost(TuringMacroEngine::Host& host_) : host{host_} {}
    u32 ReadState(u32 method) override {
        return host.ReadState(method);
    }
    void CallMethod(u32 method, u32 value) override {
        writes.emplace_back(method, value);
    }
    void ReadMemory(GPUVAddr address, std::span<u32> data) override {
        host.ReadMemory(address, data);
    }
    void WriteMemory(GPUVAddr, std::span<const u32>) override {}

    TuringMacroEngine::Host& host;
    MacroMemoPairs writes;
};
} // Anonymous namespace

bool TuringMacroEngine::Execute(u32 start, std::span<const u32> parameters, Host& host) {
    // Compiled code costs less than looking a call up in the memo
    if (JitEnabled()) {
        return ExecuteJit(start, parameters, host);
    }
    if (!MemoEnabled()) {
        return Interpret(start, parameters, host);
    }
    memo.Synchronize(code_generation);
    MacroMemo::Effects effects;
    if (memo.Find(start, parameters, data_ram, effects)) {
        if (MemoVerify()) {
            Verify(start, parameters, host, effects);
        }
        for (size_t i = 0; i < effects.data_ram_writes.size(); i += 2) {
            data_ram[effects.data_ram_writes[i]] = effects.data_ram_writes[i + 1];
        }
        for (size_t i = 0; i < effects.writes.size(); i += 2) {
            host.CallMethod(effects.writes[i], effects.writes[i + 1]);
        }
        return true;
    }
    recording = true;
    memo_ok = true;
    data_ram_reads.clear();
    data_ram_writes.clear();
    method_writes.clear();
    const bool result = Interpret(start, parameters, host);
    recording = false;
    if (result && memo_ok) {
        memo.Store(start, parameters, data_ram_reads, data_ram_writes, method_writes);
    }
    return result;
}

void TuringMacroEngine::Verify(u32 start, std::span<const u32> parameters, Host& host,
                               const MacroMemo::Effects& effects) {
    static u64 checked{};
    static u64 differences{};
    const auto saved_data_ram = data_ram;
    CaptureHost capture{host};
    Interpret(start, parameters, capture);
    auto expected_data_ram = saved_data_ram;
    for (size_t i = 0; i < effects.data_ram_writes.size(); i += 2) {
        expected_data_ram[effects.data_ram_writes[i]] = effects.data_ram_writes[i + 1];
    }
    bool same = expected_data_ram == data_ram && capture.writes.size() * 2 == effects.writes.size();
    for (size_t i = 0; same && i < capture.writes.size(); ++i) {
        same = capture.writes[i].first == effects.writes[i * 2] &&
               capture.writes[i].second == effects.writes[i * 2 + 1];
    }
    data_ram = saved_data_ram;
    ++checked;
    if (!same && differences++ < 16) {
        LOG_ERROR(HW_GPU, "MME memo differs from the interpreter: macro {:#x}, {} parameters",
                  start, parameters.size());
    }
    if ((checked & 0xFFFFF) == 0) {
        LOG_WARNING(HW_GPU, "MME memo verified {} calls, {} differences", checked, differences);
    }
}

bool TuringMacroEngine::Interpret(u32 start, std::span<const u32> parameters, Host& host) {
    fifo.assign(parameters.begin(), parameters.end());
    fifo_position = 0;

    std::array<u32, 24> regs{};
    u32 carry = 0;
    u32 method = 0;
    u32 method_increment = 0;
    u32 loop_count = 0;
    u32 loop_start = 0;
    u32 loop_end = 0;
    bool end_next = false;
    u32 pc = start;

    const auto fail = [&](const char* message) {
        LOG_ERROR(HW_GPU, "MME macro at {:#x} failed at {:#x}: {}", start, pc, message);
        return false;
    };
    const auto pop = [&](u32& value) {
        if (fifo_position >= fifo.size()) {
            return false;
        }
        value = fifo[fifo_position++];
        return true;
    };

    for (u32 step = 0; step < MAX_STEPS; ++step) {
        if (pc >= NUM_INSTRUCTIONS) {
            return fail("program counter outside the instruction RAM");
        }
        const Instruction inst{std::span<const u32, 3>{&code[pc * INSTRUCTION_WORDS], 3}};

        // LOAD1 is only read together with LOAD0, and both are taken from the FIFO first
        std::array<u32, 2> load{};
        if (inst.ReadsReg(Reg::Load0) || inst.ReadsOut(OutOp::Load0)) {
            if (!pop(load[0])) {
                return fail("parameter FIFO underflow");
            }
        }
        if (inst.ReadsReg(Reg::Load1) || inst.ReadsOut(OutOp::Load1)) {
            if (!pop(load[1])) {
                return fail("parameter FIFO underflow");
            }
        }

        // Both ALUs read the registers before either writes back
        const std::array<u32, 24> input = regs;
        const auto read = [&](u32 reg, u32 lane) -> u32 {
            if (reg <= static_cast<u32>(Reg::R23)) {
                return input[reg];
            }
            switch (static_cast<Reg>(reg)) {
            case Reg::Zero:
                return 0;
            case Reg::Imm:
                // ALU immediates are sign extended
                return static_cast<u32>(static_cast<s32>(static_cast<s16>(inst.Imm(lane))));
            case Reg::ImmPair:
                return static_cast<u32>(static_cast<s32>(static_cast<s16>(inst.Imm(1 - lane))));
            case Reg::Imm32:
                return (inst.Imm(0) << 16) | inst.Imm(1);
            case Reg::Load0:
                return load[0];
            case Reg::Load1:
                return load[1];
            default:
                return 0;
            }
        };

        // As in Mesa's simulator, a lane runs unless its letter matches the predicate value
        u32 enabled = 0xF;
        if (inst.PredMode() != 0) {
            const char skip = read(inst.Pred(), 0) != 0 ? 'T' : 'F';
            enabled = 0;
            for (u32 i = 0; i < 4; ++i) {
                if (PREDICATE_MODES[inst.PredMode()][i] != skip) {
                    enabled |= 1U << i;
                }
            }
        }

        u32 next_pc = pc + 1;
        bool stop = false;
        bool keep_running = false;
        std::array<u32, 2> results{};
        for (u32 lane = 0; lane < 2; ++lane) {
            if ((enabled & (1U << lane)) == 0) {
                continue;
            }
            const AluOp op = inst.Op(lane);
            const u32 imm = inst.Imm(lane);
            const u32 x = read(inst.Src(lane, 0), lane);
            const u32 y = read(inst.Src(lane, 1), lane);
            u32 result = 0;
            switch (op) {
            case AluOp::Add:
                result = x + y;
                carry = result < x ? 1 : 0;
                break;
            case AluOp::AddC:
                result = x + y + carry;
                break;
            case AluOp::Sub:
                result = x - y;
                carry = result > x ? 1 : 0;
                break;
            case AluOp::SubB:
                result = x - y - carry;
                break;
            case AluOp::Mul:
            case AluOp::MulU: {
                const bool is_signed = op == AluOp::Mul;
                const u64 a = is_signed ? static_cast<u64>(s64{static_cast<s32>(x)}) : x;
                const u64 b = is_signed ? static_cast<u64>(s64{static_cast<s32>(y)}) : y;
                const u64 product = a * b;
                result = static_cast<u32>(product);
                carry = static_cast<u32>(product >> 32);
                break;
            }
            case AluOp::MulH:
                result = carry;
                break;
            case AluOp::Extended:
                memo_ok = false;
                if (x == EXTENDED_READ_FIFO_BARRIER) {
                    ReadFifoBarrier(host);
                } else {
                    LOG_ERROR(HW_GPU, "Unknown MME extended operation {:#x}, {:#x}", x, y);
                }
                break;
            case AluOp::Clz:
                result = static_cast<u32>(std::countl_zero(x));
                break;
            case AluOp::Sll:
                result = x << (y & 31);
                break;
            case AluOp::Srl:
                result = x >> (y & 31);
                break;
            case AluOp::Sra:
                result = static_cast<u32>(static_cast<s32>(x) >> (y & 31));
                break;
            case AluOp::And:
                result = x & y;
                break;
            case AluOp::Nand:
                result = ~(x & y);
                break;
            case AluOp::Or:
                result = x | y;
                break;
            case AluOp::Xor:
                result = x ^ y;
                break;
            case AluOp::Merge: {
                // Inserts bits [src_pos, src_pos + bits) of y at dst_pos of x
                const u32 dst_pos = (imm >> 10) & 0x3F;
                const u32 bits = (imm >> 5) & 0x1F;
                const u32 src_pos = imm & 0x1F;
                const u32 mask = (1U << bits) - 1;
                result = dst_pos >= 32 ? x
                                       : (x & ~(mask << dst_pos)) | (((y >> src_pos) & mask) << dst_pos);
                break;
            }
            case AluOp::Slt:
            case AluOp::SltU:
            case AluOp::Sle:
            case AluOp::SleU:
            case AluOp::Seq:
                result = Compare(op, x, y) ? ~0U : 0U;
                break;
            case AluOp::State:
                result = host.ReadState((x + y) & 0xFFFF);
                memo_ok = false;
                break;
            case AluOp::Loop:
                if (loop_count != 0 || imm < 2) {
                    return fail("invalid loop");
                }
                if (x == 0) {
                    // A loop with no iterations skips its body
                    next_pc = pc + imm;
                    break;
                }
                loop_count = x - 1;
                loop_start = pc;
                loop_end = pc + imm - 1;
                break;
            case AluOp::Jal:
                // Bit 15 is always set by the NVIDIA compiler and has no known meaning
                next_pc = pc + (imm & 0x7FFF);
                break;
            case AluOp::Blt:
            case AluOp::BltU:
            case AluOp::Ble:
            case AluOp::BleU:
            case AluOp::Beq: {
                // Bit 15 is the expected outcome, [0..13) a signed offset; -0x1000 stops
                if (Compare(op, x, y) != ((imm & 0x8000) != 0)) {
                    break;
                }
                const s32 offset = static_cast<s32>(imm << 19) >> 19;
                if (offset == -0x1000) {
                    stop = true;
                    break;
                }
                next_pc = static_cast<u32>(static_cast<s32>(pc) + offset);
                // Seen in NVN macros: a taken branch with bit 14 does not end the macro after
                // an instruction with END_NEXT
                keep_running = (imm & 0x4000) != 0;
                break;
            }
            case AluOp::DRead:
                if (x >= DATA_RAM_WORDS) {
                    return fail("data RAM read out of bounds");
                }
                result = data_ram[x];
                if (recording && std::ranges::none_of(data_ram_writes, [x](const auto& write) {
                        return write.first == x;
                    })) {
                    data_ram_reads.emplace_back(x, result);
                }
                break;
            case AluOp::DWrite:
                if (x >= DATA_RAM_WORDS) {
                    return fail("data RAM write out of bounds");
                }
                data_ram[x] = y;
                if (recording) {
                    data_ram_writes.emplace_back(x, y);
                }
                break;
            }
            results[lane] = result;
            const u32 dst = inst.Dst(lane);
            if (dst <= static_cast<u32>(Reg::R23)) {
                regs[dst] = result;
            }
        }

        const auto output = [&](OutOp op) -> u32 {
            switch (op) {
            case OutOp::Alu0:
                return results[0];
            case OutOp::Alu1:
                return results[1];
            case OutOp::Load0:
                return load[0];
            case OutOp::Load1:
                return load[1];
            case OutOp::Imm0:
                return inst.Imm(0);
            case OutOp::Imm1:
                return inst.Imm(1);
            case OutOp::ImmHigh0:
                return inst.Imm(0) >> 12;
            case OutOp::ImmHigh1:
                return inst.Imm(1) >> 12;
            case OutOp::Imm32:
                return (inst.Imm(0) << 16) | inst.Imm(1);
            default:
                return 0;
            }
        };
        for (u32 lane = 0; lane < 2; ++lane) {
            if ((enabled & (4U << lane)) == 0) {
                continue;
            }
            if (inst.Method(lane) != OutOp::None) {
                const u32 value = output(inst.Method(lane));
                method = value & 0xFFF;
                method_increment = (value >> 12) & 0xF;
            }
            if (inst.Emit(lane) != OutOp::None) {
                if (recording) {
                    method_writes.emplace_back(method, output(inst.Emit(lane)));
                }
                host.CallMethod(method, output(inst.Emit(lane)));
                method += method_increment;
            }
        }

        if (end_next || stop) {
            return true;
        }
        end_next = inst.EndNext() && !keep_running;
        if (loop_count != 0 && pc == loop_end) {
            --loop_count;
            next_pc = loop_start + 1;
        }
        pc = next_pc;
    }
    return fail("instruction limit exceeded");
}

bool TuringMacroEngine::ExecuteJit(u32 start, std::span<const u32> parameters, Host& host) {
#ifdef ARCHITECTURE_x86_64
    fifo.assign(parameters.begin(), parameters.end());
    if (!jit) {
        jit = std::make_shared<TuringMacroJit>();
    }
    if (jit->Prepare(code, code_generation)) {
        static u64 compiles{};
        if ((++compiles & (compiles - 1)) == 0) {
            LOG_WARNING(HW_GPU, "MME macro code compiled {} times", compiles);
        }
    }
    const auto result = jit->Execute(
        start, fifo, data_ram, [&](u32 address) { return host.ReadState(address); },
        [&](u32 method, u32 value) {
            host.CallMethod(method, value);
            return true;
        },
        [&](u32 operation) {
            if (operation == EXTENDED_READ_FIFO_BARRIER) {
                ReadFifoBarrier(host);
            } else {
                LOG_ERROR(HW_GPU, "Unknown MME extended operation {:#x}", operation);
            }
        });
    if (!result.success) {
        LOG_ERROR(HW_GPU, "MME macro at {:#x} failed: {}", start, result.error);
    }
    return result.success;
#else
    return Interpret(start, parameters, host);
#endif
}

void TuringMacroEngine::ReadFifoBarrier(Host& host) {
    if (read_fifo_count == 0) {
        return;
    }
    const size_t offset = fifo.size();
    fifo.resize(offset + read_fifo_count);
    host.ReadMemory(MemoryAddress(), std::span<u32>{fifo.data() + offset, read_fifo_count});
    read_fifo_count = 0;
}

bool TuringMacroEngine::ProcessMethod(u32 method, u32 value, Host& host) {
    // A macro that drives the MME's DMA reads or writes memory while it runs
    switch (method) {
    case SET_MME_MEM_ADDRESS_A:
    case SET_MME_MEM_ADDRESS_B:
    case SET_MME_DATA_RAM_ADDRESS:
    case MME_DMA_READ:
    case MME_DMA_WRITE:
    case MME_DMA_READ_FIFOED:
    case MME_DMA_REDUCTION:
        memo_ok = false;
        break;
    default:
        break;
    }
    switch (method) {
    case SET_MME_MEM_ADDRESS_A:
        mem_address_high = value;
        return true;
    case SET_MME_MEM_ADDRESS_B:
        mem_address_low = value;
        return true;
    case SET_MME_DATA_RAM_ADDRESS:
        data_ram_address = value;
        return true;
    case MME_DMA_READ:
    case MME_DMA_WRITE: {
        if (data_ram_address >= DATA_RAM_WORDS ||
            value > DATA_RAM_WORDS - data_ram_address) {
            LOG_ERROR(HW_GPU, "MME DMA of {} words at {:#x} is outside the data RAM", value,
                      data_ram_address);
            return true;
        }
        const std::span<u32> words{&data_ram[data_ram_address], value};
        if (auto& capture{VideoCommon::DrawCapture::Get()}; capture.Active()) {
            capture.Line(fmt::format("mme dma {} {:#x} words={} data=[{}]",
                                     method == MME_DMA_READ ? "read" : "write", MemoryAddress(),
                                     value,
                                     fmt::join(words.begin(),
                                               words.begin() + std::min<size_t>(value, 6), ",")));
        }
        if (method == MME_DMA_READ) {
            host.ReadMemory(MemoryAddress(), words);
        } else {
            host.WriteMemory(MemoryAddress(), words);
        }
        return true;
    }
    case MME_DMA_READ_FIFOED: {
        const size_t offset = fifo.size();
        fifo.resize(offset + value);
        host.ReadMemory(MemoryAddress(), std::span<u32>{fifo.data() + offset, value});
        read_fifo_count = 0;
        return true;
    }
    case MME_DMA_REDUCTION: {
        // Reduces the data RAM word (or pair) at the data RAM address into memory
        const auto op = static_cast<Engines::SemaphoreReduction>(value & 7);
        const bool is_signed = ((value >> 4) & 3) == 1;
        const bool is_64 = ((value >> 8) & 1) != 0;
        if (data_ram_address + (is_64 ? 1 : 0) >= DATA_RAM_WORDS) {
            LOG_ERROR(HW_GPU, "MME DMA reduction at {:#x} is outside the data RAM",
                      data_ram_address);
            return true;
        }
        std::array<u32, 2> memory{};
        const std::span<u32> target{memory.data(), is_64 ? 2U : 1U};
        host.ReadMemory(MemoryAddress(), target);
        const u64 old_value = (u64{memory[1]} << 32) | memory[0];
        const u64 operand =
            (u64{is_64 ? data_ram[data_ram_address + 1] : 0} << 32) | data_ram[data_ram_address];
        const u64 result = Engines::ReduceSemaphore(op, old_value, operand, is_signed, is_64);
        if (auto& capture{VideoCommon::DrawCapture::Get()}; capture.Active()) {
            capture.Line(fmt::format("mme dma reduction {:#x} op={} {:#x} -> {:#x}",
                                     MemoryAddress(), value & 7, old_value, result));
        }
        memory = {static_cast<u32>(result), static_cast<u32>(result >> 32)};
        host.WriteMemory(MemoryAddress(), target);
        return true;
    }
    case MME_DMA_SYNC:
        // DMA is synchronous here, so the sync value is written at once
        if (data_ram_address < DATA_RAM_WORDS) {
            data_ram[data_ram_address] = value;
        }
        return true;
    case MME_DMA_SYSMEMBAR:
    case SET_MME_DATA_FIFO_CONFIG:
    case MME_DMA_WRITE_METHOD_BARRIER:
        return true;
    default:
        return false;
    }
}

} // namespace Tegra
