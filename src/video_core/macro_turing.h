// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <memory>
#include <span>
#include <vector>

#include "common/common_types.h"

namespace Tegra {

/// Macro engine of the Turing and later 3D classes (TU104 MME). Instructions are 96 bits: a
/// predicate, two ALUs with a 16-bit immediate each and two method outputs. The encoding and the
/// semantics follow Mesa src/nouveau/mme (mme_tu104.c, mme_tu104_sim.c; MIT, Collabora Ltd.).
class TuringMacroEngine {
public:
    static constexpr u32 NUM_INSTRUCTIONS = 0x1000;
    static constexpr u32 INSTRUCTION_WORDS = 3;
    static constexpr u32 DATA_RAM_WORDS = 0xC00;
    /// Bound on executed instructions, so that a broken macro cannot hang the GPU thread
    static constexpr u32 MAX_STEPS = 1U << 22;

    /// The engine the macro runs on
    class Host {
    public:
        virtual ~Host() = default;
        /// Register value, by method index
        virtual u32 ReadState(u32 method) = 0;
        /// Method call emitted by the macro, by method index
        virtual void CallMethod(u32 method, u32 value) = 0;
        virtual void ReadMemory(GPUVAddr address, std::span<u32> data) = 0;
        virtual void WriteMemory(GPUVAddr address, std::span<const u32> data) = 0;
    };

    /// LOAD_MME_INSTRUCTION_RAM_POINTER, in instructions
    void SetUploadPointer(u32 instruction);
    /// LOAD_MME_INSTRUCTION_RAM
    void Upload(u32 word);

    /// Runs the macro starting at an instruction; false if it failed
    bool Execute(u32 start, std::span<const u32> parameters, Host& host);

    /// Handles the MME DMA methods of the 3D class; false for any other method
    bool ProcessMethod(u32 method, u32 value, Host& host);

    [[nodiscard]] std::span<const u32> DataRam() const {
        return data_ram;
    }

private:
    /// Method of the EXTENDED operation that waits for MME_DMA_READ_FIFOED
    static constexpr u32 EXTENDED_READ_FIFO_BARRIER = 0x1000;

    /// Loads the parameter FIFO with the words requested by MME_DMA_READ_FIFOED
    void ReadFifoBarrier(Host& host);

    [[nodiscard]] GPUVAddr MemoryAddress() const {
        return (GPUVAddr{mem_address_high & 0xFF} << 32) | mem_address_low;
    }

    std::array<u32, NUM_INSTRUCTIONS * INSTRUCTION_WORDS> code{};
    u32 upload_word{};

    std::array<u32, DATA_RAM_WORDS> data_ram{};
    u32 data_ram_address{};
    u32 mem_address_high{};
    u32 mem_address_low{};
    u32 read_fifo_count{};

    /// Parameters of the running macro; MME_DMA_READ_FIFOED replaces them
    std::vector<u32> fifo;
    size_t fifo_position{};
};

/// Macros shared by the channels of a TSG: NVN uploads them once for all of its channels. The 3D
/// and compute classes share one instruction RAM and table of macro start positions: Welcome Tour
/// calls on its compute subchannel the macros it uploaded through the 3D class.
struct NativeMacroContext {
    TuringMacroEngine engine;
    std::array<u32, 0x80> positions{};
};

} // namespace Tegra
