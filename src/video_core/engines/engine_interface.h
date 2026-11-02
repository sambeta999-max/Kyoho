// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <bitset>
#include <limits>
#include <vector>

#include "common/common_types.h"

namespace Core {
class System;
}

namespace Tegra::Engines {

enum class EngineTypes : u32 {
    Nv01Timer,
    KeplerCompute,
    Maxwell3D,
    Fermi2D,
    MaxwellDMA,
    KeplerMemory,
};

class EngineInterface {
public:
    virtual ~EngineInterface() = default;

    /// Write the value to the register identified by method.
    virtual void CallMethod(Core::System& system, u32 method, u32 method_argument, bool is_last_call) = 0;

    /// Write multiple values to the register identified by method.
    virtual void CallMultiMethod(Core::System& system, u32 method, const u32* base_start, u32 amount, u32 methods_pending) = 0;

    /// Turing and later MMEs read their parameter FIFO as data arrives, so one macro call can
    /// span several method headers. An engine that defers such a call until its data ends
    /// reports it here, and the pusher flushes it before any method that is not more data.
    virtual bool HasPendingMacro() const {
        return false;
    }
    virtual bool IsMacroData(u32 method) const {
        return false;
    }
    virtual void FlushPendingMacro(Core::System& system) {}
    /// With a split GPU thread (gpu_split.h) the front runs the macros of AMPERE_B and
    /// AMPERE_COMPUTE_B, which share the MME of the channel, against its own copy of the
    /// registers; the back executes the method writes
    virtual bool SupportsFrontMacros() const {
        return false;
    }
    virtual void EnableFrontMacros() {}
    virtual bool MacrosOnFront() const {
        return false;
    }
    /// Macro calls, macro uploads and MME DMA methods, which the front handles itself
    virtual bool IsFrontMethod(u32 method) const {
        return false;
    }
    virtual void FrontCallMethod(Core::System& system, u32 method, u32 argument) {}
    virtual void FrontCallMultiMethod(Core::System& system, u32 method, const u32* base_start,
                                      u32 amount) {}
    /// Applies a method write that the back executes to the front's copy of the registers
    virtual void FrontWrite(u32 method, u32 argument) {}
    virtual void FrontMultiWrite(u32 method, const u32* base_start, u32 amount) {}

    void ConsumeSink(Core::System& system) {
        if (!method_sink.empty()) {
            ConsumeSinkImpl(system);
        }
    }

    std::bitset<(std::numeric_limits<u16>::max)()> execution_mask{};
    std::vector<std::pair<u32, u32>> method_sink{};
    GPUVAddr current_dma_segment;
    bool current_dirty{};
protected:
    virtual void ConsumeSinkImpl(Core::System& system) {
        for (auto [method, value] : method_sink) {
            CallMethod(system, method, value, true);
        }
        method_sink.clear();
    }
};

} // namespace Tegra::Engines
