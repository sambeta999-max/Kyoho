// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <source_location>

#include <array>
#include <atomic>
#include <functional>
#include <memory>
#include <thread>
#include <vector>

#include "common/common_types.h"

namespace Core {
class System;
}

namespace Tegra {
class DmaPusher;
namespace Engines {
class EngineInterface;
}
} // namespace Tegra

namespace VideoCommon {

/// Splits the GPU thread in two for Switch 2 titles (Vulkan only; EDEN_GPU_SPLIT=0 keeps one
/// thread). The back starts with the GPU and the split takes effect at the first submission of
/// a native-ABI channel, on the GPU thread between two command lists. The front (the GPU thread)
/// parses command lists and runs the 3D macros of AMPERE_B channels against a copy of the 3D
/// registers; the back executes the method writes this produces in order: engine state, draws,
/// dispatches, queries, fences and everything else that reaches the rasterizer. Only the back
/// uses the rasterizer while it runs; the front reaches it only after Drain(), when the back is
/// idle (semaphore acquires, MME DMA reads, channel binding).
class GpuSplit {
public:
    explicit GpuSplit(Core::System& system);
    ~GpuSplit();

    /// The split of this process, or null when it is off
    [[nodiscard]] static GpuSplit* Active() noexcept {
        return active;
    }
    /// On unless EDEN_GPU_SPLIT=0
    [[nodiscard]] static bool Requested();

    void Start();
    void Stop();

    /// The split created at GPU start, active or not
    [[nodiscard]] static GpuSplit* Created() noexcept {
        return created;
    }
    /// Routes the GPU thread's work through the back from now on; on the GPU thread only
    void Activate();

    /// A method from a command list, executed as DmaPusher would on one thread
    void PushMethod(Tegra::DmaPusher& pusher, u32 subchannel, u32 method, u32 argument,
                    bool is_last_call, GPUVAddr segment);
    /// Several data words of one method; the data is copied
    void PushMultiMethod(Tegra::DmaPusher& pusher, u32 subchannel, u32 method, const u32* data,
                         u32 amount, u32 methods_pending, GPUVAddr segment);
    /// A method written by a macro, which goes straight to the engine
    void PushEngineMethod(Tegra::Engines::EngineInterface& engine, u32 method, u32 argument);
    /// A function of one pointer, cheaper than a task
    void PushCall(void (*function)(void*), void* argument);
    /// Anything else that must run on the back in order
    void PushTask(std::function<void()>&& task);
    /// Waits until the back has executed everything pushed so far
    void Drain(std::source_location where = std::source_location::current());

    /// Entries pushed so far, and executed so far: the back has executed everything before a
    /// position once Executed() reaches it
    [[nodiscard]] u64 Pushed() const noexcept {
        return entry_head.load(std::memory_order_relaxed);
    }
    [[nodiscard]] u64 Executed() const noexcept {
        return entry_tail.load(std::memory_order_acquire);
    }

    /// Notes device memory the GPU wrote, from any thread; nothing when the split is off
    static void MarkGpuWritten(DAddr address, u64 size) noexcept {
        if (GpuSplit* const split = active) {
            split->SetGpuWritten(address, size);
        }
    }
    /// Whether the GPU ever wrote a page of [address, +size). The note is never cleared: the
    /// work that rewrites such a page (a culling dispatch filling the next frame's command list)
    /// may still be queued on the back when the front reads it, so it is not marked again yet
    [[nodiscard]] bool WasGpuWritten(DAddr address, u64 size) const noexcept;

    [[nodiscard]] bool OnBackThread() const noexcept {
        return std::this_thread::get_id() == back_id;
    }

private:
    enum class Kind : u8 { Method, MultiMethod, EngineMethod, Call, Task };
    struct Entry {
        Kind kind;
        u8 subchannel;
        bool is_last_call;
        u32 method;
        u32 argument;
        u32 amount;
        u32 methods_pending;
        void* target;
        GPUVAddr segment;
        u64 data_end; ///< Data ring position after this entry's words
    };
    static constexpr size_t EntryCapacity = 1 << 16;
    static constexpr size_t DataCapacity = 1 << 22;
    static constexpr size_t TaskCapacity = 1 << 12;

    void Push(const Entry& entry);
    void SetGpuWritten(DAddr address, u64 size) noexcept;

    static constexpr size_t PageBits = 12;
    static constexpr size_t DevicePages = size_t{1} << (36 - PageBits);
    /// A bit per device page the GPU wrote, so the front checks command lists without the
    /// buffer cache's lock
    std::unique_ptr<std::atomic<u64>[]> gpu_written;
    u64 ReserveData(u32 amount, u64& offset);
    void Run(std::stop_token stop_token);
    void Execute(Entry& entry);

    static inline GpuSplit* active{};
    static inline GpuSplit* created{};

    Core::System& system;
    std::vector<Entry> entries;
    std::vector<u32> data;
    std::vector<std::function<void()>> tasks;

    alignas(64) std::atomic<u64> entry_head{}; ///< Written by the front
    u64 data_head{};
    u64 task_head{};
    alignas(64) std::atomic<u64> entry_tail{}; ///< Written by the back
    std::atomic<u64> data_tail{};
    std::atomic<u64> task_tail{};
    std::atomic<bool> back_sleeping{};
    std::atomic<bool> stopping{};

    std::jthread thread;
    std::thread::id back_id{};
};

} // namespace VideoCommon
