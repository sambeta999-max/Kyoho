// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <source_location>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <deque>
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
class MemoryManager;
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
    /// Pages of the lock-free notes below
    static constexpr size_t PageBits = 12;
    static constexpr u64 PageSize = u64{1} << PageBits;
    static constexpr size_t DevicePages = size_t{1} << (36 - PageBits);

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
    /// A method from a command list the engine only stores (not in its execution mask), once
    /// the front classified it: consecutive ones of one subchannel reach the back as one entry.
    /// Outdoors in Ocarina of Time a frame pushed ~200k entries of one method each, and the
    /// entry cost both threads more than most methods.
    void PushStateMethod(Tegra::DmaPusher& pusher, u32 subchannel, u32 method, u32 argument);
    /// A method a macro writes: consecutive ones of one engine reach the back as one entry,
    /// which calls them in order as PushEngineMethod's entries would. Outdoors in Ocarina of
    /// Time macros write ~120k methods a frame (constant buffer data most of them).
    void PushEngineStateMethod(Tegra::Engines::EngineInterface& engine, u32 method,
                               u32 argument);
    /// A method written by a macro, which goes straight to the engine
    void PushEngineMethod(Tegra::Engines::EngineInterface& engine, u32 method, u32 argument);
    /// A function of one pointer, cheaper than a task
    void PushCall(void (*function)(void*), void* argument);
    /// Anything else that must run on the back in order
    void PushTask(std::function<void()>&& task);
    /// Waits until the back has executed everything pushed so far
    void Drain(std::source_location where = std::source_location::current());
    /// Waits until the back has executed everything pushed so far or `done` holds, which it
    /// checks while the back works; true when `done` held first
    template <typename Done>
    bool DrainUnless(Done&& done) {
        FlushStates();
        const u64 head = entry_head.load(std::memory_order_relaxed);
        for (u32 spins = 0; entry_tail.load(std::memory_order_acquire) != head; ++spins) {
            if (done()) {
                return true;
            }
            if (spins < 4000) {
                Pause();
            } else {
                std::this_thread::yield();
            }
        }
        return false;
    }

    /// Entries pushed so far, and executed so far: the back has executed everything before a
    /// position once Executed() reaches it
    [[nodiscard]] u64 Pushed() {
        FlushStates();
        return entry_head.load(std::memory_order_relaxed);
    }
    [[nodiscard]] u64 Executed() const noexcept {
        return entry_tail.load(std::memory_order_acquire);
    }

    /// What the next pushed entry may write to guest memory, set by the front before it pushes
    /// it. Unclassified entries (and every task) are Unknown. From Lichii e691cc3f2
    /// (SplitWriteTracker, src/video_core/split_write_tracker.h), adapted.
    enum class Effect : u8 {
        None,    ///< Only the ranges noted with ClassifyWrite
        Gpu,     ///< Host GPU work (draws): the guest relies on it after a wait for idle only
        Unknown, ///< Anything, at once
    };
    void Classify(Effect effect) noexcept {
        pending_effect = pending_classified ? (std::max)(pending_effect, effect) : effect;
        pending_classified = true;
    }
    /// The next entry writes device memory [begin, end)
    void ClassifyWrite(DAddr begin, DAddr end) noexcept {
        pending_begin = (std::min)(pending_begin, begin);
        pending_end = (std::max)(pending_end, end);
    }
    /// The next entry writes GPU memory [address, +size) of `memory`: an unknown write where it
    /// is unmapped now or too large to track
    void ClassifyGpuWrite(const Tegra::MemoryManager& memory, GPUVAddr address, u64 size);
    /// The next entry waits for idle: the GPU work before it is done once it ran
    void ClassifySync() noexcept {
        pending_sync = true;
    }
    /// The position the back has to reach before a front read of device memory [begin, end)
    /// sees what the guest may rely on: every unknown write, the GPU work before the last wait
    /// for idle and the noted writes to the range (MME DMA reads run ahead of the pipeline on
    /// the console: a game waits for idle before it reads what draws write)
    [[nodiscard]] u64 ReadTarget(DAddr begin, DAddr end);
    /// Parks the back between two entries once it executed everything before `target` (or
    /// everything pushed), as idle as after Drain; the front may then use the rasterizer until
    /// Release. Nothing may be pushed meanwhile.
    void Hold(u64 target);
    void Release();
    /// Waits until the back executed everything before `target`, without stopping it
    void WaitUntil(u64 target) noexcept;

    /// Work that can write command lists the GPU fills in (compute dispatches, DMA copies, inline
    /// uploads) was just pushed
    void NoteWriter() noexcept {
        last_writer = entry_head.load(std::memory_order_relaxed);
    }
    /// Waits until the back has executed the work NoteWriter noted, not what came after it
    void WaitForWriters() noexcept;

    /// Front: asks the back to submit the host GPU work it recorded so far (copies of command
    /// list pages the front waits for), between two entries or at once when it is idle
    void RequestFlush();

    /// Notes device pages the front read command lists from that the GPU wrote: the buffer cache
    /// copies them to host memory after the work that writes them (BufferCache::NoteListWrite)
    void MarkCommandListPage(DAddr address) noexcept;
    [[nodiscard]] bool IsCommandListPage(DAddr address) const noexcept {
        const u64 page = address >> PageBits;
        return page < DevicePages &&
               ((command_list_pages[page / 64].load(std::memory_order_relaxed) >> (page % 64)) &
                1) != 0;
    }
    /// Calls func with the address of each command list page in [address, +size)
    template <typename Func>
    void ForEachCommandListPage(DAddr address, u64 size, Func&& func) const {
        if (size == 0) {
            return;
        }
        const u64 last = (std::min<u64>)((address + size - 1) >> PageBits, DevicePages - 1);
        for (u64 page = address >> PageBits; page <= last;) {
            // Storage buffers can be large: skip 64 pages at a time where none is a list page
            const u64 word =
                command_list_pages[page / 64].load(std::memory_order_relaxed) >> (page % 64);
            if (word == 0) {
                page = NextCommandListWord(page / 64 + 1, last / 64) * 64;
                continue;
            }
            if ((word & 1) != 0) {
                func(static_cast<DAddr>(page << PageBits));
            }
            ++page;
        }
    }
    /// 64 pages of the command list page bits, from page index * 64
    [[nodiscard]] u64 CommandListPageWord(size_t index) const noexcept {
        return command_list_pages[index].load(std::memory_order_relaxed);
    }
    /// The first word index in [index, last_word] that may hold a command list page, or one past
    /// last_word: one summary bit per word, set with it and never cleared
    [[nodiscard]] u64 NextCommandListWord(u64 index, u64 last_word) const noexcept {
        last_word = (std::min<u64>)(last_word, DevicePages / 64 - 1);
        while (index <= last_word) {
            const u64 summary =
                command_list_summary[index / 64].load(std::memory_order_relaxed) >> (index % 64);
            if (summary != 0) {
                return index + std::countr_zero(summary);
            }
            index = (index / 64 + 1) * 64;
        }
        return last_word + 1;
    }
    /// Whether any page was marked: the buffer cache skips its checks until then
    [[nodiscard]] bool HasCommandListPages() const noexcept {
        return has_command_list_pages.load(std::memory_order_relaxed);
    }
    /// The active split once its front marked a command list page
    [[nodiscard]] static GpuSplit* WithCommandListPages() noexcept {
        GpuSplit* const split = active;
        return split != nullptr && split->HasCommandListPages() ? split : nullptr;
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
    static void Pause() noexcept;

    enum class Kind : u8 { Method, MultiMethod, EngineMethod, Call, Task, States, EngineStates };
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
    /// Pushes the stores PushStateMethod collected, with their own classification (the
    /// pending one is for the entry the front prepares)
    void FlushStates() {
        if (!state_words.empty()) {
            PushStates();
        }
    }
    void PushStates();
    void AppendState(Kind kind, void* target, u32 subchannel, u32 method, u32 argument);
    static constexpr size_t MaxStateWords = 2048;
    std::vector<u32> state_words; ///< (method, argument) pairs
    Kind state_kind{};
    void* state_target{};
    /// PushCall folded into the macro's methods: called on the target after them
    void (*state_call)(void*){};
    u32 state_subchannel{};
    Effect state_effect{};
    void SetGpuWritten(DAddr address, u64 size) noexcept;
    /// Back: submits the recorded host GPU work when the front asked for it
    void BackFlush();
    std::atomic<bool> flush_requested{};

    /// A bit per device page the GPU wrote, so the front checks command lists without the
    /// buffer cache's lock
    std::unique_ptr<std::atomic<u64>[]> gpu_written;
    /// A bit per device page of the GPU-written command lists the front read (same layout)
    std::unique_ptr<std::atomic<u64>[]> command_list_pages;
    /// Bit per word of command_list_pages: set once the word has a page (NextCommandListWord)
    std::unique_ptr<std::atomic<u64>[]> command_list_summary;
    std::atomic<bool> has_command_list_pages{};
    u64 ReserveData(u32 amount, u64& offset);
    void Run(std::stop_token stop_token);
    void Execute(Entry& entry);

    static inline GpuSplit* active{};
    static inline GpuSplit* created{};

    Core::System& system;
    std::vector<Entry> entries;
    std::vector<u32> data;
    std::vector<std::function<void()>> tasks;

    // The front pushes ~350k entries a frame outdoors in Ocarina of Time. Each line below is
    // written by one thread only, and the front reads the back's positions only when its copy
    // says a ring is full: reading entry_tail and back_sleeping at every push, from the line the
    // back writes after every entry, moved that line between the two cores for each entry.
    alignas(64) std::atomic<u64> entry_head{}; ///< Written by the front
    alignas(64) u64 data_head{};
    u64 seen_entry_tail{}; ///< Front: entry_tail, data_tail and task_tail as last read
    u64 seen_data_tail{};
    u64 seen_task_tail{};
    u64 task_head{};
    u64 last_writer{}; ///< Entries pushed when the last writer was (NoteWriter)
    struct NotedWrite {
        DAddr begin;
        DAddr end;
        u64 position; ///< Executed once the back passed it
    };
    static constexpr size_t MaxNotedWrites = 512;
    std::deque<NotedWrite> noted_writes;
    u64 last_unknown{}; ///< Positions just past the last entry of each kind
    u64 last_gpu{};
    u64 last_sync{};
    Effect pending_effect{};
    bool pending_classified{};
    DAddr pending_begin{~DAddr{0}};
    DAddr pending_end{};
    bool pending_sync{};
    void ConsumeClassification(u64 position);
    alignas(64) std::atomic<u64> hold_at{~u64{0}}; ///< Written by the front
    std::atomic<u64> parked{}; ///< Back: one past the entry it waits before (Hold), or 0
    alignas(64) std::atomic<u64> entry_tail{}; ///< Written by the back
    std::atomic<u64> data_tail{};
    std::atomic<u64> task_tail{};
    alignas(64) std::atomic<bool> back_sleeping{}; ///< Read by the front at every push
    std::atomic<bool> stopping{};

    alignas(64) std::jthread thread;
    std::thread::id back_id{};
};

} // namespace VideoCommon
