// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstdlib>
#include <algorithm>
#include <cstring>
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#endif

#include <exception>

#include "common/assert.h"
#include "common/logging.h"
#include "common/thread.h"
#include "core/core.h"
#include "video_core/dma_pusher.h"
#include "video_core/engines/engine_interface.h"
#include "video_core/gpu_split.h"

namespace VideoCommon {

namespace {
/// Spins before sleeping, the back usually has more work within microseconds
constexpr int SpinIterations = 4000;

void Relax(int& spins) {
    if (++spins < SpinIterations) {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
        _mm_pause();
#endif
    } else {
        std::this_thread::yield();
    }
}
} // Anonymous namespace

GpuSplit::GpuSplit(Core::System& system_) : system{system_} {
    entries.resize(EntryCapacity);
    data.resize(DataCapacity);
    tasks.resize(TaskCapacity);
    gpu_written = std::make_unique<std::atomic<u64>[]>(DevicePages / 64);
}

void GpuSplit::SetGpuWritten(DAddr address, u64 size) noexcept {
    if (size == 0) {
        return;
    }
    const u64 first = address >> PageBits;
    const u64 last = std::min<u64>((address + size - 1) >> PageBits, DevicePages - 1);
    for (u64 page = first; page <= last; ++page) {
        auto& word = gpu_written[page / 64];
        const u64 bit = u64{1} << (page % 64);
        if ((word.load(std::memory_order_relaxed) & bit) == 0) {
            word.fetch_or(bit, std::memory_order_release);
        }
    }
}

bool GpuSplit::TakeGpuWritten(DAddr address, u64 size) noexcept {
    if (size == 0) {
        return false;
    }
    bool written = false;
    const u64 first = address >> PageBits;
    const u64 last = std::min<u64>((address + size - 1) >> PageBits, DevicePages - 1);
    for (u64 page = first; page <= last; ++page) {
        auto& word = gpu_written[page / 64];
        const u64 bit = u64{1} << (page % 64);
        if ((word.load(std::memory_order_acquire) & bit) != 0) {
            word.fetch_and(~bit, std::memory_order_acq_rel);
            written = true;
        }
    }
    return written;
}

GpuSplit::~GpuSplit() {
    Stop();
}

bool GpuSplit::Requested() {
    static const bool requested = [] {
        const char* const value = std::getenv("EDEN_GPU_SPLIT");
        return value != nullptr && value[0] == '1';
    }();
    return requested;
}

void GpuSplit::Start() {
    thread = std::jthread([this](std::stop_token stop_token) { Run(stop_token); });
    while (back_id == std::thread::id{}) {
        std::this_thread::yield();
    }
    created = this;
}

void GpuSplit::Activate() {
    if (active == nullptr) {
        LOG_INFO(HW_GPU, "The GPU thread is split in two (EDEN_GPU_SPLIT=1)");
        active = this;
    }
}

void GpuSplit::Stop() {
    if (!thread.joinable()) {
        return;
    }
    active = nullptr;
    created = nullptr;
    stopping.store(true);
    entry_head.notify_one();
    thread.request_stop();
    thread.join();
}

void GpuSplit::Push(const Entry& entry) {
    const u64 head = entry_head.load(std::memory_order_relaxed);
    int spins = 0;
    while (head - entry_tail.load(std::memory_order_acquire) >= EntryCapacity) {
        Relax(spins);
    }
    entries[head % EntryCapacity] = entry;
    entry_head.store(head + 1, std::memory_order_seq_cst);
    if (back_sleeping.load(std::memory_order_seq_cst)) {
        entry_head.notify_one();
    }
}

u64 GpuSplit::ReserveData(u32 amount, u64& offset) {
    ASSERT(amount <= DataCapacity / 2);
    u64 head = data_head;
    if (const u64 position = head % DataCapacity; position + amount > DataCapacity) {
        // Keep the words of one entry contiguous
        head += DataCapacity - position;
    }
    int spins = 0;
    while (head + amount - data_tail.load(std::memory_order_acquire) > DataCapacity) {
        Relax(spins);
    }
    offset = head % DataCapacity;
    data_head = head + amount;
    return data_head;
}

void GpuSplit::PushMethod(Tegra::DmaPusher& pusher, u32 subchannel, u32 method, u32 argument,
                          bool is_last_call, GPUVAddr segment) {
    Push(Entry{
        .kind = Kind::Method,
        .subchannel = static_cast<u8>(subchannel),
        .is_last_call = is_last_call,
        .method = method,
        .argument = argument,
        .target = &pusher,
        .segment = segment,
        .data_end = data_head,
    });
}

void GpuSplit::PushMultiMethod(Tegra::DmaPusher& pusher, u32 subchannel, u32 method,
                               const u32* words, u32 amount, u32 methods_pending,
                               GPUVAddr segment) {
    u64 offset{};
    const u64 data_end = ReserveData(amount, offset);
    std::memcpy(data.data() + offset, words, amount * sizeof(u32));
    Push(Entry{
        .kind = Kind::MultiMethod,
        .subchannel = static_cast<u8>(subchannel),
        .method = method,
        .argument = static_cast<u32>(offset),
        .amount = amount,
        .methods_pending = methods_pending,
        .target = &pusher,
        .segment = segment,
        .data_end = data_end,
    });
}

void GpuSplit::PushEngineMethod(Tegra::Engines::EngineInterface& engine, u32 method,
                                u32 argument) {
    Push(Entry{
        .kind = Kind::EngineMethod,
        .method = method,
        .argument = argument,
        .target = &engine,
        .data_end = data_head,
    });
}

void GpuSplit::PushCall(void (*function)(void*), void* argument) {
    Push(Entry{
        .kind = Kind::Call,
        .target = argument,
        .segment = reinterpret_cast<GPUVAddr>(function),
        .data_end = data_head,
    });
}

void GpuSplit::PushTask(std::function<void()>&& task) {
    int spins = 0;
    while (task_head - task_tail.load(std::memory_order_acquire) >= TaskCapacity) {
        Relax(spins);
    }
    tasks[task_head % TaskCapacity] = std::move(task);
    ++task_head;
    Push(Entry{.kind = Kind::Task, .data_end = data_head});
}

void GpuSplit::Drain() {
    const u64 head = entry_head.load(std::memory_order_relaxed);
    int spins = 0;
    while (entry_tail.load(std::memory_order_acquire) != head) {
        Relax(spins);
    }
}

void GpuSplit::Execute(Entry& entry) {
    switch (entry.kind) {
    case Kind::Method:
        static_cast<Tegra::DmaPusher*>(entry.target)
            ->BackCallMethod(entry.subchannel, entry.method, entry.argument, entry.is_last_call,
                             entry.segment);
        break;
    case Kind::MultiMethod:
        static_cast<Tegra::DmaPusher*>(entry.target)
            ->BackCallMultiMethod(entry.subchannel, entry.method, data.data() + entry.argument,
                                  entry.amount, entry.methods_pending, entry.segment);
        break;
    case Kind::EngineMethod: {
        auto* const engine = static_cast<Tegra::Engines::EngineInterface*>(entry.target);
        engine->ConsumeSink(system);
        engine->CallMethod(system, entry.method, entry.argument, true);
        break;
    }
    case Kind::Call:
        reinterpret_cast<void (*)(void*)>(entry.segment)(entry.target);
        break;
    case Kind::Task: {
        const u64 index = task_tail.load(std::memory_order_relaxed);
        auto& task = tasks[index % TaskCapacity];
        task();
        task = nullptr;
        task_tail.store(index + 1, std::memory_order_release);
        break;
    }
    }
}

void GpuSplit::Run(std::stop_token stop_token) {
    Common::SetCurrentThreadName("GPUBack");
    Common::SetCurrentThreadPriority(Common::ThreadPriority::Critical);
    Common::SetCurrentThreadToPerformanceCores();
    system.RegisterHostThread();
    back_id = std::this_thread::get_id();
    u64 tail = 0;
    while (!stopping.load(std::memory_order_relaxed)) {
        u64 head = entry_head.load(std::memory_order_acquire);
        if (head == tail) {
            int spins = 0;
            while (head == tail && spins < SpinIterations) {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
                _mm_pause();
#endif
                ++spins;
                head = entry_head.load(std::memory_order_acquire);
            }
            if (head == tail) {
                back_sleeping.store(true, std::memory_order_seq_cst);
                head = entry_head.load(std::memory_order_seq_cst);
                if (head == tail && !stopping.load()) {
                    entry_head.wait(tail, std::memory_order_seq_cst);
                }
                back_sleeping.store(false, std::memory_order_relaxed);
                continue;
            }
        }
        for (; tail != head; ++tail) {
            Entry& entry = entries[tail % EntryCapacity];
            try {
                Execute(entry);
            } catch (const std::exception& exception) {
                LOG_CRITICAL(HW_GPU, "GPU back thread: {} (entry kind {}, method {:#x})",
                             exception.what(), static_cast<int>(entry.kind), entry.method);
            } catch (...) {
                LOG_CRITICAL(HW_GPU, "GPU back thread: unknown exception (entry kind {})",
                             static_cast<int>(entry.kind));
            }
            data_tail.store(entry.data_end, std::memory_order_release);
            entry_tail.store(tail + 1, std::memory_order_release);
        }
    }
}

} // namespace VideoCommon
