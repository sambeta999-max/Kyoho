// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstdlib>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <string_view>
#include <vector>
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
#include "video_core/gpu.h"
#include "video_core/gpu_split.h"
#include "video_core/memory_manager.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/renderer_base.h"

namespace VideoCommon {

namespace {
/// Spins before sleeping, the back usually has more work within microseconds
constexpr int SpinIterations = 4000;

/// Marks the front as waiting for the back while it spins: the back submits deferred work at once
struct FrontWaiting {
    explicit FrontWaiting(std::atomic<bool>& flag_) : flag{flag_} {
        flag.store(true, std::memory_order_relaxed);
    }
    ~FrontWaiting() {
        flag.store(false, std::memory_order_relaxed);
    }
    std::atomic<bool>& flag;
};

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

void GpuSplit::Pause() noexcept {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    _mm_pause();
#endif
}

GpuSplit::GpuSplit(Core::System& system_) : system{system_} {
    entries.resize(EntryCapacity);
    data.resize(DataCapacity);
    tasks.resize(TaskCapacity);
    gpu_written = std::make_unique<std::atomic<u64>[]>(DevicePages / 64);
    command_list_pages = std::make_unique<std::atomic<u64>[]>(DevicePages / 64);
    command_list_summary = std::make_unique<std::atomic<u64>[]>(DevicePages / 64 / 64);
}

void GpuSplit::MarkCommandListPage(DAddr address) noexcept {
    const u64 page = address >> PageBits;
    if (page >= DevicePages) {
        return;
    }
    auto& word = command_list_pages[page / 64];
    const u64 bit = u64{1} << (page % 64);
    if ((word.load(std::memory_order_relaxed) & bit) == 0) {
        word.fetch_or(bit, std::memory_order_relaxed);
        const u64 index = page / 64;
        command_list_summary[index / 64].fetch_or(u64{1} << (index % 64),
                                                  std::memory_order_relaxed);
        has_command_list_pages.store(true, std::memory_order_relaxed);
    }
}

void GpuSplit::RequestFlush() {
    flush_requested.store(true, std::memory_order_seq_cst);
    if (Executed() == Pushed()) {
        // The back may be asleep
        PushCall([](void* split) { static_cast<GpuSplit*>(split)->BackFlush(); }, this);
    }
}

void GpuSplit::BackFlush() {
    if (flush_requested.exchange(false, std::memory_order_acq_rel)) {
        system.GPU().Renderer().ReadRasterizer()->FlushListCopies();
    }
}

void GpuSplit::SubmitWhenIdle(u64 next) {
    // Submissions deferred by the scheduler go with the next one. When the front sends nothing
    // more within a short spin, or waits for the back, the back submits them before it reports
    // its last entry done: the fencing thread waits for them, and once the back is idle the
    // front may use the scheduler itself (Drain), so the back may not submit after that
    int spins = 0;
    while (entry_head.load(std::memory_order_acquire) == next) {
        if (++spins >= SpinIterations || front_waiting.load(std::memory_order_relaxed) ||
            flush_requested.load(std::memory_order_relaxed) ||
            hold_at.load(std::memory_order_relaxed) <= next ||
            stopping.load(std::memory_order_relaxed)) {
            BackFlush();
            deferred_submit.store(false, std::memory_order_relaxed);
            system.GPU().Renderer().ReadRasterizer()->SubmitDeferred();
            return;
        }
        Pause();
    }
}

void GpuSplit::WaitForWriters() noexcept {
    // The back publishes its position after every entry
    FrontWaiting waiting{front_waiting};
    int spins = 0;
    while (entry_tail.load(std::memory_order_acquire) < last_writer) {
        Relax(spins);
    }
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

bool GpuSplit::WasGpuWritten(DAddr address, u64 size) const noexcept {
    if (size == 0) {
        return false;
    }
    const u64 first = address >> PageBits;
    const u64 last = std::min<u64>((address + size - 1) >> PageBits, DevicePages - 1);
    for (u64 page = first; page <= last; ++page) {
        const u64 bit = u64{1} << (page % 64);
        if ((gpu_written[page / 64].load(std::memory_order_acquire) & bit) != 0) {
            return true;
        }
    }
    return false;
}

GpuSplit::~GpuSplit() {
    Stop();
}

bool GpuSplit::Requested() {
    static const bool requested = [] {
        const char* const value = std::getenv("EDEN_GPU_SPLIT");
        return value == nullptr || value[0] != '0';
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
        LOG_INFO(HW_GPU, "The GPU thread is split in two (EDEN_GPU_SPLIT=0 keeps one thread)");
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

void GpuSplit::PushStateMethod(Tegra::DmaPusher& pusher, u32 subchannel, u32 method,
                               u32 argument) {
    if (pending_begin < pending_end || pending_sync) [[unlikely]] {
        // Noted writes and waits for idle keep an entry of their own
        PushMethod(pusher, subchannel, method, argument, true, 0);
        return;
    }
    AppendState(Kind::States, &pusher, subchannel, method, argument);
}

void GpuSplit::PushEngineStateMethod(Tegra::Engines::EngineInterface& engine, u32 method,
                                     u32 argument) {
    if (pending_begin < pending_end || pending_sync) [[unlikely]] {
        PushEngineMethod(engine, method, argument);
        return;
    }
    AppendState(Kind::EngineStates, &engine, 0, method, argument);
}

void GpuSplit::AppendState(Kind kind, void* target, u32 subchannel, u32 method, u32 argument) {
    if (!state_words.empty() && (state_kind != kind || state_target != target ||
                                 state_subchannel != subchannel ||
                                 state_words.size() >= MaxStateWords)) {
        PushStates();
    }
    const Effect effect = pending_classified ? pending_effect : Effect::Unknown;
    state_effect = state_words.empty() ? effect : (std::max)(state_effect, effect);
    pending_classified = false;
    state_kind = kind;
    state_target = target;
    state_subchannel = subchannel;
    state_words.push_back(method);
    state_words.push_back(argument);
}

void GpuSplit::PushStates() {
    const Effect effect = pending_effect;
    const bool classified = pending_classified;
    const DAddr begin = pending_begin;
    const DAddr end = pending_end;
    const bool sync = pending_sync;
    pending_effect = state_effect;
    pending_classified = true;
    pending_begin = ~DAddr{0};
    pending_end = 0;
    pending_sync = false;
    const u32 words = static_cast<u32>(state_words.size());
    u64 offset{};
    const u64 data_end = ReserveData(words, offset);
    std::memcpy(data.data() + offset, state_words.data(), words * sizeof(u32));
    state_words.clear();
    const auto call = state_call;
    state_call = nullptr;
    Push(Entry{
        .kind = state_kind,
        .subchannel = static_cast<u8>(state_subchannel),
        .argument = static_cast<u32>(offset),
        .amount = words / 2,
        .target = state_target,
        .segment = reinterpret_cast<GPUVAddr>(call),
        .data_end = data_end,
    });
    pending_effect = effect;
    pending_classified = classified;
    pending_begin = begin;
    pending_end = end;
    pending_sync = sync;
}

void GpuSplit::Push(const Entry& entry) {
    FlushStates();
    const u64 head = entry_head.load(std::memory_order_relaxed);
    int spins = 0;
    if (head - seen_entry_tail >= EntryCapacity) {
        while (head - (seen_entry_tail = entry_tail.load(std::memory_order_acquire)) >=
               EntryCapacity) {
            Relax(spins);
        }
    }
    entries[head % EntryCapacity] = entry;
    ConsumeClassification(head + 1);
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
    while (head + amount - seen_data_tail > DataCapacity) {
        seen_data_tail = data_tail.load(std::memory_order_acquire);
        if (head + amount - seen_data_tail > DataCapacity) {
            Relax(spins);
        }
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
    // The stores before go first in the data ring too
    FlushStates();
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
    // Inline uploads of the 3D engine (launch_dma, inline_data) a macro writes
    if (method == 0x6C || method == 0x6D) {
        NoteWriter();
    }
}

void GpuSplit::PushCall(void (*function)(void*), void* argument) {
    if (!state_words.empty() && state_kind == Kind::EngineStates && state_target == argument &&
        pending_begin >= pending_end && !pending_sync) {
        // The draw a macro ends with goes in the entry of the methods it wrote
        const Effect effect = pending_classified ? pending_effect : Effect::Unknown;
        state_effect = (std::max)(state_effect, effect);
        pending_classified = false;
        state_call = function;
        PushStates();
        return;
    }
    Push(Entry{
        .kind = Kind::Call,
        .target = argument,
        .segment = reinterpret_cast<GPUVAddr>(function),
        .data_end = data_head,
    });
}

void GpuSplit::PushTask(std::function<void()>&& task) {
    FlushStates();
    int spins = 0;
    while (task_head - seen_task_tail >= TaskCapacity) {
        seen_task_tail = task_tail.load(std::memory_order_acquire);
        if (task_head - seen_task_tail >= TaskCapacity) {
            Relax(spins);
        }
    }
    tasks[task_head % TaskCapacity] = std::move(task);
    ++task_head;
    Push(Entry{.kind = Kind::Task, .data_end = data_head});
}

void GpuSplit::ConsumeClassification(u64 position) {
    const Effect effect = pending_classified ? pending_effect : Effect::Unknown;
    if (effect == Effect::Unknown) {
        last_unknown = position;
    } else if (effect == Effect::Gpu) {
        last_gpu = position;
    }
    if (pending_begin < pending_end) {
        if (noted_writes.size() >= MaxNotedWrites) {
            last_unknown = position;
            noted_writes.clear();
        } else if (!noted_writes.empty() && pending_begin <= noted_writes.back().end &&
                   noted_writes.back().begin <= pending_end) {
            // Joined with the previous write, it retires with this one (conservative)
            NotedWrite& last = noted_writes.back();
            last.begin = (std::min)(last.begin, pending_begin);
            last.end = (std::max)(last.end, pending_end);
            last.position = position;
        } else {
            noted_writes.push_back({pending_begin, pending_end, position});
        }
    }
    if (pending_sync) {
        last_sync = position;
    }
    pending_classified = false;
    pending_begin = ~DAddr{0};
    pending_end = 0;
    pending_sync = false;
}

void GpuSplit::ClassifyGpuWrite(const Tegra::MemoryManager& memory, GPUVAddr address, u64 size) {
    Classify(Effect::None);
    if (size > 0x10000) {
        Classify(Effect::Unknown);
        return;
    }
    constexpr GPUVAddr page_mask = PageSize - 1;
    const GPUVAddr end = address + size;
    for (GPUVAddr page = address & ~page_mask; page < end; page += page_mask + 1) {
        const auto device = memory.GpuToCpuAddress(page);
        if (!device) {
            // It may be mapped by the time the back writes
            Classify(Effect::Unknown);
            return;
        }
        const GPUVAddr begin = (std::max)(page, address);
        const GPUVAddr last = (std::min)(page + page_mask + 1, end);
        ClassifyWrite(*device + (begin - page), *device + (last - page));
    }
}

u64 GpuSplit::ReadTarget(DAddr begin, DAddr end) {
    FlushStates();
    const u64 executed = entry_tail.load(std::memory_order_acquire);
    while (!noted_writes.empty() && noted_writes.front().position <= executed) {
        noted_writes.pop_front();
    }
    u64 target = (std::max)(last_unknown, (std::min)(last_gpu, last_sync));
    // Positions grow towards the back: the newest overlap is the latest
    for (auto it = noted_writes.rbegin(); it != noted_writes.rend(); ++it) {
        if (it->begin < end && begin < it->end) {
            target = (std::max)(target, it->position);
            break;
        }
    }
    return target;
}

void GpuSplit::Hold(u64 target) {
    FlushStates();
    hold_at.store(target, std::memory_order_seq_cst);
    FrontWaiting waiting{front_waiting};
    int spins = 0;
    for (;;) {
        const u64 tail = entry_tail.load(std::memory_order_seq_cst);
        if (tail >= target) {
            // Idle: nothing can be pushed while held. Parked: the back stored `parked` after its
            // last entry and checks hold_at again before it leaves (Run)
            if (tail == entry_head.load(std::memory_order_relaxed) ||
                parked.load(std::memory_order_seq_cst) == tail + 1) {
                return;
            }
        }
        Relax(spins);
    }
}

void GpuSplit::WaitUntil(u64 target) noexcept {
    FrontWaiting waiting{front_waiting};
    int spins = 0;
    while (entry_tail.load(std::memory_order_acquire) < target) {
        Relax(spins);
    }
}

void GpuSplit::Release() {
    hold_at.store(~u64{0}, std::memory_order_seq_cst);
}

void GpuSplit::Drain(std::source_location where) {
    FlushStates();
    const u64 head = entry_head.load(std::memory_order_relaxed);
    // EDEN_GPU_SPLIT_STATS: how often each call site drains and how long it waits (front only)
    static const bool stats = std::getenv("EDEN_GPU_SPLIT_STATS") != nullptr;
    if (stats) [[unlikely]] {
        struct Site {
            const char* file;
            u32 line;
            u64 count;
            u64 busy;
            s64 wait_ns;
        };
        static std::vector<Site> sites;
        static auto last_log = std::chrono::steady_clock::now();
        const auto start = std::chrono::steady_clock::now();
        const bool busy = entry_tail.load(std::memory_order_acquire) != head;
        FrontWaiting waiting{front_waiting};
        int spins = 0;
        while (entry_tail.load(std::memory_order_acquire) != head) {
            Relax(spins);
        }
        const auto end = std::chrono::steady_clock::now();
        auto it = std::find_if(sites.begin(), sites.end(), [&](const Site& site) {
            return site.line == where.line() && site.file == where.file_name();
        });
        if (it == sites.end()) {
            sites.push_back({where.file_name(), where.line(), 0, 0, 0});
            it = sites.end() - 1;
        }
        ++it->count;
        it->busy += busy ? 1 : 0;
        it->wait_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        if (end - last_log >= std::chrono::seconds{5}) {
            for (const Site& site : sites) {
                const std::string_view file{site.file};
                LOG_WARNING(HW_GPU, "split drain {}:{} count {} busy {} wait {} us",
                            file.substr(file.find_last_of("/\\") + 1), site.line, site.count,
                            site.busy, site.wait_ns / 1000);
            }
            sites.clear();
            last_log = end;
        }
        return;
    }
    FrontWaiting waiting{front_waiting};
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
    case Kind::States:
        static_cast<Tegra::DmaPusher*>(entry.target)
            ->BackStoreMethods(entry.subchannel, data.data() + entry.argument, entry.amount);
        break;
    case Kind::EngineStates: {
        // What an EngineMethod entry does for each
        auto* const engine = static_cast<Tegra::Engines::EngineInterface*>(entry.target);
        engine->ConsumeSink(system);
        const u32* const words = data.data() + entry.argument;
        for (u32 i = 0; i < entry.amount; ++i) {
            engine->CallMethod(system, words[2 * i], words[2 * i + 1], true);
        }
        if (entry.segment != 0) {
            reinterpret_cast<void (*)(void*)>(entry.segment)(entry.target);
        }
        break;
    }
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
            if (tail >= hold_at.load(std::memory_order_seq_cst)) [[unlikely]] {
                // The front reads guest memory (Hold): wait between two entries
                parked.store(tail + 1, std::memory_order_seq_cst);
                int spins = 0;
                for (;;) {
                    while (tail >= hold_at.load(std::memory_order_seq_cst)) {
                        Relax(spins);
                    }
                    parked.store(0, std::memory_order_seq_cst);
                    if (tail < hold_at.load(std::memory_order_seq_cst)) {
                        break;
                    }
                    parked.store(tail + 1, std::memory_order_seq_cst);
                }
            }
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
            if (flush_requested.load(std::memory_order_relaxed)) [[unlikely]] {
                BackFlush();
            }
            if (deferred_submit.load(std::memory_order_relaxed)) [[unlikely]] {
                SubmitWhenIdle(tail + 1);
            }
            data_tail.store(entry.data_end, std::memory_order_release);
            entry_tail.store(tail + 1, std::memory_order_release);
        }
    }
}

} // namespace VideoCommon
