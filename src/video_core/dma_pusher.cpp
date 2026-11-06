// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <bit>

#include <fmt/ranges.h>

#include "common/settings.h"
#include "core/core.h"
#include "video_core/dma_pusher.h"
#include "video_core/draw_capture.h"
#include "video_core/control/channel_state.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/gpu.h"
#include "video_core/gpu_split.h"
#include "video_core/guest_memory.h"
#include "video_core/memory_manager.h"
#include "video_core/rasterizer_interface.h"

namespace Tegra {

constexpr u32 MacroRegistersStart = 0xE00;
constexpr u32 UploadLaunchDma = 0x6C;
constexpr u32 ComputeInline = 0x6D;

DmaPusher::DmaPusher(Core::System& system_, MemoryManager& memory_manager_, Control::ChannelState& channel_state_)
    : system{system_}
    , memory_manager{memory_manager_}
    , channel_state{channel_state_}
    , signal_sync{false}
    , synced{false}
{}

DmaPusher::~DmaPusher() = default;

void DmaPusher::DispatchCalls() {
    dma_state.is_last_call = true;
    while (system.IsPoweredOn()) {
        if (!Step()) {
            break;
        }
    }
    // A macro whose data may continue in a later submission cannot wait for it.
    FlushPendingMacros();
    if (auto* const split = VideoCommon::GpuSplit::Active()) {
        split->PushTask([this] {
            BackFlushPendingMacros(max_subchannels, 0);
            system.GPU().FlushCommands();
            system.GPU().OnCommandListEnd();
        });
        return;
    }
    system.GPU().FlushCommands();
    system.GPU().OnCommandListEnd();
}

bool DmaPusher::Step() {
    if (!ib_enable || dma_pushbuffer.empty()) {
        return false;
    }

    auto* const split = VideoCommon::GpuSplit::Active();
    // A channel suspended on a semaphore continues where it stopped once the semaphore is ready
    if (split != nullptr) {
        if (puller.HasPendingNativeAcquire()) {
            if (!CanResumeNativeAcquire()) {
                return false;
            }
            puller.ClearPendingAcquire();
        }
    } else if (!puller.ResumeNativeAcquire(*this)) {
        return false;
    }

    CommandList& command_list = dma_pushbuffer.front();

    const size_t prefetch_size = command_list.prefetch_command_list.size();
    const size_t command_list_size = command_list.command_lists.size();

    if (prefetch_size == 0 && command_list_size == 0) {
        dma_pushbuffer.pop();
        dma_pushbuffer_subindex = 0;
        return true;
    }

    if (prefetch_size > 0) {
        const std::span<const CommandHeader> commands{command_list.prefetch_command_list};
        command_resume_offset += ProcessCommands(commands.subspan(command_resume_offset), command_resume_offset);
        if (puller.HasPendingNativeAcquire()) return false;
        command_resume_offset = 0;
        dma_pushbuffer.pop();
        return true;
    }

    auto& current_command = command_list.command_lists[dma_pushbuffer_subindex];
    const CommandListHeader& header = current_command;
    dma_state.dma_get = header.addr;

    if (signal_sync && !synced) {
        std::unique_lock lk(sync_mutex);
        sync_cv.wait(lk, [this]() { return synced; });
        signal_sync = false;
        synced = false;
    }

    if (header.size > 0) {
        // The front of a split GPU thread fetches ahead of what the back has executed, as a
        // PBDMA does: a SYNC entry waits for the methods before it, and memory the GPU wrote has
        // to be flushed; both need the back to be idle. Lists on pages the GPU wrote are read
        // from the copies the buffer cache makes after the work that writes them when it can
        bool split_safe = false;
        bool split_copied = false;
        if (split) {
            const size_t bytes = size_t{header.size} * sizeof(u32);
            if (header.sync) {
                split_safe = true;
            } else if (SplitReadsGpuWritten(*split, dma_state.dma_get, bytes)) {
                split_copied = SplitReadGpuWrittenCommands(*split, dma_state.dma_get,
                                                           static_cast<u32>(header.size));
                split_safe = !split_copied;
            }
            if (split_safe) {
                split->Drain();
                SplitFlushCommandPages(*split, dma_state.dma_get, bytes);
            }
        }
        if (subchannels[dma_state.subchannel] && dma_state.method_count) {
            const auto engine = subchannel_type[dma_state.subchannel];
            const bool kepler_payload = engine == Engines::EngineTypes::KeplerCompute && dma_state.method == ComputeInline && dma_state.non_incrementing;
            const bool macro_payload = engine == Engines::EngineTypes::Maxwell3D && dma_state.method >= MacroRegistersStart;
            if (kepler_payload || macro_payload) {
                const size_t words = std::min<size_t>(dma_state.method_count, header.size);
                auto* const payload_engine = subchannels[dma_state.subchannel];
                // On a split GPU thread the same lock-free check as for command words: the
                // buffer cache lock IsMemoryDirty takes is held by the back most of the time
                // (Fast Fusion: a fifth of the front waited for it). Pages the GPU ever wrote
                // are a superset of the dirty ones; a parameter wrongly taken as dirty only
                // goes through the GPU-side (indirect) path or is read again
                const bool dirty =
                    split ? SplitReadsGpuWritten(*split, dma_state.dma_get, words * sizeof(u32))
                          : memory_manager.IsMemoryDirty(dma_state.dma_get, words * sizeof(u32));
                if (auto& capture{VideoCommon::DrawCapture::Get()}; capture.Active()) {
                    // Macro parameters fetched from memory (indirect draws): what the front read
                    std::array<u32, 6> first{};
                    memory_manager.ReadBlockUnsafe(dma_state.dma_get, first.data(),
                                                   std::min<size_t>(words, first.size()) * sizeof(u32));
                    capture.Line(fmt::format("pb macro {:#x} payload {:#x} words={} dirty={} "
                                             "gpu_written={} split_safe={} first={}",
                                             dma_state.method, dma_state.dma_get, words, dirty,
                                             split != nullptr && SplitReadsGpuWritten(
                                                 *split, dma_state.dma_get, words * sizeof(u32)),
                                             split_safe, fmt::join(first, ",")));
                }
                if (split != nullptr && kepler_payload) {
                    // Compute data uploads run on the back of a split GPU thread
                    split->PushTask([payload_engine, dirty] { payload_engine->current_dirty = dirty; });
                } else {
                    payload_engine->current_dirty = dirty;
                }
            }
        }
        // A split GPU thread flushed the pages above when it had to
        const bool use_safe = split ? false : (Settings::IsDMALevelDefault() ? Settings::IsGPULevelHigh() : Settings::IsDMALevelSafe());
        if (split && command_resume_offset != 0 && split_saved_address == dma_state.dma_get &&
            split_saved_commands.size() == header.size) {
            // A channel resumed in the middle of a segment goes on with the words the front
            // fetched, as a PBDMA does; the guest may have reused that memory since
            const std::span<const CommandHeader> commands{split_saved_commands};
            command_resume_offset += ProcessCommands(commands.subspan(command_resume_offset), command_resume_offset);
        } else if (split_copied) {
            const std::span<const CommandHeader> commands{command_headers.data(), header.size};
            command_resume_offset += ProcessCommands(commands.subspan(command_resume_offset), command_resume_offset);
            if (puller.HasPendingNativeAcquire()) {
                split_saved_commands.assign(commands.begin(), commands.end());
                split_saved_address = dma_state.dma_get;
            }
        } else if (use_safe) {
            Tegra::Memory::GpuGuestMemory<Tegra::CommandHeader, Tegra::Memory::GuestMemoryFlags::SafeRead>headers(memory_manager, dma_state.dma_get, header.size, &command_headers);
            const std::span<const CommandHeader> commands{headers.data(), headers.size()};
            command_resume_offset += ProcessCommands(commands.subspan(command_resume_offset), command_resume_offset);
            if (split && puller.HasPendingNativeAcquire()) {
                split_saved_commands.assign(commands.begin(), commands.end());
                split_saved_address = dma_state.dma_get;
            }
        } else {
            Tegra::Memory::GpuGuestMemory<Tegra::CommandHeader, Tegra::Memory::GuestMemoryFlags::UnsafeRead>headers(memory_manager, dma_state.dma_get, header.size, &command_headers);
            const std::span<const CommandHeader> commands{headers.data(), headers.size()};
            command_resume_offset += ProcessCommands(commands.subspan(command_resume_offset), command_resume_offset);
            if (split && puller.HasPendingNativeAcquire()) {
                split_saved_commands.assign(commands.begin(), commands.end());
                split_saved_address = dma_state.dma_get;
            }
        }
    }

    if (puller.HasPendingNativeAcquire()) return false;
    command_resume_offset = 0;
    if (++dma_pushbuffer_subindex >= command_list_size) {
        dma_pushbuffer.pop();
        dma_pushbuffer_subindex = 0;
    } else {
        signal_sync = command_list.command_lists[dma_pushbuffer_subindex].sync && Settings::values.sync_memory_operations.GetValue();
    }

    if (signal_sync) {
        auto signal = [this] {
            rasterizer->SignalFence([this]() {
                std::scoped_lock lk(sync_mutex);
                synced = true;
                sync_cv.notify_all();
            });
        };
        if (split) {
            split->PushTask(std::move(signal));
        } else {
            signal();
        }
    }

    return true;
}

size_t DmaPusher::ProcessCommands(std::span<const CommandHeader> commands, size_t word_offset) {
    for (size_t index = 0; index < commands.size();) {
        // Data word of methods command
        if (dma_state.method_count && dma_state.non_incrementing) {
            auto const& command_header = commands[index]; //must ref (MUltiMethod re)
            dma_state.dma_word_offset = u64((word_offset + index) * sizeof(u32));
            u32 max_write = u32(std::min<std::size_t>(index + dma_state.method_count, commands.size()) - index);
            if (channel_state.native_abi && dma_state.method < non_puller_methods) max_write = 1;
            CallMultiMethod(&command_header.argument, max_write);
            dma_state.method_count -= max_write;
            dma_state.is_last_call = true;
            index += max_write;
        } else if (dma_state.method_count) {
            auto const command_header = commands[index]; //can copy
            dma_state.dma_word_offset = u64((word_offset + index) * sizeof(u32));
            dma_state.is_last_call = dma_state.method_count <= 1;
            CallMethod(command_header.argument);
            dma_state.method += !dma_state.non_incrementing ? 1 : 0;
            dma_state.non_incrementing |= dma_increment_once;
            dma_state.method_count--;
            index++;
        } else {
            auto const command_header = commands[index]; //can copy
            // No command active - this is the first word of a new one
            switch (command_header.mode) {
            case SubmissionMode::Increasing:
                SetState(command_header);
                dma_state.non_incrementing = false;
                dma_increment_once = false;
                break;
            case SubmissionMode::NonIncreasing:
                SetState(command_header);
                dma_state.non_incrementing = true;
                dma_increment_once = false;
                break;
            case SubmissionMode::Inline:
                dma_state.method = command_header.method;
                dma_state.subchannel = command_header.subchannel;
                dma_state.dma_word_offset = u64(-s64(dma_state.dma_get)); // negate to set address as 0
                CallMethod(command_header.arg_count);
                dma_state.non_incrementing = true;
                dma_increment_once = false;
                break;
            case SubmissionMode::IncreaseOnce:
                SetState(command_header);
                dma_state.non_incrementing = false;
                dma_increment_once = true;
                break;
            default:
                break;
            }
            index++;
        }
        if (puller.HasPendingNativeAcquire()) return index;
    }
    return commands.size();
}

void DmaPusher::SetState(const CommandHeader& command_header) {
    dma_state.method = command_header.method;
    dma_state.subchannel = command_header.subchannel;
    dma_state.method_count = command_header.method_count;
}

void DmaPusher::FlushPendingMacros(u32 subchannel, u32 method) {
    for (u32 mask = pending_macro_mask; mask != 0; mask &= mask - 1) {
        const u32 index = static_cast<u32>(std::countr_zero(mask));
        auto* const engine = subchannels[index];
        if (index == subchannel && method >= non_puller_methods && engine->IsMacroData(method)) {
            continue;
        }
        engine->FlushPendingMacro(system);
        pending_macro_mask &= ~(1U << index);
    }
}

void DmaPusher::FlushPendingMacros() {
    FlushPendingMacros(max_subchannels, 0);
}

void DmaPusher::CallMethod(u32 argument) {
    if (auto* const split = VideoCommon::GpuSplit::Active()) {
        SplitCallMethod(*split, argument);
        return;
    }
    if (pending_macro_mask != 0) {
        FlushPendingMacros(dma_state.subchannel, dma_state.method);
    }
    if (dma_state.method < non_puller_methods) {
        puller.CallPullerMethod(*this, Engines::Puller::MethodCall{
            dma_state.method,
            argument,
            dma_state.subchannel,
            dma_state.method_count,
        });
    } else {
        auto subchannel = subchannels[dma_state.subchannel];
        if (!subchannel->execution_mask[dma_state.method]) {
            subchannel->method_sink.emplace_back(dma_state.method, argument);
        } else {
            subchannel->ConsumeSink(system);
            subchannel->current_dma_segment = dma_state.dma_get + dma_state.dma_word_offset;
            subchannel->CallMethod(system, dma_state.method, argument, dma_state.is_last_call);
            if (subchannel->HasPendingMacro()) {
                pending_macro_mask |= 1U << dma_state.subchannel;
            }
        }
    }
}

void DmaPusher::CallMultiMethod(const u32* base_start, u32 num_methods) {
    if (auto* const split = VideoCommon::GpuSplit::Active()) {
        SplitCallMultiMethod(*split, base_start, num_methods);
        return;
    }
    if (pending_macro_mask != 0) {
        FlushPendingMacros(dma_state.subchannel, dma_state.method);
    }
    if (dma_state.method < non_puller_methods) {
        puller.CallMultiMethod(*this, dma_state.method, dma_state.subchannel, base_start, num_methods, dma_state.method_count);
    } else {
        auto subchannel = subchannels[dma_state.subchannel];
        subchannel->ConsumeSink(system);
        subchannel->current_dma_segment = dma_state.dma_get + dma_state.dma_word_offset;
        subchannel->CallMultiMethod(system, dma_state.method, base_start, num_methods, dma_state.method_count);
        if (subchannel->HasPendingMacro()) {
            pending_macro_mask |= 1U << dma_state.subchannel;
        }
    }
}

bool DmaPusher::SplitReadsGpuWritten(VideoCommon::GpuSplit& split, GPUVAddr address,
                                     size_t size) {
    // Pages the GPU ever wrote always take the drained, flushing read: whether their data is still
    // only on the GPU can't be told before the back ran what the front queued (the dispatch that
    // rewrites a command list each frame may not have run yet; its draws would then use an older
    // frame's words). A lock-free check; for a range the GPU never wrote, nothing is only on it
    if (const auto device = memory_manager.GpuToCpuAddress(address, size)) {
        return split.WasGpuWritten(*device, size);
    }
    return memory_manager.IsMemoryDirty(address, size);
}

bool DmaPusher::SplitReadGpuWrittenCommands(VideoCommon::GpuSplit& split, GPUVAddr address,
                                             u32 count) {
    // Ported from Eglantine 8f1435f8db ("nx2: read GPU-written command lists from copies instead
    // of draining", dma_pusher.cpp ReadGpuWrittenCommands). The pages are marked as command list
    // pages: from the next write on, the buffer cache copies them to host memory right after the
    // dispatch or DMA operation that writes them and submits. The front then waits only for the
    // back to execute that work and for the host GPU to finish the copy, not for everything
    // queued and a cache flush with a full GPU wait (Fast Fusion: ~1500 drains and ~37 of those
    // waits a second for the 5-word indirect draw parameters its culling dispatches write, with
    // the back mostly idle meanwhile). Pages flushed whole since no GPU work wrote are read from
    // guest memory
    if (rasterizer == nullptr || count == 0) {
        return false;
    }
    constexpr GPUVAddr page_mask = VideoCommon::GpuSplit::PageSize - 1;
    const GPUVAddr end = address + u64{count} * sizeof(u32);
    for (GPUVAddr page = address & ~page_mask; page < end; page += page_mask + 1) {
        const auto device = memory_manager.GpuToCpuAddress(page);
        if (!device) {
            return false;
        }
        if (split.WasGpuWritten(*device, 1)) {
            split.MarkCommandListPage(*device);
        }
    }
    split.WaitForWriters();
    using GpuWrittenList = VideoCore::RasterizerInterface::GpuWrittenList;
    command_headers.resize_destructive(count);
    u8* const out = reinterpret_cast<u8*>(command_headers.data());
    for (GPUVAddr chunk = address; chunk < end;) {
        const GPUVAddr chunk_end = (std::min)(end, (chunk & ~page_mask) + page_mask + 1);
        const std::span<u8> part{out + (chunk - address), static_cast<size_t>(chunk_end - chunk)};
        const auto device = memory_manager.GpuToCpuAddress(chunk);
        if (!device) {
            return false;
        }
        GpuWrittenList where = GpuWrittenList::Synced;
        if (split.WasGpuWritten(*device, part.size())) {
            where = rasterizer->ReadGpuWrittenCommandList(*device, part);
        }
        if (where == GpuWrittenList::Unknown) {
            return false;
        }
        if (where == GpuWrittenList::Synced) {
            memory_manager.ReadBlockUnsafe(chunk, part.data(), part.size());
        }
        chunk = chunk_end;
    }
    return true;
}

void DmaPusher::SplitFlushCommandPages(VideoCommon::GpuSplit& split, GPUVAddr address,
                                       size_t size) {
    // The back is idle. The whole pages are flushed: their next lists read guest memory until GPU
    // work writes them again (SplitReadGpuWrittenCommands)
    constexpr GPUVAddr page_mask = VideoCommon::GpuSplit::PageSize - 1;
    const GPUVAddr first_page = address & ~page_mask;
    const GPUVAddr end_page = (address + size + page_mask) & ~page_mask;
    if (memory_manager.IsMemoryDirty(first_page, end_page - first_page)) {
        memory_manager.FlushRegion(first_page, end_page - first_page);
    }
    if (rasterizer == nullptr || !split.HasCommandListPages()) {
        return;
    }
    for (GPUVAddr page = first_page; page < end_page; page += page_mask + 1) {
        if (const auto device = memory_manager.GpuToCpuAddress(page);
            device && split.IsCommandListPage(*device)) {
            rasterizer->NoteGpuWrittenCommandListSynced(*device);
        }
    }
}

bool DmaPusher::IsSplitWriter(u32 subchannel, u32 method) const {
    // What can write the command lists the GPU fills in: compute dispatches, DMA operations and
    // inline uploads (SplitReadGpuWrittenCommands waits for the back to execute them)
    switch (subchannel_type[subchannel]) {
    case Engines::EngineTypes::KeplerCompute:
    case Engines::EngineTypes::MaxwellDMA:
    case Engines::EngineTypes::KeplerMemory:
        return true;
    case Engines::EngineTypes::Maxwell3D:
        return method == UploadLaunchDma || method == ComputeInline;
    default:
        return false;
    }
}

bool DmaPusher::CanResumeNativeAcquire() {
    auto* const split = VideoCommon::GpuSplit::Active();
    if (split == nullptr || split->OnBackThread()) {
        return puller.CanResumeNativeAcquire(*this);
    }
    if (!puller.HasPendingNativeAcquire() || split->Executed() < split_acquire_position) {
        return false;
    }
    if (puller.IsPendingAcquireMet(*this)) {
        // No wait for the host GPU here: Raiders never started, the fences being waited for
        // needed work of the suspended channel
        return true;
    }
    // Usually the release is work the back has not executed yet: check again later. Once the
    // back is idle, only the host GPU can still release it; waiting for that is expensive, so
    // the back does it only then
    if (split->Executed() == split->Pushed()) {
        split->PushTask([this] { rasterizer->ReleaseFences(); });
        split_acquire_position = split->Pushed();
    }
    return false;
}

Engines::EngineInterface* DmaPusher::FrontMacroEngine(u32 subchannel) {
    auto* const engine = subchannels[subchannel];
    if (engine == nullptr || !engine->SupportsFrontMacros()) {
        return nullptr;
    }
    if (!engine->MacrosOnFront()) {
        // The back may still hold a call of the macros of this engine
        auto& split = *VideoCommon::GpuSplit::Active();
        split.PushTask([this] { BackFlushPendingMacros(max_subchannels, 0); });
        split.Drain();
        engine->EnableFrontMacros();
    }
    return engine;
}

void DmaPusher::SplitPullerMethod(VideoCommon::GpuSplit& split,
                                  const Engines::Puller::MethodCall& call) {
    switch (puller.StoreForSplit(*this, call)) {
    case Engines::Puller::SplitPlace::State:
        return;
    case Engines::Puller::SplitPlace::Back:
        puller.ClassifyBackAction(*this, call);
        split.PushTask(puller.MakeBackAction(*this, call));
        return;
    case Engines::Puller::SplitPlace::Front:
        if (const auto semaphore = puller.DecodeNativeAcquire(*this, call)) {
            // A greater-or-equal counter that holds already still holds after the releases the
            // back has not executed yet
            const bool greater_equal =
                semaphore->operation == Engines::HostSemaphore::Operation::AcquireStrictGeq ||
                semaphore->operation == Engines::HostSemaphore::Operation::AcquireCircGeq;
            if (greater_equal && puller.IsAcquiredNow(*this, *semaphore)) {
                return;
            }
            // Most acquires hold once the back has executed the release they wait for (its
            // value is pending until the host GPU reaches it, as on one thread): the front goes
            // on as soon as it has, the work after the acquire is queued behind the release.
            // One that still does not hold once the back is idle suspends the channel without
            // stopping the front; it is checked again once the back has executed everything
            // before it
            puller.SetPendingAcquire(*semaphore);
            split_acquire_position = split.Pushed();
            if (split.DrainUnless([&] { return puller.IsPendingAcquireMet(*this); })) {
                puller.ClearPendingAcquire();
                return;
            }
            if (!puller.IsPendingAcquireMet(*this)) {
                rasterizer->ReleaseFences();
            }
            if (puller.IsPendingAcquireMet(*this)) {
                puller.ClearPendingAcquire();
            }
            return;
        }
        // Acquires decide whether this channel goes on now and binding changes the subchannels;
        // the rasterizer needs the back to be idle
        split.PushTask([this] { BackFlushPendingMacros(max_subchannels, 0); });
        split.Drain();
        puller.CallPullerMethod(*this, call);
        return;
    }
}

void DmaPusher::SplitCallMethod(VideoCommon::GpuSplit& split, u32 argument) {
    const u32 method = dma_state.method;
    const u32 subchannel = dma_state.subchannel;
    if (pending_macro_mask != 0) {
        FlushPendingMacros(subchannel, method);
    }
    if (method < non_puller_methods) {
        SplitPullerMethod(split, Engines::Puller::MethodCall{method, argument, subchannel,
                                                             dma_state.method_count});
        return;
    }
    const GPUVAddr segment = dma_state.dma_get + dma_state.dma_word_offset;
    if (auto* const engine = FrontMacroEngine(subchannel)) {
        if (engine->IsFrontMethod(method)) {
            engine->current_dma_segment = segment;
            engine->FrontCallMethod(system, method, argument);
            if (engine->HasPendingMacro()) {
                pending_macro_mask |= 1U << subchannel;
            }
            return;
        }
        engine->FrontWrite(method, argument);
    }
    split.PushMethod(*this, subchannel, method, argument, dma_state.is_last_call, segment);
    if (IsSplitWriter(subchannel, method)) {
        split.NoteWriter();
    }
}

void DmaPusher::SplitCallMultiMethod(VideoCommon::GpuSplit& split, const u32* base_start,
                                     u32 num_methods) {
    const u32 method = dma_state.method;
    const u32 subchannel = dma_state.subchannel;
    if (pending_macro_mask != 0) {
        FlushPendingMacros(subchannel, method);
    }
    if (method < non_puller_methods) {
        for (u32 i = 0; i < num_methods; ++i) {
            SplitPullerMethod(split, Engines::Puller::MethodCall{method, base_start[i], subchannel,
                                                                 dma_state.method_count - i});
        }
        return;
    }
    const GPUVAddr segment = dma_state.dma_get + dma_state.dma_word_offset;
    if (auto* const engine = FrontMacroEngine(subchannel)) {
        if (engine->IsFrontMethod(method)) {
            engine->current_dma_segment = segment;
            engine->FrontCallMultiMethod(system, method, base_start, num_methods);
            if (engine->HasPendingMacro()) {
                pending_macro_mask |= 1U << subchannel;
            }
            return;
        }
        engine->FrontMultiWrite(method, base_start, num_methods);
    }
    split.PushMultiMethod(*this, subchannel, method, base_start, num_methods,
                          dma_state.method_count, segment);
    if (IsSplitWriter(subchannel, method)) {
        split.NoteWriter();
    }
}

void DmaPusher::BackFlushPendingMacros(u32 subchannel, u32 method) {
    for (u32 mask = back_pending_macro_mask; mask != 0; mask &= mask - 1) {
        const u32 index = static_cast<u32>(std::countr_zero(mask));
        auto* const engine = subchannels[index];
        if (index == subchannel && method >= non_puller_methods && engine->IsMacroData(method)) {
            continue;
        }
        engine->FlushPendingMacro(system);
        back_pending_macro_mask &= ~(1U << index);
    }
}

void DmaPusher::BackCallMethod(u32 subchannel, u32 method, u32 argument, bool is_last_call,
                               GPUVAddr segment) {
    if (back_pending_macro_mask != 0) {
        BackFlushPendingMacros(subchannel, method);
    }
    auto* const engine = subchannels[subchannel];
    if (!engine->execution_mask[method]) {
        engine->method_sink.emplace_back(method, argument);
        return;
    }
    engine->ConsumeSink(system);
    engine->current_dma_segment = segment;
    engine->CallMethod(system, method, argument, is_last_call);
    if (!engine->MacrosOnFront() && engine->HasPendingMacro()) {
        back_pending_macro_mask |= 1U << subchannel;
    }
}

void DmaPusher::BackCallMultiMethod(u32 subchannel, u32 method, const u32* base_start,
                                    u32 amount, u32 methods_pending, GPUVAddr segment) {
    if (back_pending_macro_mask != 0) {
        BackFlushPendingMacros(subchannel, method);
    }
    auto* const engine = subchannels[subchannel];
    engine->ConsumeSink(system);
    engine->current_dma_segment = segment;
    engine->CallMultiMethod(system, method, base_start, amount, methods_pending);
    if (!engine->MacrosOnFront() && engine->HasPendingMacro()) {
        back_pending_macro_mask |= 1U << subchannel;
    }
}

void DmaPusher::BindRasterizer(VideoCore::RasterizerInterface* rasterizer_) {
    rasterizer = rasterizer_;
}

} // namespace Tegra
