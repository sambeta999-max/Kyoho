// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>
#include <thread>

#include "common/assert.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "common/thread.h"
#include "core/core.h"
#include "core/frontend/graphics_context.h"
#include "video_core/control/scheduler.h"
#include "video_core/dma_pusher.h"
#include "video_core/gpu.h"
#include "video_core/gpu_split.h"
#include "video_core/gpu_thread.h"
#include "video_core/host1x/host1x.h"
#include "video_core/renderer_base.h"

namespace VideoCommon::GPUThread {

ThreadManager::ThreadManager(Core::System& system_)
    : system{system_}
{}

ThreadManager::~ThreadManager() = default;

void ThreadManager::StartThread(VideoCore::RendererBase& renderer, Core::Frontend::GraphicsContext& context, Tegra::Control::Scheduler& scheduler) {
    rasterizer = renderer.ReadRasterizer();
    thread = std::jthread([&](std::stop_token stop_token) {
        Common::SetCurrentThreadName("GPU");
        Common::SetCurrentThreadPriority(Common::ThreadPriority::Critical);
        Common::SetCurrentThreadToPerformanceCores();
        if (Common::GpuThreadPinningRequested()) {
            Common::PinCurrentThread(Common::PinnedRole::GpuFront);
        }
        system.RegisterHostThread();

        auto current_context = context.Acquire();
        CommandDataContainer next;
        // A suspended channel usually resumes within microseconds, once the host GPU or another
        // channel releases its semaphore: poll without sleeping for a while
        auto poll_until = std::chrono::steady_clock::time_point{};
        while (!stop_token.stop_requested()) {
            if (scheduler.HasPendingNativeChannels()) {
                scheduler.ResumePending(system.GPU());
                if (!state.queue.TryPop(next)) {
                    const auto now = std::chrono::steady_clock::now();
                    if (poll_until == std::chrono::steady_clock::time_point{}) {
                        // With a split GPU thread the release is usually work the back is
                        // still executing
                        poll_until = now + std::chrono::milliseconds{GpuSplit::Active() ? 20 : 2};
                    }
                    if (now < poll_until) {
                        std::this_thread::yield();
                    } else {
                        std::this_thread::sleep_for(std::chrono::milliseconds{1});
                    }
                    continue;
                }
                poll_until = {};
            } else {
                poll_until = {};
                state.queue.PopWait(next, stop_token);
            }
            if (stop_token.stop_requested()) {
                break;
            }
            auto* const split = GpuSplit::Active();
            if (auto* submit_list = std::get_if<SubmitListCommand>(&next.data)) {
                scheduler.Push(system.GPU(), submit_list->channel, std::move(submit_list->entries));
            } else if (split) {
                // Everything else reaches the rasterizer: the back runs it in order
                split->PushTask([this, &renderer, data = std::move(next.data)] {
                    if (std::holds_alternative<GPUTickCommand>(data)) {
                        system.GPU().TickWork();
                    } else if (const auto* flush = std::get_if<FlushRegionCommand>(&data)) {
                        renderer.ReadRasterizer()->FlushRegion(flush->addr, flush->size);
                    } else if (const auto* invalidate =
                                   std::get_if<InvalidateRegionCommand>(&data)) {
                        renderer.ReadRasterizer()->OnCacheInvalidation(invalidate->addr,
                                                                       invalidate->size);
                    }
                });
            } else if (std::holds_alternative<GPUTickCommand>(next.data)) {
                system.GPU().TickWork();
            } else if (const auto* flush = std::get_if<FlushRegionCommand>(&next.data)) {
                renderer.ReadRasterizer()->FlushRegion(flush->addr, flush->size);
            } else if (const auto* invalidate = std::get_if<InvalidateRegionCommand>(&next.data)) {
                renderer.ReadRasterizer()->OnCacheInvalidation(invalidate->addr, invalidate->size);
            } else {
                ASSERT(false);
            }
            const auto signal = [this, fence = next.fence, block = next.block] {
                state.signaled_fence.store(fence);
                if (block) {
                    // We have to lock the write_lock to ensure that the condition_variable wait
                    // not get a race between the check and the lock itself.
                    std::scoped_lock lk{state.write_lock};
                    state.cv.notify_all();
                }
            };
            // The first Switch 2 submission may have started the split
            if (auto* const active_split = GpuSplit::Active()) {
                active_split->PushTask(signal);
            } else {
                signal();
            }
        }
    });
}

void ThreadManager::SubmitList(s32 channel, Tegra::CommandList&& entries, bool is_async) {
    PushCommand(SubmitListCommand(channel, std::move(entries)), false, is_async);
}

void ThreadManager::FlushRegion(DAddr addr, u64 size, bool is_async) {
    if (!is_async) {
        // Always flush with synchronous GPU mode
        PushCommand(FlushRegionCommand(addr, size), false, is_async);
    }
}

void ThreadManager::TickGPU(bool is_async) {
    PushCommand(GPUTickCommand(), false, is_async);
}

void ThreadManager::InvalidateRegion(DAddr addr, u64 size) {
    rasterizer->OnCacheInvalidation(addr, size);
}

void ThreadManager::FlushAndInvalidateRegion(DAddr addr, u64 size, bool is_async) {
    if (Settings::IsGPULevelHigh()) {
        if (!is_async) {
            PushCommand(FlushRegionCommand(addr, size), false, is_async);
        } else {
            auto& gpu = system.GPU();
            const u64 fence = gpu.RequestFlush(addr, size);
            TickGPU(is_async);
            gpu.WaitForSyncOperation(fence);
        }
    }
    rasterizer->OnCacheInvalidation(addr, size);
}

u64 ThreadManager::PushCommand(CommandData&& command_data, bool block, bool is_async) {
    if (!is_async) {
        // In synchronous GPU mode, block the caller until the command has executed
        block = true;
    }

    std::unique_lock lk(state.write_lock);
    const u64 fence{++state.last_fence};
    state.queue.EmplaceWait(std::move(command_data), fence, block);

    if (block) {
        state.cv.wait(lk, thread.get_stop_token(), [this, fence] {
            return fence <= state.signaled_fence.load(std::memory_order_relaxed);
        });
    }

    return fence;
}

void ThreadManager::NotifyShutdown() {
    if (thread.joinable()) {
        thread.request_stop();
        thread.join();
    }
}

} // namespace VideoCommon::GPUThread
