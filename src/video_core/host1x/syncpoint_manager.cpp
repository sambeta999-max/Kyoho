// SPDX-FileCopyrightText: Copyright 2025 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "video_core/host1x/syncpoint_manager.h"

namespace Tegra {

namespace Host1x {

// Set while this thread runs due actions; a cancel from inside one must not wait for itself.
static thread_local bool running_on_this_thread = false;

SyncpointManager::ActionHandle SyncpointManager::RegisterAction(
    std::atomic<u32>& syncpoint, std::list<RegisteredAction>& action_storage, u32 expected_value,
    std::function<void()>&& action) {
    if (syncpoint.load(std::memory_order_acquire) >= expected_value) {
        action();
        return {};
    }

    std::scoped_lock lk(guard);
    if (syncpoint.load(std::memory_order_relaxed) >= expected_value) {
        action();
        return {};
    }
    auto it = action_storage.begin();
    while (it != action_storage.end()) {
        if (it->expected_value >= expected_value) {
            break;
        }
        ++it;
    }
    return action_storage.emplace(it, expected_value, ++last_action_id, std::move(action))->id;
}

void SyncpointManager::DeregisterAction(std::list<RegisteredAction>& action_storage,
                                        const ActionHandle& handle) {
    if (handle == 0) {
        return; // It ran inside RegisterAction
    }
    std::unique_lock lk(guard);

    for (auto it = action_storage.begin(); it != action_storage.end(); it++) {
        if (it->id == handle) {
            action_storage.erase(it);
            return;
        }
    }
    // Not registered any more: it may be due and about to run outside the guard. Return only once
    // it ran, so a cancelled action never fires after its cancel, as when actions ran under it.
    if (!running_on_this_thread) {
        actions_done_cv.wait(lk, [this] { return running_actions == 0; });
    }
}

void SyncpointManager::DeregisterGuestAction(u32 syncpoint_id, const ActionHandle& handle) {
    DeregisterAction(guest_action_storage[syncpoint_id], handle);
}

void SyncpointManager::DeregisterHostAction(u32 syncpoint_id, const ActionHandle& handle) {
    DeregisterAction(host_action_storage[syncpoint_id], handle);
}

void SyncpointManager::IncrementGuest(u32 syncpoint_id) {
    Increment(syncpoints_guest[syncpoint_id], wait_guest_cv, guest_action_storage[syncpoint_id]);
}

void SyncpointManager::IncrementHost(u32 syncpoint_id) {
    Increment(syncpoints_host[syncpoint_id], wait_host_cv, host_action_storage[syncpoint_id]);
}

void SyncpointManager::WaitGuest(u32 syncpoint_id, u32 expected_value) {
    Wait(syncpoints_guest[syncpoint_id], wait_guest_cv, expected_value);
}

void SyncpointManager::WaitHost(u32 syncpoint_id, u32 expected_value) {
    Wait(syncpoints_host[syncpoint_id], wait_host_cv, expected_value);
}

void SyncpointManager::Increment(std::atomic<u32>& syncpoint, std::condition_variable& wait_cv,
                                 std::list<RegisteredAction>& action_storage) {
    auto new_value{syncpoint.fetch_add(1, std::memory_order_acq_rel) + 1};

    // The due actions run after the guard is released: the composite action can block on the
    // display (AcquireNextImage), and every other syncpoint user (the fencing thread's Increment,
    // the guest's RegisterAction) then waited on the guard behind it, freezing Donkey Kong
    // Bananza. Splicing keeps their value order. From Orchid aa266e457 (adapted from
    // andrewleeog/Eglantine 9ef2dde1a) and 8837a265c (from zortos293/Eglantine d5b258210).
    std::list<RegisteredAction> due;
    {
        std::scoped_lock lk(guard);
        auto it = action_storage.begin();
        while (it != action_storage.end() && it->expected_value <= new_value) {
            due.splice(due.end(), action_storage, it++);
        }
        running_actions += due.empty() ? 0 : 1;
        wait_cv.notify_all();
    }
    if (due.empty()) {
        return;
    }
    const bool outer = running_on_this_thread;
    running_on_this_thread = true;
    for (auto& registered : due) {
        registered.action();
    }
    running_on_this_thread = outer;
    std::scoped_lock lk(guard);
    --running_actions;
    actions_done_cv.notify_all();
}

void SyncpointManager::Wait(std::atomic<u32>& syncpoint, std::condition_variable& wait_cv,
                            u32 expected_value) {
    const auto pred = [&]() { return syncpoint.load(std::memory_order_acquire) >= expected_value; };
    if (pred()) {
        return;
    }

    std::unique_lock lk(guard);
    wait_cv.wait(lk, pred);
}

} // namespace Host1x

} // namespace Tegra
