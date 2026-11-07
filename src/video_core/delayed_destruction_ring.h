// SPDX-FileCopyrightText: Copyright 2025 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <deque>
#include <utility>
#include <vector>

namespace VideoCommon {

/// Container to push objects to be destroyed a few ticks in the future
template <typename T, size_t TICKS_TO_DESTROY>
class DelayedDestructionRing {
public:
    void Tick() {
        index = (index + 1) % TICKS_TO_DESTROY;
        elements[index].clear();
    }

    /// Like Tick, but destroys at most min_per_tick or a 32nd of the objects due, whichever is
    /// more; the rest, already unused, wait in order and go first on the next ticks. Drivers take
    /// about 0.3 ms to free an image's memory (GA102), so the few hundred render targets retired
    /// at once in Ocarina of Time stalled the frame they were due on for 300 ms.
    void TickSpread(size_t min_per_tick) {
        index = (index + 1) % TICKS_TO_DESTROY;
        auto& due = elements[index];
        if (overdue.empty() && due.size() <= min_per_tick) {
            due.clear();
            return;
        }
        for (T& object : due) {
            overdue.push_back(std::move(object));
        }
        due.clear();
        const size_t count = (std::min)(overdue.size(),
                                        (std::max)(min_per_tick, overdue.size() / 32));
        overdue.erase(overdue.begin(), overdue.begin() + count);
    }

    void Push(T&& object) {
        elements[index].push_back(std::move(object));
    }

private:
    size_t index = 0;
    std::array<std::vector<T>, TICKS_TO_DESTROY> elements;
    std::deque<T> overdue;
};

} // namespace VideoCommon
