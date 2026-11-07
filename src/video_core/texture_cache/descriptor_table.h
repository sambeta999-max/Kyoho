// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <cstring>
#include <vector>

#include "common/alignment.h"
#include "common/common_types.h"
#include "common/div_ceil.h"
#include "common/slot_vector.h"
#include "common/assert.h"
#include "core/memory.h"
#include "video_core/memory_manager.h"
#include "video_core/rasterizer_interface.h"

namespace VideoCommon {

template <typename T>
class DescriptorTable {
public:
    [[nodiscard]] bool Synchronize(GPUVAddr gpu_addr, u32 limit) noexcept {
        bool ret = !(current_gpu_addr == gpu_addr && current_limit == limit);
        if (ret) {
            Refresh(gpu_addr, limit);
        }
        return ret;
    }

    // Every image deletion invalidates the tables of all channels, and Ocarina of Time deletes a
    // few images every frame; clearing the whole bitmap (8K words or more per table) each time
    // took most of a deletion. The clear waits for the next read and covers only the words
    // marked since the last one.
    void Invalidate() noexcept {
        invalidated = true;
    }

    [[nodiscard]] std::pair<T, bool> Read(Tegra::MemoryManager const& gpu_memory, u32 index) noexcept {
        DEBUG_ASSERT(index <= current_limit);
        const GPUVAddr gpu_addr = current_gpu_addr + index * sizeof(T);
        std::pair<T, bool> result;
        // This runs for every texture and sampler of every draw. An entry within one page is
        // contiguous in host memory, so a single page lookup replaces the generic block read.
        // From Eglantine a667551cdf (src/video_core/texture_cache/descriptor_table.h).
        const bool one_page =
            (gpu_addr & Core::Memory::YUZU_PAGEMASK) + sizeof(T) <= Core::Memory::YUZU_PAGESIZE;
        if (const u8* const host = one_page ? gpu_memory.GetPointer(gpu_addr) : nullptr) {
            std::memcpy(std::addressof(result.first), host, sizeof(T));
        } else {
            gpu_memory.ReadBlockUnsafe(gpu_addr, std::addressof(result.first), sizeof(T));
        }
        if (invalidated) [[unlikely]] {
            ClearReadMarks();
        }
        if ((read_descriptors[index / 64] & (1ULL << (index % 64))) != 0) {
            result.second = result.first != descriptors[index];
        } else {
            read_descriptors[index / 64] |= 1ULL << (index % 64);
            marked_words_end = (std::max<size_t>)(marked_words_end, index / 64 + 1);
            result.second = true;
        }
        if (result.second) {
            descriptors[index] = result.first;
        }
        return result;
    }

    void Refresh(GPUVAddr gpu_addr, u32 limit) noexcept {
        current_gpu_addr = gpu_addr;
        current_limit = limit;
        // Mario Brothership reallocates a lot of times, so use aggressive pre-alloc sizes
        // std::vector<T> by default uses quadratic growth, but that isn't even enough to satisfy brothership
        const size_t num_descriptors = ((limit + 0x80000) & (~0x7ffff)) + 1;
        ClearReadMarks();
        read_descriptors.resize(Common::DivCeil(num_descriptors, 64U));
        descriptors.resize(num_descriptors);
        bound_ids.resize(num_descriptors);
    }

    void ClearReadMarks() noexcept {
        const size_t end = (std::min)(marked_words_end, read_descriptors.size());
        std::fill(read_descriptors.begin(), read_descriptors.begin() + end, 0);
        marked_words_end = 0;
        invalidated = false;
    }

    std::vector<u64> read_descriptors;
    std::vector<T> descriptors;
    /// The cache's view or sampler of each descriptor read: valid where Read reported it
    /// unchanged. An array instead of a map keyed by index, which every texture and sampler of
    /// every draw looked up. After Lichii (TextureCache::VisitImageView's table.bound_ids).
    std::vector<Common::SlotId> bound_ids;
    GPUVAddr current_gpu_addr{};
    u32 current_limit{};
    size_t marked_words_end{};
    bool invalidated{};
};

} // namespace VideoCommon
