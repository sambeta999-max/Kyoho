// SPDX-FileCopyrightText: Copyright 2023 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>

#include "common/common_types.h"

namespace VideoCore {

struct RasterizerDownloadArea {
    VAddr start_address;
    VAddr end_address;
    bool preemtive;
};

/// Bumped when the GPU writes buffer data that no fence downloads: each CPU core then drops the
/// read area it keeps and asks again at its next read of cached memory (BufferCache::ArmReadTraps)
inline std::atomic<u64> read_area_generation{};

} // namespace VideoCore
