// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "common/common_types.h"
#include "shader_recompiler/frontend/maxwell/location.h"

namespace Shader::Ampere {

// SM70+ instructions are 16 bytes with no scheduling slots. They are placed one per slot on the
// Maxwell location grid, so the Maxwell control flow graph and structurizer can be shared.

[[nodiscard]] constexpr Maxwell::Location ToLocation(u32 offset) {
    const u32 index{offset / 16};
    return Maxwell::Location{32 * (index / 3) + 8 * (index % 3 + 1)};
}

[[nodiscard]] constexpr u32 ToOffset(Maxwell::Location location) {
    const u32 offset{location.Offset()};
    const u32 index{(offset / 32) * 3 + (offset % 32) / 8 - 1};
    return index * 16;
}

} // namespace Shader::Ampere
