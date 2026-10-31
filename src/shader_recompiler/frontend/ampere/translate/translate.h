// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "shader_recompiler/environment.h"
#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/maxwell/location.h"

namespace Shader::Ampere {

/// Called with the message of each instruction the translator skips as not implemented, when
/// set (offline coverage tools); the instruction is logged and skipped either way
inline void (*unimplemented_observer)(const char* message){};

/// Translate the SM86 instructions in [begin, end) of the shared location grid
void Translate(Environment& env, IR::Block* block, Maxwell::Location begin,
               Maxwell::Location end);

} // namespace Shader::Ampere
