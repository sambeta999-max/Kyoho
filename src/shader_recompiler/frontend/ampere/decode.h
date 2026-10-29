// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "shader_recompiler/frontend/ampere/instruction.h"
#include "shader_recompiler/frontend/ampere/opcodes.h"

namespace Shader::Ampere {

// Returns Opcode::Invalid for encodings outside the SM86 instruction set
[[nodiscard]] Opcode Decode(const Instruction& insn) noexcept;

} // namespace Shader::Ampere
