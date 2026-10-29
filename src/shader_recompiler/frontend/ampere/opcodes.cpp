// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ampere/opcodes.h"

namespace Shader::Ampere {

const char* NameOf(Opcode opcode) {
    constexpr const char* NAME_TABLE[] = {
#define INST(name, cute, encode) cute,
#define INST_ALIAS(name, cute, encode)
#include "ampere.inc"
#undef INST_ALIAS
#undef INST
        "Invalid",
    };
    if (size_t(opcode) >= sizeof(NAME_TABLE) / sizeof(NAME_TABLE[0]))
        throw InvalidArgument("Invalid opcode with raw value {}", int(opcode));
    return NAME_TABLE[size_t(opcode)];
}

} // namespace Shader::Ampere
