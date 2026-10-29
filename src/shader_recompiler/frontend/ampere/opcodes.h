// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <fmt/ranges.h>

namespace Shader::Ampere {

enum class Opcode {
#define INST(name, cute, encode) name,
#define INST_ALIAS(name, cute, encode)
#include "ampere.inc"
#undef INST_ALIAS
#undef INST
    Invalid,
};

const char* NameOf(Opcode opcode);

} // namespace Shader::Ampere

template <>
struct fmt::formatter<Shader::Ampere::Opcode> {
    constexpr auto parse(format_parse_context& ctx) {
        return ctx.begin();
    }
    template <typename FormatContext>
    auto format(const Shader::Ampere::Opcode& opcode, FormatContext& ctx) const {
        return fmt::format_to(ctx.out(), "{}", NameOf(opcode));
    }
};
