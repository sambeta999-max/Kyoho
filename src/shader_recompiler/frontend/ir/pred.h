// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <fmt/ranges.h>

#include "common/common_types.h"

namespace Shader::IR {

enum class Pred : u64 {
    P0,
    P1,
    P2,
    P3,
    P4,
    P5,
    P6,
    PT,
    // Uniform datapath predicates (SM75+)
    UP0,
    UP1,
    UP2,
    UP3,
    UP4,
    UP5,
    UP6,
    UPT,
};

constexpr size_t NUM_USER_PREDS = 7;
constexpr size_t NUM_PREDS = 16;

[[nodiscard]] constexpr size_t PredIndex(Pred pred) noexcept {
    return static_cast<size_t>(pred);
}

[[nodiscard]] constexpr bool IsUniform(Pred pred) noexcept {
    return pred > Pred::PT;
}

[[nodiscard]] constexpr bool IsTrue(Pred pred) noexcept {
    return pred == Pred::PT || pred == Pred::UPT;
}

/// Uniform predicate from its encoded index, where 7 is UPT
[[nodiscard]] constexpr Pred UniformPred(u32 index) noexcept {
    return static_cast<Pred>(static_cast<u32>(Pred::UP0) + (index & 7));
}

} // namespace Shader::IR

template <>
struct fmt::formatter<Shader::IR::Pred> {
    constexpr auto parse(format_parse_context& ctx) {
        return ctx.begin();
    }
    template <typename FormatContext>
    auto format(const Shader::IR::Pred& pred, FormatContext& ctx) const {
        if (pred == Shader::IR::Pred::PT) {
            return fmt::format_to(ctx.out(), "PT");
        } else if (pred == Shader::IR::Pred::UPT) {
            return fmt::format_to(ctx.out(), "UPT");
        } else if (Shader::IR::IsUniform(pred)) {
            return fmt::format_to(ctx.out(), "UP{}",
                                  static_cast<int>(pred) - static_cast<int>(Shader::IR::Pred::UP0));
        } else {
            return fmt::format_to(ctx.out(), "P{}", static_cast<int>(pred));
        }
    }
};
