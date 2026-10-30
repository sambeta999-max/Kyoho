// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <string>

#include <fmt/ranges.h>

#include "shader_recompiler/frontend/ir/condition.h"

namespace Shader::IR {

std::string NameOf(Condition condition) {
    std::string ret;
    if (condition.GetFlowTest() != FlowTest::T) {
        ret = fmt::to_string(condition.GetFlowTest());
    }
    const auto [pred, negated]{condition.GetPred()};
    if (!ret.empty()) {
        ret += '&';
    }
    if (negated) {
        ret += '!';
    }
    ret += fmt::to_string(pred);
    const auto [cond_pred, cond_negated]{condition.GetCondPred()};
    if (!IsTrue(cond_pred) || cond_negated) {
        ret += '&';
        if (cond_negated) {
            ret += '!';
        }
        ret += fmt::to_string(cond_pred);
    }
    return ret;
}

} // namespace Shader::IR
