// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>

#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ampere/translate/impl/impl.h"
#include "shader_recompiler/program_header.h"

namespace Shader::Ampere {

void TranslatorVisitor::EXIT(Instruction) {
    if (env.ShaderStage() != Stage::Fragment) {
        return;
    }
    // Fragment outputs are taken from consecutive registers as described by the program header
    const ProgramHeader sph{env.SPH()};
    IR::Reg src_reg{IR::Reg::R0};
    for (u32 render_target = 0; render_target < 8; ++render_target) {
        if (!sph.ps.HasOutputComponents(render_target)) {
            continue;
        }
        const std::array<bool, 4> mask{sph.ps.EnabledOutputComponents(render_target)};
        for (u32 component = 0; component < 4; ++component) {
            if (mask[component]) {
                ir.SetFragColor(render_target, component, F(src_reg));
            }
            ++src_reg;
        }
    }
    if (sph.ps.omap.sample_mask != 0) {
        ir.SetSampleMask(X(src_reg));
    }
    if (sph.ps.omap.depth != 0) {
        ir.SetFragDepth(F(src_reg + 1));
    }
}

void TranslatorVisitor::BAR(Instruction insn) {
    // Only BAR.SYNC over the whole workgroup with an immediate barrier id
    const bool immediate_id{insn.Opcode() == 0xb1d};
    const bool all_threads{insn.Bits(42, 12) == 0};
    if (!immediate_id || !all_threads || insn.Bits(77, 3) != 0) {
        throw NotImplementedException("BAR mode");
    }
    ir.Barrier();
}

void TranslatorVisitor::MEMBAR(Instruction insn) {
    switch (insn.Bits(76, 3)) {
    case 0:
        ir.WorkgroupMemoryBarrier();
        break;
    default:
        ir.DeviceMemoryBarrier();
        break;
    }
}

} // namespace Shader::Ampere
