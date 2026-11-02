// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>

#include <fmt/format.h>

#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/backend/spirv/emit_spirv_instructions.h"
#include "shader_recompiler/backend/spirv/spirv_emit_context.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ir/modifiers.h"

// SPV_KHR_cooperative_matrix. Sirit has no emitters for it, so its types are declared as opaque
// types and its instructions as calls of opaque marker ids; RewriteCooperativeMatrices turns them
// into the real instructions after the module is assembled.
namespace Shader::Backend::SPIRV {
namespace {
constexpr u32 OP_TYPE_OPAQUE{31};
constexpr u32 OP_FUNCTION_CALL{57};
constexpr u32 OP_TYPE_COOPERATIVE_MATRIX{4456};
constexpr u32 OP_COOPERATIVE_MATRIX_LOAD{4457};
constexpr u32 OP_COOPERATIVE_MATRIX_STORE{4458};
constexpr u32 OP_COOPERATIVE_MATRIX_MUL_ADD{4459};
constexpr u32 CAPABILITY_COOPERATIVE_MATRIX{6022};

constexpr u32 USE_MATRIX_A{0};
constexpr u32 USE_MATRIX_B{1};
constexpr u32 USE_ACCUMULATOR{2};
constexpr u32 ROW_MAJOR{0};
constexpr u32 COLUMN_MAJOR{1};

void Enable(EmitContext& ctx) {
    if (ctx.cooperative_matrix.load.value != 0) {
        return;
    }
    ctx.AddExtension("SPV_KHR_cooperative_matrix");
    ctx.AddCapability(static_cast<spv::Capability>(CAPABILITY_COOPERATIVE_MATRIX));
    ctx.cooperative_matrix.load = ctx.TypeOpaque("eden.coopmat.load");
    ctx.cooperative_matrix.store = ctx.TypeOpaque("eden.coopmat.store");
    ctx.cooperative_matrix.mul_add = ctx.TypeOpaque("eden.coopmat.muladd");
}

/// F16 matrix of the subgroup
Id MatrixType(EmitContext& ctx, u32 rows, u32 columns, u32 use) {
    for (const auto& type : ctx.cooperative_matrix.types) {
        if (type.rows == rows && type.columns == columns && type.use == use) {
            return type.id;
        }
    }
    // The operands must be declared before the type that uses them
    const Id scope{ctx.Const(static_cast<u32>(spv::Scope::Subgroup))};
    const Id rows_id{ctx.Const(rows)};
    const Id columns_id{ctx.Const(columns)};
    const Id use_id{ctx.Const(use)};
    const Id id{ctx.TypeOpaque(fmt::format("eden.coopmat.{}x{}.{}", rows, columns, use))};
    ctx.cooperative_matrix.types.push_back({
        .id = id,
        .rows = rows,
        .columns = columns,
        .use = use,
        .operands = {ctx.F16[1].value, scope.value, rows_id.value, columns_id.value,
                     use_id.value},
    });
    return id;
}

/// Pointer to the shared memory word at a byte offset
Id SharedWord(EmitContext& ctx, Id offset) {
    const Id index{ctx.OpShiftRightLogical(ctx.U32[1], offset, ctx.Const(2U))};
    if (ctx.uses_explicit_workgroup_layout) {
        return ctx.OpAccessChain(ctx.shared_u32, ctx.shared_memory_u32, ctx.u32_zero_value, index);
    }
    return ctx.OpAccessChain(ctx.shared_u32, ctx.shared_memory_u32, index);
}

Id Load(EmitContext& ctx, Id type, Id offset, u32 layout, u32 stride_words) {
    return ctx.OpFunctionCall(type, ctx.cooperative_matrix.load, SharedWord(ctx, offset),
                              ctx.Const(layout), ctx.Const(stride_words));
}
} // Anonymous namespace

void EmitSubgroupBarrier(EmitContext& ctx) {
    const auto semantics{spv::MemorySemanticsMask::AcquireRelease |
                         spv::MemorySemanticsMask::WorkgroupMemory};
    ctx.OpControlBarrier(ctx.Const(static_cast<u32>(spv::Scope::Subgroup)),
                         ctx.Const(static_cast<u32>(spv::Scope::Subgroup)),
                         ctx.Const(static_cast<u32>(semantics)));
}

Id EmitHmmaF16(EmitContext&, Id, Id, Id) {
    throw LogicError("HmmaF16 must be lowered");
}

// A is a row-major 16xK matrix, B a column-major Kx8 matrix and C a row-major 16x8 matrix of F16
// elements; D replaces C.
void EmitCooperativeMatrixMulAddF16(EmitContext& ctx, IR::Inst* inst, Id a, Id b, Id c) {
    const auto info{inst->Flags<IR::HmmaInfo>()};
    const u32 k{info.k16 != 0 ? 16U : 8U};
    Enable(ctx);
    const Id a_type{MatrixType(ctx, 16, k, USE_MATRIX_A)};
    const Id b_type{MatrixType(ctx, k, 8, USE_MATRIX_B)};
    const Id c_type{MatrixType(ctx, 16, 8, USE_ACCUMULATOR)};
    const Id a_matrix{Load(ctx, a_type, a, ROW_MAJOR, k / 2)};
    const Id b_matrix{Load(ctx, b_type, b, COLUMN_MAJOR, k / 2)};
    const Id c_matrix{Load(ctx, c_type, c, ROW_MAJOR, 4)};
    const Id d_matrix{ctx.OpFunctionCall(c_type, ctx.cooperative_matrix.mul_add, a_matrix,
                                         b_matrix, c_matrix)};
    (void)ctx.OpFunctionCall(ctx.void_id, ctx.cooperative_matrix.store, SharedWord(ctx, c),
                             d_matrix, ctx.Const(ROW_MAJOR), ctx.Const(4U));
}

void RewriteCooperativeMatrices(std::vector<u32>& code, const EmitContext& ctx) {
    const auto& matrices{ctx.cooperative_matrix};
    if (matrices.load.value == 0) {
        return;
    }
    std::vector<u32> result(code.begin(), code.begin() + 5);
    result.reserve(code.size());
    size_t offset{5};
    while (offset < code.size()) {
        const u32 word_count{code[offset] >> 16};
        const u32 opcode{code[offset] & 0xFFFFu};
        if (word_count == 0 || offset + word_count > code.size()) {
            throw LogicError("Malformed SPIR-V at word {}", offset);
        }
        const auto words{std::span{code}.subspan(offset, word_count)};
        offset += word_count;
        if (opcode == OP_TYPE_OPAQUE) {
            const u32 id{words[1]};
            if (id == matrices.load.value || id == matrices.store.value ||
                id == matrices.mul_add.value) {
                continue;
            }
            const auto it{std::ranges::find(matrices.types, id, [](const auto& type) {
                return type.id.value;
            })};
            if (it != matrices.types.end()) {
                result.push_back((7U << 16) | OP_TYPE_COOPERATIVE_MATRIX);
                result.push_back(id);
                result.insert(result.end(), it->operands.begin(), it->operands.end());
                continue;
            }
        } else if (opcode == OP_FUNCTION_CALL) {
            // result type, result, function, arguments...
            const u32 function{words[3]};
            const auto arguments{words.subspan(4)};
            if (function == matrices.load.value || function == matrices.mul_add.value) {
                const u32 op{function == matrices.load.value ? OP_COOPERATIVE_MATRIX_LOAD
                                                              : OP_COOPERATIVE_MATRIX_MUL_ADD};
                result.push_back(((word_count - 1) << 16) | op);
                result.push_back(words[1]);
                result.push_back(words[2]);
                result.insert(result.end(), arguments.begin(), arguments.end());
                continue;
            }
            if (function == matrices.store.value) {
                result.push_back(((word_count - 3) << 16) | OP_COOPERATIVE_MATRIX_STORE);
                result.insert(result.end(), arguments.begin(), arguments.end());
                continue;
            }
        }
        result.insert(result.end(), words.begin(), words.end());
    }
    code = std::move(result);
}

} // namespace Shader::Backend::SPIRV
