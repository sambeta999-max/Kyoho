// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <optional>
#include <span>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <boost/container/flat_set.hpp>
#include <boost/container/small_vector.hpp>

#include "common/alignment.h"
#include "shader_recompiler/environment.h"
#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/ir/breadth_first_search.h"
#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "shader_recompiler/frontend/ir/value.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/ir_opt/passes.h"

namespace Shader::Optimization {
namespace {
/// Address in constant buffers to the storage buffer descriptor
struct StorageBufferAddr {
    auto operator<=>(const StorageBufferAddr&) const noexcept = default;

    u32 index;
    u32 offset;
    /// Offset of the buffer pointer in the memory the constant buffer pointer points to
    u32 indirect{~0U};
};

/// Block iterator to a global memory instruction and the storage buffer it uses
struct StorageInst {
    StorageBufferAddr storage_buffer;
    IR::Inst* inst;
    IR::Block* block;
};

/// Bias towards a certain range of constant buffers when looking for storage buffers
struct Bias {
    u32 index;
    u32 offset_begin;
    u32 offset_end;
    u32 alignment;
};

using boost::container::flat_set;
using boost::container::small_vector;
using StorageBufferSet =
    flat_set<StorageBufferAddr, std::less<StorageBufferAddr>, small_vector<StorageBufferAddr, 16>>;
using StorageInstVector = small_vector<StorageInst, 24>;
using StorageWritesSet =
    flat_set<StorageBufferAddr, std::less<StorageBufferAddr>, small_vector<StorageBufferAddr, 16>>;

struct StorageInfo {
    StorageBufferSet set;
    StorageInstVector to_replace;
    StorageWritesSet writes;
    /// Track results by the instruction the search starts from, with and without the NVN bias:
    /// accesses through the same address share them, and a search can walk thousands of
    /// instructions (the IR does not change while storage buffers are collected).
    std::unordered_map<const IR::Inst*, std::optional<StorageBufferAddr>> biased_tracks;
    std::unordered_map<const IR::Inst*, std::optional<StorageBufferAddr>> unbiased_tracks;
    /// Buffers only reached through the global memory table: slots of pointer arrays indexed at
    /// run time, and entries of the tables such slots point to (compute only)
    small_vector<StorageBufferAddr, 16> global_pointers;
    StorageWritesSet global_writes;
};

/// Returns true when the instruction returns data read from memory (loads and atomics)
bool IsMemoryRead(const IR::Inst& inst) {
    switch (inst.GetOpcode()) {
    case IR::Opcode::LoadGlobalU8:
    case IR::Opcode::LoadGlobalS8:
    case IR::Opcode::LoadGlobalU16:
    case IR::Opcode::LoadGlobalS16:
    case IR::Opcode::LoadGlobal32:
    case IR::Opcode::LoadGlobal64:
    case IR::Opcode::LoadGlobal128:
    case IR::Opcode::LoadStorageU8:
    case IR::Opcode::LoadStorageS8:
    case IR::Opcode::LoadStorageU16:
    case IR::Opcode::LoadStorageS16:
    case IR::Opcode::LoadStorage32:
    case IR::Opcode::LoadStorage64:
    case IR::Opcode::LoadStorage128:
    case IR::Opcode::LoadLocal:
    case IR::Opcode::LoadSharedU8:
    case IR::Opcode::LoadSharedS8:
    case IR::Opcode::LoadSharedU16:
    case IR::Opcode::LoadSharedS16:
    case IR::Opcode::LoadSharedU32:
    case IR::Opcode::LoadSharedU64:
    case IR::Opcode::LoadSharedU128:
        return true;
    default:
        break;
    }
    const std::string_view name{IR::NameOf(inst.GetOpcode())};
    return name.starts_with("GlobalAtomic") || name.starts_with("StorageAtomic") ||
           name.starts_with("SharedAtomic");
}

/// Returns true when the instruction is a global memory instruction
bool IsGlobalMemory(const IR::Inst& inst) {
    switch (inst.GetOpcode()) {
    case IR::Opcode::LoadGlobalS8:
    case IR::Opcode::LoadGlobalU8:
    case IR::Opcode::LoadGlobalS16:
    case IR::Opcode::LoadGlobalU16:
    case IR::Opcode::LoadGlobal32:
    case IR::Opcode::LoadGlobal64:
    case IR::Opcode::LoadGlobal128:
    case IR::Opcode::WriteGlobalS8:
    case IR::Opcode::WriteGlobalU8:
    case IR::Opcode::WriteGlobalS16:
    case IR::Opcode::WriteGlobalU16:
    case IR::Opcode::WriteGlobal32:
    case IR::Opcode::WriteGlobal64:
    case IR::Opcode::WriteGlobal128:
    case IR::Opcode::GlobalAtomicIAdd32:
    case IR::Opcode::GlobalAtomicSMin32:
    case IR::Opcode::GlobalAtomicUMin32:
    case IR::Opcode::GlobalAtomicSMax32:
    case IR::Opcode::GlobalAtomicUMax32:
    case IR::Opcode::GlobalAtomicInc32:
    case IR::Opcode::GlobalAtomicDec32:
    case IR::Opcode::GlobalAtomicAnd32:
    case IR::Opcode::GlobalAtomicOr32:
    case IR::Opcode::GlobalAtomicXor32:
    case IR::Opcode::GlobalAtomicExchange32:
    case IR::Opcode::GlobalAtomicIAdd64:
    case IR::Opcode::GlobalAtomicSMin64:
    case IR::Opcode::GlobalAtomicUMin64:
    case IR::Opcode::GlobalAtomicSMax64:
    case IR::Opcode::GlobalAtomicUMax64:
    case IR::Opcode::GlobalAtomicAnd64:
    case IR::Opcode::GlobalAtomicOr64:
    case IR::Opcode::GlobalAtomicXor64:
    case IR::Opcode::GlobalAtomicExchange64:
    case IR::Opcode::GlobalAtomicIAdd32x2:
    case IR::Opcode::GlobalAtomicSMin32x2:
    case IR::Opcode::GlobalAtomicUMin32x2:
    case IR::Opcode::GlobalAtomicSMax32x2:
    case IR::Opcode::GlobalAtomicUMax32x2:
    case IR::Opcode::GlobalAtomicAnd32x2:
    case IR::Opcode::GlobalAtomicOr32x2:
    case IR::Opcode::GlobalAtomicXor32x2:
    case IR::Opcode::GlobalAtomicExchange32x2:
    case IR::Opcode::GlobalAtomicAddF32:
    case IR::Opcode::GlobalAtomicAddF16x2:
    case IR::Opcode::GlobalAtomicAddF32x2:
    case IR::Opcode::GlobalAtomicMinF16x2:
    case IR::Opcode::GlobalAtomicMinF32x2:
    case IR::Opcode::GlobalAtomicMaxF16x2:
    case IR::Opcode::GlobalAtomicMaxF32x2:
        return true;
    default:
        return false;
    }
}

/// Returns true when the instruction is a global memory instruction
bool IsGlobalMemoryWrite(const IR::Inst& inst) {
    switch (inst.GetOpcode()) {
    case IR::Opcode::WriteGlobalS8:
    case IR::Opcode::WriteGlobalU8:
    case IR::Opcode::WriteGlobalS16:
    case IR::Opcode::WriteGlobalU16:
    case IR::Opcode::WriteGlobal32:
    case IR::Opcode::WriteGlobal64:
    case IR::Opcode::WriteGlobal128:
    case IR::Opcode::GlobalAtomicIAdd32:
    case IR::Opcode::GlobalAtomicSMin32:
    case IR::Opcode::GlobalAtomicUMin32:
    case IR::Opcode::GlobalAtomicSMax32:
    case IR::Opcode::GlobalAtomicUMax32:
    case IR::Opcode::GlobalAtomicInc32:
    case IR::Opcode::GlobalAtomicDec32:
    case IR::Opcode::GlobalAtomicAnd32:
    case IR::Opcode::GlobalAtomicOr32:
    case IR::Opcode::GlobalAtomicXor32:
    case IR::Opcode::GlobalAtomicExchange32:
    case IR::Opcode::GlobalAtomicIAdd64:
    case IR::Opcode::GlobalAtomicSMin64:
    case IR::Opcode::GlobalAtomicUMin64:
    case IR::Opcode::GlobalAtomicSMax64:
    case IR::Opcode::GlobalAtomicUMax64:
    case IR::Opcode::GlobalAtomicAnd64:
    case IR::Opcode::GlobalAtomicOr64:
    case IR::Opcode::GlobalAtomicXor64:
    case IR::Opcode::GlobalAtomicExchange64:
    case IR::Opcode::GlobalAtomicIAdd32x2:
    case IR::Opcode::GlobalAtomicSMin32x2:
    case IR::Opcode::GlobalAtomicUMin32x2:
    case IR::Opcode::GlobalAtomicSMax32x2:
    case IR::Opcode::GlobalAtomicUMax32x2:
    case IR::Opcode::GlobalAtomicAnd32x2:
    case IR::Opcode::GlobalAtomicOr32x2:
    case IR::Opcode::GlobalAtomicXor32x2:
    case IR::Opcode::GlobalAtomicExchange32x2:
    case IR::Opcode::GlobalAtomicAddF32:
    case IR::Opcode::GlobalAtomicAddF16x2:
    case IR::Opcode::GlobalAtomicAddF32x2:
    case IR::Opcode::GlobalAtomicMinF16x2:
    case IR::Opcode::GlobalAtomicMinF32x2:
    case IR::Opcode::GlobalAtomicMaxF16x2:
    case IR::Opcode::GlobalAtomicMaxF32x2:
        return true;
    default:
        return false;
    }
}

/// Converts a global memory opcode to its storage buffer equivalent
IR::Opcode GlobalToStorage(IR::Opcode opcode) {
    switch (opcode) {
    case IR::Opcode::LoadGlobalS8:
        return IR::Opcode::LoadStorageS8;
    case IR::Opcode::LoadGlobalU8:
        return IR::Opcode::LoadStorageU8;
    case IR::Opcode::LoadGlobalS16:
        return IR::Opcode::LoadStorageS16;
    case IR::Opcode::LoadGlobalU16:
        return IR::Opcode::LoadStorageU16;
    case IR::Opcode::LoadGlobal32:
        return IR::Opcode::LoadStorage32;
    case IR::Opcode::LoadGlobal64:
        return IR::Opcode::LoadStorage64;
    case IR::Opcode::LoadGlobal128:
        return IR::Opcode::LoadStorage128;
    case IR::Opcode::WriteGlobalS8:
        return IR::Opcode::WriteStorageS8;
    case IR::Opcode::WriteGlobalU8:
        return IR::Opcode::WriteStorageU8;
    case IR::Opcode::WriteGlobalS16:
        return IR::Opcode::WriteStorageS16;
    case IR::Opcode::WriteGlobalU16:
        return IR::Opcode::WriteStorageU16;
    case IR::Opcode::WriteGlobal32:
        return IR::Opcode::WriteStorage32;
    case IR::Opcode::WriteGlobal64:
        return IR::Opcode::WriteStorage64;
    case IR::Opcode::WriteGlobal128:
        return IR::Opcode::WriteStorage128;
    case IR::Opcode::GlobalAtomicIAdd32:
        return IR::Opcode::StorageAtomicIAdd32;
    case IR::Opcode::GlobalAtomicSMin32:
        return IR::Opcode::StorageAtomicSMin32;
    case IR::Opcode::GlobalAtomicUMin32:
        return IR::Opcode::StorageAtomicUMin32;
    case IR::Opcode::GlobalAtomicSMax32:
        return IR::Opcode::StorageAtomicSMax32;
    case IR::Opcode::GlobalAtomicUMax32:
        return IR::Opcode::StorageAtomicUMax32;
    case IR::Opcode::GlobalAtomicInc32:
        return IR::Opcode::StorageAtomicInc32;
    case IR::Opcode::GlobalAtomicDec32:
        return IR::Opcode::StorageAtomicDec32;
    case IR::Opcode::GlobalAtomicAnd32:
        return IR::Opcode::StorageAtomicAnd32;
    case IR::Opcode::GlobalAtomicOr32:
        return IR::Opcode::StorageAtomicOr32;
    case IR::Opcode::GlobalAtomicXor32:
        return IR::Opcode::StorageAtomicXor32;
    case IR::Opcode::GlobalAtomicExchange32:
        return IR::Opcode::StorageAtomicExchange32;
    case IR::Opcode::GlobalAtomicIAdd64:
        return IR::Opcode::StorageAtomicIAdd64;
    case IR::Opcode::GlobalAtomicSMin64:
        return IR::Opcode::StorageAtomicSMin64;
    case IR::Opcode::GlobalAtomicUMin64:
        return IR::Opcode::StorageAtomicUMin64;
    case IR::Opcode::GlobalAtomicSMax64:
        return IR::Opcode::StorageAtomicSMax64;
    case IR::Opcode::GlobalAtomicUMax64:
        return IR::Opcode::StorageAtomicUMax64;
    case IR::Opcode::GlobalAtomicAnd64:
        return IR::Opcode::StorageAtomicAnd64;
    case IR::Opcode::GlobalAtomicOr64:
        return IR::Opcode::StorageAtomicOr64;
    case IR::Opcode::GlobalAtomicXor64:
        return IR::Opcode::StorageAtomicXor64;
    case IR::Opcode::GlobalAtomicExchange64:
        return IR::Opcode::StorageAtomicExchange64;
    case IR::Opcode::GlobalAtomicIAdd32x2:
        return IR::Opcode::StorageAtomicIAdd32x2;
    case IR::Opcode::GlobalAtomicSMin32x2:
        return IR::Opcode::StorageAtomicSMin32x2;
    case IR::Opcode::GlobalAtomicUMin32x2:
        return IR::Opcode::StorageAtomicUMin32x2;
    case IR::Opcode::GlobalAtomicSMax32x2:
        return IR::Opcode::StorageAtomicSMax32x2;
    case IR::Opcode::GlobalAtomicUMax32x2:
        return IR::Opcode::StorageAtomicUMax32x2;
    case IR::Opcode::GlobalAtomicAnd32x2:
        return IR::Opcode::StorageAtomicAnd32x2;
    case IR::Opcode::GlobalAtomicOr32x2:
        return IR::Opcode::StorageAtomicOr32x2;
    case IR::Opcode::GlobalAtomicXor32x2:
        return IR::Opcode::StorageAtomicXor32x2;
    case IR::Opcode::GlobalAtomicExchange32x2:
        return IR::Opcode::StorageAtomicExchange32x2;
    case IR::Opcode::GlobalAtomicAddF32:
        return IR::Opcode::StorageAtomicAddF32;
    case IR::Opcode::GlobalAtomicAddF16x2:
        return IR::Opcode::StorageAtomicAddF16x2;
    case IR::Opcode::GlobalAtomicMinF16x2:
        return IR::Opcode::StorageAtomicMinF16x2;
    case IR::Opcode::GlobalAtomicMaxF16x2:
        return IR::Opcode::StorageAtomicMaxF16x2;
    case IR::Opcode::GlobalAtomicAddF32x2:
        return IR::Opcode::StorageAtomicAddF32x2;
    case IR::Opcode::GlobalAtomicMinF32x2:
        return IR::Opcode::StorageAtomicMinF32x2;
    case IR::Opcode::GlobalAtomicMaxF32x2:
        return IR::Opcode::StorageAtomicMaxF32x2;
    default:
        throw InvalidArgument("Invalid global memory opcode {}", opcode);
    }
}

/// Returns true when a storage buffer address satisfies a bias
bool MeetsBias(const StorageBufferAddr& storage_buffer, const Bias& bias) noexcept {
    return storage_buffer.index == bias.index && storage_buffer.offset >= bias.offset_begin &&
           storage_buffer.offset < bias.offset_end;
}

struct LowAddrInfo {
    IR::U32 value;
    s32 imm_offset;
    std::optional<IR::U32> high;
};

/// Tries to track the first 32-bits of a global memory instruction
std::optional<LowAddrInfo> TrackLowAddress(IR::Inst* inst) {
    // The first argument is the low level GPU pointer to the global memory instruction
    const IR::Value addr{inst->Arg(0)};
    if (addr.IsImmediate()) {
        // Not much we can do if it's an immediate
        return std::nullopt;
    }
    // This address is expected to either be a PackUint2x32, a IAdd64, or a CompositeConstructU32x2
    IR::Inst* addr_inst{addr.InstRecursive()};
    s32 imm_offset{0};
    if (addr_inst->GetOpcode() == IR::Opcode::IAdd64) {
        // If it's an IAdd64, get the immediate offset it is applying and grab the address
        // instruction. This expects for the instruction to be canonicalized having the address on
        // the first argument and the immediate offset on the second one.
        const IR::U64 imm_offset_value{addr_inst->Arg(1)};
        if (!imm_offset_value.IsImmediate()) {
            return std::nullopt;
        }
        imm_offset = static_cast<s32>(static_cast<s64>(imm_offset_value.U64()));
        const IR::U64 iadd_addr{addr_inst->Arg(0)};
        if (iadd_addr.IsImmediate()) {
            return std::nullopt;
        }
        addr_inst = iadd_addr.InstRecursive();
    }
    // With IAdd64 handled, now PackUint2x32 is expected
    if (addr_inst->GetOpcode() == IR::Opcode::PackUint2x32) {
        // PackUint2x32 is expected to be generated from a vector
        const IR::Value vector{addr_inst->Arg(0)};
        if (vector.IsImmediate()) {
            return std::nullopt;
        }
        addr_inst = vector.InstRecursive();
    }
    // The vector is expected to be a CompositeConstructU32x2
    if (addr_inst->GetOpcode() != IR::Opcode::CompositeConstructU32x2) {
        return std::nullopt;
    }
    // Grab the first argument from the CompositeConstructU32x2, this is the low address.
    return LowAddrInfo{
        .value{IR::U32{addr_inst->Arg(0)}},
        .imm_offset = imm_offset,
        .high{IR::U32{addr_inst->Arg(1)}},
    };
}

/// Tries to track the storage buffer address used by a global memory instruction
std::optional<StorageBufferAddr> Track(const IR::Value& value, const Bias* bias) {
    const auto pred{[bias](const IR::Inst* inst) -> std::optional<StorageBufferAddr> {
        if (inst->GetOpcode() != IR::Opcode::GetCbufU32 &&
            inst->GetOpcode() != IR::Opcode::GetCbufU32x2) {
            return std::nullopt;
        }
        const IR::Value index{inst->Arg(0)};
        const IR::Value offset{inst->Arg(1)};
        if (!index.IsImmediate()) {
            // Definitely not a storage buffer if it's read from a
            // non-immediate index
            return std::nullopt;
        }
        if (!offset.IsImmediate()) {
            // TODO: Support SSBO arrays
            return std::nullopt;
        }
        const StorageBufferAddr storage_buffer{
            .index = index.U32(),
            .offset = offset.U32(),
        };
        const u32 alignment{bias ? bias->alignment : 8U};
        if (!Common::IsAligned(storage_buffer.offset, alignment)) {
            // The SSBO pointer has to be aligned
            return std::nullopt;
        }
        if (bias && !MeetsBias(storage_buffer, *bias)) {
            // We have to blacklist some addresses in case we wrongly
            // point to them
            return std::nullopt;
        }
        return storage_buffer;
    }};
    return BreadthFirstSearch(value, pred);
}

/// Returns true when a word is data read from memory (directly or plus an offset), not something
/// computed from constant buffer words. A phi is when all its defined inputs are: a pointer loaded
/// under a condition, undefined on the other path (Ocarina of Time's vegetation culling picks one
/// of two tables that way).
bool IsLoadedWord(const IR::Value& value, int depth = 2, int phi_depth = 4) {
    if (value.IsImmediate()) {
        return false;
    }
    const IR::Inst* inst{value.InstRecursive()};
    switch (inst->GetOpcode()) {
    case IR::Opcode::Phi: {
        if (phi_depth <= 0) {
            return false;
        }
        bool loaded{};
        for (size_t arg = 0; arg < inst->NumArgs(); ++arg) {
            const IR::Value input{inst->Arg(arg)};
            if (!input.IsImmediate() &&
                input.InstRecursive()->GetOpcode() == IR::Opcode::UndefU32) {
                continue;
            }
            if (!IsLoadedWord(input, depth, phi_depth - 1)) {
                return false;
            }
            loaded = true;
        }
        return loaded;
    }
    case IR::Opcode::CompositeExtractU32x2:
    case IR::Opcode::CompositeExtractU32x3:
    case IR::Opcode::CompositeExtractU32x4:
        if (inst->Arg(0).IsImmediate()) {
            return false;
        }
        return IsMemoryRead(*inst->Arg(0).InstRecursive());
    case IR::Opcode::IAdd32:
        return depth > 0 && (IsLoadedWord(inst->Arg(0), depth - 1, phi_depth) ||
                             IsLoadedWord(inst->Arg(1), depth - 1, phi_depth));
    default:
        return IsMemoryRead(*inst);
    }
}

/// Constant buffer words a value is a sum of (immediate index and offset). Only additions (and the
/// carries and copies around them) are followed: a pointer is added to offsets, while words that
/// are multiplied, shifted or compared are strides, sizes and bounds.
std::vector<StorageBufferAddr> CbufSources(const IR::Value& value) {
    std::vector<StorageBufferAddr> result;
    std::vector<const IR::Inst*> visited;
    std::vector<IR::Value> stack{value};
    while (!stack.empty()) {
        const IR::Value current{stack.back()};
        stack.pop_back();
        if (current.IsImmediate()) {
            continue;
        }
        const IR::Inst* const inst{current.InstRecursive()};
        if (std::ranges::find(visited, inst) != visited.end()) {
            continue;
        }
        visited.push_back(inst);
        switch (inst->GetOpcode()) {
        case IR::Opcode::GetCbufU32:
        case IR::Opcode::GetCbufU32x2:
            if (inst->Arg(0).IsImmediate() && inst->Arg(1).IsImmediate()) {
                const u32 index{inst->Arg(0).U32()};
                const u32 offset{inst->Arg(1).U32()};
                result.push_back({.index = index, .offset = offset});
                if (inst->GetOpcode() == IR::Opcode::GetCbufU32x2) {
                    // A 64-bit load (ULDC.64) provides both words of a pointer
                    result.push_back({.index = index, .offset = offset + 4});
                }
            }
            break;
        case IR::Opcode::IAdd32:
        case IR::Opcode::IAdd64:
        case IR::Opcode::ISub32:
        case IR::Opcode::Phi:
        case IR::Opcode::SelectU32:
        case IR::Opcode::CompositeConstructU32x2:
        case IR::Opcode::CompositeExtractU32x2:
        case IR::Opcode::PackUint2x32:
        case IR::Opcode::UnpackUint2x32:
        case IR::Opcode::BitwiseAnd32:
            for (size_t arg = 0; arg < inst->NumArgs(); ++arg) {
                stack.push_back(inst->Arg(arg));
            }
            break;
        default:
            break;
        }
    }
    return result;
}

/// A 64-bit address built from a pointer in a constant buffer takes its low word from c[x] and
/// its high word from c[x + 4]. Among the low word's sources, picks the one whose high word is
/// also a source of the address's high half, which rules out sizes and other words that only
/// enter the offset (Fast Fusion: c[0x188] + (c[0x250] << 1) * y + ...).
std::optional<StorageBufferAddr> TrackPointerPair(const LowAddrInfo& info) {
    if (!info.high || info.high->IsImmediate()) {
        return std::nullopt;
    }
    const std::vector<StorageBufferAddr> high_sources{CbufSources(*info.high)};
    std::optional<StorageBufferAddr> result;
    for (const StorageBufferAddr& low : CbufSources(info.value)) {
        if (!Common::IsAligned(low.offset, 8)) {
            continue;
        }
        const bool paired{std::ranges::any_of(high_sources, [&](const StorageBufferAddr& high) {
            return high.index == low.index && high.offset == low.offset + 4;
        })};
        // 64-bit offsets added to the pointer pair up as well (Fast Fusion adds c[0x200] to the
        // pointer at c[0x160]); kernel parameters put the pointers before the offsets, so the
        // lowest pair is the base
        if (paired && (!result || *result > low)) {
            result = low;
        }
    }
    return result;
}

/// A pointer read from a table at a constant buffer pointer plus a fixed offset: the table entry,
/// as a storage buffer whose pointer the binding reads from the table. The access stays global and
/// finds the buffer through the global memory table, where the binding registers it; without the
/// binding, memory that no other storage buffer covers is missing from the table and reads 0
/// (Ocarina of Time's culling and vegetation programs read their instance and material data
/// through such pointers).
struct TableLoad {
    LowAddrInfo table; ///< Address of the table the pointer is read from
    s64 indirect;      ///< Offset of the pointer from the table address
};

/// The loads a pointer read from memory comes from: even elements of 64 or 128-bit loads its low
/// word is (plus offsets), following phis of pointers loaded on different paths
small_vector<TableLoad, 2> FindTableLoads(const LowAddrInfo& addr) {
    small_vector<TableLoad, 2> result;
    small_vector<IR::Value, 8> pending{addr.value};
    for (size_t step = 0; step < pending.size() && step < 16; ++step) {
        if (pending[step].IsImmediate()) {
            continue;
        }
        const IR::Inst* const inst{pending[step].InstRecursive()};
        switch (inst->GetOpcode()) {
        case IR::Opcode::CompositeExtractU32x2:
        case IR::Opcode::CompositeExtractU32x4: {
            if (inst->Arg(0).IsImmediate() || !inst->Arg(1).IsImmediate() ||
                inst->Arg(1).U32() % 2 != 0) {
                break;
            }
            IR::Inst* const load{inst->Arg(0).InstRecursive()};
            if (load->GetOpcode() != IR::Opcode::LoadGlobal64 &&
                load->GetOpcode() != IR::Opcode::LoadGlobal128) {
                break;
            }
            const std::optional<LowAddrInfo> table{TrackLowAddress(load)};
            if (table && !table->value.IsImmediate() && result.size() < 4) {
                result.push_back(TableLoad{
                    .table = *table,
                    .indirect = s64{table->imm_offset} + s64{inst->Arg(1).U32()} * 4,
                });
            }
            break;
        }
        case IR::Opcode::IAdd32:
        case IR::Opcode::Phi:
            for (size_t arg = 0; arg < inst->NumArgs(); ++arg) {
                pending.push_back(inst->Arg(arg));
            }
            break;
        default:
            break;
        }
    }
    return result;
}

/// The entry of a table at a constant buffer pointer plus a fixed offset a load reads
std::optional<StorageBufferAddr> TrackTableEntry(const TableLoad& load) {
    // The table: a pointer read whole from a constant buffer, plus a fixed offset
    const LowAddrInfo* const table{&load.table};
    const IR::Inst* word{table->value.InstRecursive()};
    if (word->GetOpcode() == IR::Opcode::CompositeExtractU32x2 && word->Arg(1).IsImmediate() &&
        word->Arg(1).U32() == 0 && !word->Arg(0).IsImmediate()) {
        word = word->Arg(0).InstRecursive();
    }
    if ((word->GetOpcode() != IR::Opcode::GetCbufU32 &&
         word->GetOpcode() != IR::Opcode::GetCbufU32x2) ||
        !word->Arg(0).IsImmediate() || !word->Arg(1).IsImmediate()) {
        return std::nullopt;
    }
    const s64 indirect{load.indirect};
    const u32 offset{word->Arg(1).U32()};
    if (indirect < 0 || !Common::IsAligned(static_cast<u64>(indirect), 8) ||
        !Common::IsAligned(offset, 8)) {
        return std::nullopt;
    }
    return StorageBufferAddr{
        .index = word->Arg(0).U32(),
        .offset = offset,
        .indirect = static_cast<u32>(indirect),
    };
}

/// The first slot and stride of a pointer array read at a dynamic constant buffer offset:
/// (index << shift) + base or index * stride + base.
/// From Lichii f26ad432a (DynamicPointerSlots, global_memory_to_storage_buffer_pass.cpp).
std::optional<std::pair<u32, u32>> DynamicPointerSlots(const IR::Value& offset) {
    const auto stride_of = [](const IR::Value& value) -> std::optional<u32> {
        if (value.IsImmediate()) {
            return std::nullopt;
        }
        const IR::Inst* const inst{value.InstRecursive()};
        if (inst->GetOpcode() == IR::Opcode::ShiftLeftLogical32 && inst->Arg(1).IsImmediate() &&
            inst->Arg(1).U32() < 16) {
            return 1U << inst->Arg(1).U32();
        }
        if (inst->GetOpcode() == IR::Opcode::IMul32 && inst->Arg(1).IsImmediate()) {
            return inst->Arg(1).U32();
        }
        return std::nullopt;
    };
    u32 base{};
    std::optional<u32> stride{stride_of(offset)};
    if (!stride && !offset.IsImmediate() &&
        offset.InstRecursive()->GetOpcode() == IR::Opcode::IAdd32) {
        const IR::Inst* const add{offset.InstRecursive()};
        for (size_t arg = 0; arg < 2 && !stride; ++arg) {
            if (add->Arg(arg ^ 1).IsImmediate()) {
                base = add->Arg(arg ^ 1).U32();
                stride = stride_of(add->Arg(arg));
            }
        }
    }
    if (!stride || *stride < 8 || !Common::IsAligned(*stride, 8)) {
        return std::nullopt;
    }
    return std::pair{base, *stride};
}

/// Slots of a pointer array read at a dynamic offset, following phis and selects of the offset.
/// A stride of zero is a single slot. From Lichii f26ad432a (CollectPointerSlots).
void CollectPointerSlots(const IR::Value& offset, small_vector<std::pair<u32, u32>, 4>& slots,
                         std::unordered_set<const IR::Inst*>& visited) {
    if (offset.IsImmediate()) {
        if (std::ranges::find(slots, std::pair{offset.U32(), 0U}) == slots.end()) {
            slots.emplace_back(offset.U32(), 0U);
        }
        return;
    }
    const IR::Inst* const inst{offset.InstRecursive()};
    if (!visited.insert(inst).second) {
        return;
    }
    if (const auto pattern{DynamicPointerSlots(offset)}) {
        if (std::ranges::find(slots, *pattern) == slots.end()) {
            slots.push_back(*pattern);
        }
        return;
    }
    switch (inst->GetOpcode()) {
    case IR::Opcode::Phi:
        for (size_t arg = 0; arg < inst->NumArgs(); ++arg) {
            CollectPointerSlots(inst->Arg(arg), slots, visited);
        }
        break;
    case IR::Opcode::SelectU32:
        CollectPointerSlots(inst->Arg(1), slots, visited);
        CollectPointerSlots(inst->Arg(2), slots, visited);
        break;
    default:
        break;
    }
}

/// The 64-bit constant buffer word at (index, offset)
u64 ReadCbufU64(Environment& env, u32 index, u32 offset) {
    return u64{env.ReadCbufValue(index, offset)} | (u64{env.ReadCbufValue(index, offset + 4)} << 32);
}

/// Whether a 64-bit word looks like a GPU address: below 2^40, above 4 GiB and 4-byte aligned
/// (floats and small integers are not)
bool IsPointerValue(u64 value) {
    return (value >> 40) == 0 && (value >> 32) != 0 && Common::IsAligned(value, 4);
}

/// A global address whose low word is a pointer read from a constant buffer array at a run-time
/// offset (one entry per object type; Ocarina of Time's GPU culling adds its per-type task counts
/// through c4[(type << 4) + 0xe0]): the array slots that can hold pointers, following additions,
/// phis and selects of the address. Empty when the address is not of that form.
small_vector<StorageBufferAddr, 16> DynamicArrayPointers(const IR::U32& low_addr,
                                                          Environment& env) {
    small_vector<StorageBufferAddr, 16> result;
    if (low_addr.IsImmediate()) {
        return result;
    }
    small_vector<const IR::Inst*, 16> pending{low_addr.InstRecursive()};
    std::unordered_set<const IR::Inst*> visited{pending.front()};
    const auto enqueue = [&](const IR::Value& value) {
        if (!value.IsImmediate() && pending.size() < 64 &&
            visited.insert(value.InstRecursive()).second) {
            pending.push_back(value.InstRecursive());
        }
    };
    for (size_t step = 0; step < pending.size(); ++step) {
        const IR::Inst* const inst{pending[step]};
        switch (inst->GetOpcode()) {
        case IR::Opcode::GetCbufU32:
        case IR::Opcode::GetCbufU32x2: {
            if (!inst->Arg(0).IsImmediate() || inst->Arg(1).IsImmediate()) {
                break;
            }
            static constexpr u32 max_slots{16};
            const u32 index{inst->Arg(0).U32()};
            small_vector<std::pair<u32, u32>, 4> slots;
            std::unordered_set<const IR::Inst*> slot_visited;
            CollectPointerSlots(inst->Arg(1), slots, slot_visited);
            for (const auto& [first, stride] : slots) {
                // The words are those of the program's first use. An array with a pointer then
                // is a pointer array: its null slots (object types not in the scene yet) can hold
                // pointers later, and the binding skips the slots that are null at the time. The
                // other arrays are offsets and the like the address adds to the pointer.
                small_vector<StorageBufferAddr, 16> array;
                bool has_pointer{};
                for (u32 slot = 0; slot < (stride ? max_slots : 1U); ++slot) {
                    const StorageBufferAddr pointer{.index = index, .offset = first + slot * stride};
                    if (!Common::IsAligned(pointer.offset, 8)) {
                        continue;
                    }
                    const u64 value{ReadCbufU64(env, pointer.index, pointer.offset)};
                    if (value != 0 && !IsPointerValue(value)) {
                        continue;
                    }
                    has_pointer |= value != 0;
                    array.push_back(pointer);
                }
                if (!has_pointer) {
                    continue;
                }
                for (const StorageBufferAddr& pointer : array) {
                    if (std::ranges::find(result, pointer) == result.end()) {
                        result.push_back(pointer);
                    }
                }
            }
            break;
        }
        case IR::Opcode::IAdd32:
        case IR::Opcode::Phi:
            for (size_t arg = 0; arg < inst->NumArgs(); ++arg) {
                enqueue(inst->Arg(arg));
            }
            break;
        case IR::Opcode::SelectU32:
            enqueue(inst->Arg(1));
            enqueue(inst->Arg(2));
            break;
        case IR::Opcode::CompositeExtractU32x2:
            if (inst->Arg(1).IsImmediate() && inst->Arg(1).U32() == 0) {
                enqueue(inst->Arg(0));
            }
            break;
        default:
            break;
        }
    }
    return result;
}

/// A pointer read from a table whose own pointer comes from an array indexed at run time (Ocarina
/// of Time's culling reads each object type's instance transforms this way): the entry in the
/// table of every array slot. The binding reads the entry through the slot at bind time.
small_vector<StorageBufferAddr, 16> DynamicTableEntries(const TableLoad& load,
                                                         Environment& env) {
    if (load.indirect < 0 || !Common::IsAligned(static_cast<u64>(load.indirect), 8)) {
        return {};
    }
    small_vector<StorageBufferAddr, 16> entries{DynamicArrayPointers(load.table.value, env)};
    for (StorageBufferAddr& entry : entries) {
        entry.indirect = static_cast<u32>(load.indirect);
    }
    return entries;
}

/// Records buffers only reached through the global memory table
void AddGlobalPointers(std::span<const StorageBufferAddr> pointers, const IR::Inst& inst,
                       StorageInfo& info) {
    for (const StorageBufferAddr& pointer : pointers) {
        if (std::ranges::find(info.global_pointers, pointer) == info.global_pointers.end()) {
            info.global_pointers.push_back(pointer);
        }
        if (IsGlobalMemoryWrite(inst)) {
            info.global_writes.insert(pointer);
        }
    }
}

/// Collects the storage buffer used by a global memory instruction and the instruction itself
void CollectStorageBuffers(IR::Block& block, IR::Inst& inst, StorageInfo& info, Environment& env,
                           Stage stage) {
    // NVN puts storage buffers in a specific range, we have to bias towards these addresses to
    // avoid getting false positives
    static constexpr Bias nvn_bias{
        .index = 0,
        .offset_begin = 0x100,
        .offset_end = 0x700,
        .alignment = 16,
    };
    // Track the low address of the instruction
    const std::optional<LowAddrInfo> low_addr_info{TrackLowAddress(&inst)};
    if (!low_addr_info) {
        // Failed to track the low address, use NVN fallbacks
        return;
    }
    // First try to find storage buffers in the NVN address
    const IR::U32 low_addr{low_addr_info->value};
    const auto cached_track = [&](const Bias* bias, auto& cache) {
        if (low_addr.IsImmediate()) {
            return Track(low_addr, bias);
        }
        const IR::Inst* const start{low_addr.InstRecursive()};
        if (const auto it = cache.find(start); it != cache.end()) {
            return it->second;
        }
        const std::optional<StorageBufferAddr> result{Track(low_addr, bias)};
        cache.emplace(start, result);
        return result;
    };
    std::optional<StorageBufferAddr> storage_buffer{TrackPointerPair(*low_addr_info)};
    if (!storage_buffer && low_addr_info->high && IsLoadedWord(*low_addr_info->high)) {
        // The pointer itself was read from memory: the address it was read from says nothing
        // about where it points (Donkey Kong Bananza stores through pointers it loads from a list
        // at c0[0x110], which tracking would take for offsets into the list). Leave it to the
        // global memory fallbacks, with the table entry bound when the table is at a fixed place.
        // Compute programs register the entries without a binding; graphics programs bind them.
        for (const TableLoad& load : FindTableLoads(*low_addr_info)) {
            const std::optional<StorageBufferAddr> entry{TrackTableEntry(load)};
            if (stage == Stage::Compute) {
                if (entry) {
                    AddGlobalPointers(std::span{&*entry, 1}, inst, info);
                } else {
                    AddGlobalPointers(DynamicTableEntries(load, env), inst, info);
                }
            } else if (entry && (info.set.contains(*entry) || info.set.size() < Info::MAX_SSBOS)) {
                info.set.insert(*entry);
                if (IsGlobalMemoryWrite(inst)) {
                    info.writes.insert(*entry);
                }
            }
        }
        LOG_INFO(Shader,
                 "Global memory address is a pointer read from memory, using global fallbacks");
        return;
    }
    if (!storage_buffer && stage == Stage::Compute) {
        // A pointer from an array indexed at run time: the access stays global and finds its
        // buffer through the global memory table, where the slots' buffers are registered.
        // Tracking would pick some other word the index is computed from.
        const small_vector<StorageBufferAddr, 16> slots{DynamicArrayPointers(low_addr, env)};
        if (!slots.empty()) {
            AddGlobalPointers(slots, inst, info);
            LOG_INFO(Shader, "Global memory address is a pointer from an array in cbuf, {} slots",
                     slots.size());
            return;
        }
    }
    if (!storage_buffer) {
        storage_buffer = cached_track(&nvn_bias, info.biased_tracks);
    }
    if (!storage_buffer) {
        // If it fails, track without a bias
        storage_buffer = cached_track(nullptr, info.unbiased_tracks);
        if (!storage_buffer) {
            // If that also fails, use NVN fallbacks
            LOG_WARNING(Shader, "Storage buffer failed to track, using global memory fallbacks");
            return;
        }
        LOG_WARNING(Shader, "Storage buffer tracked without bias, index {} offset {}",
                    storage_buffer->index, storage_buffer->offset);
    }
    // Collect storage buffer and the instruction
    if (IsGlobalMemoryWrite(inst)) {
        info.writes.insert(*storage_buffer);
    }
    info.set.insert(*storage_buffer);
    info.to_replace.push_back(StorageInst{
        .storage_buffer{*storage_buffer},
        .inst = &inst,
        .block = &block,
    });
}

/// Returns the offset in indices (not bytes) for an equivalent storage instruction
IR::U32 StorageOffset(IR::Block& block, IR::Inst& inst, StorageBufferAddr buffer, u32 alignment) {
    IR::IREmitter ir{block, IR::Block::InstructionList::s_iterator_to(inst)};
    IR::U32 offset;
    if (const std::optional<LowAddrInfo> low_addr{TrackLowAddress(&inst)}) {
        offset = low_addr->value;
        if (low_addr->imm_offset != 0) {
            offset = ir.IAdd(offset, ir.Imm32(low_addr->imm_offset));
        }
    } else {
        offset = ir.UConvert(32, IR::U64{inst.Arg(0)});
    }
    // Subtract the least significant 32 bits from the guest offset. The result is the storage
    // buffer offset in bytes.
    IR::U32 low_cbuf{ir.GetCbuf(ir.Imm32(buffer.index), ir.Imm32(buffer.offset))};

    // Align the offset base to match the host alignment requirements
    low_cbuf = ir.BitwiseAnd(low_cbuf, ir.Imm32(~(alignment - 1U)));
    return ir.ISub(offset, low_cbuf);
}

/// Replace a global memory load instruction with its storage buffer equivalent
void ReplaceLoad(IR::Block& block, IR::Inst& inst, const IR::U32& storage_index,
                 const IR::U32& offset) {
    const IR::Opcode new_opcode{GlobalToStorage(inst.GetOpcode())};
    const auto it{IR::Block::InstructionList::s_iterator_to(inst)};
    const IR::Value value{&*block.PrependNewInst(it, new_opcode, {storage_index, offset})};
    inst.ReplaceUsesWith(value);
}

/// Replace a global memory write instruction with its storage buffer equivalent
void ReplaceWrite(IR::Block& block, IR::Inst& inst, const IR::U32& storage_index,
                  const IR::U32& offset) {
    const IR::Opcode new_opcode{GlobalToStorage(inst.GetOpcode())};
    const auto it{IR::Block::InstructionList::s_iterator_to(inst)};
    block.PrependNewInst(it, new_opcode, {storage_index, offset, inst.Arg(1)});
    inst.Invalidate();
}

/// Replace an atomic operation on global memory instruction with its storage buffer equivalent
void ReplaceAtomic(IR::Block& block, IR::Inst& inst, const IR::U32& storage_index,
                   const IR::U32& offset) {
    const IR::Opcode new_opcode{GlobalToStorage(inst.GetOpcode())};
    const auto it{IR::Block::InstructionList::s_iterator_to(inst)};
    const IR::Value value{
        &*block.PrependNewInst(it, new_opcode, {storage_index, offset, inst.Arg(1)})};
    inst.ReplaceUsesWith(value);
}

/// Replace a global memory instruction with its storage buffer equivalent
void Replace(IR::Block& block, IR::Inst& inst, const IR::U32& storage_index,
             const IR::U32& offset) {
    switch (inst.GetOpcode()) {
    case IR::Opcode::LoadGlobalS8:
    case IR::Opcode::LoadGlobalU8:
    case IR::Opcode::LoadGlobalS16:
    case IR::Opcode::LoadGlobalU16:
    case IR::Opcode::LoadGlobal32:
    case IR::Opcode::LoadGlobal64:
    case IR::Opcode::LoadGlobal128:
        return ReplaceLoad(block, inst, storage_index, offset);
    case IR::Opcode::WriteGlobalS8:
    case IR::Opcode::WriteGlobalU8:
    case IR::Opcode::WriteGlobalS16:
    case IR::Opcode::WriteGlobalU16:
    case IR::Opcode::WriteGlobal32:
    case IR::Opcode::WriteGlobal64:
    case IR::Opcode::WriteGlobal128:
        return ReplaceWrite(block, inst, storage_index, offset);
    case IR::Opcode::GlobalAtomicIAdd32:
    case IR::Opcode::GlobalAtomicSMin32:
    case IR::Opcode::GlobalAtomicUMin32:
    case IR::Opcode::GlobalAtomicSMax32:
    case IR::Opcode::GlobalAtomicUMax32:
    case IR::Opcode::GlobalAtomicInc32:
    case IR::Opcode::GlobalAtomicDec32:
    case IR::Opcode::GlobalAtomicAnd32:
    case IR::Opcode::GlobalAtomicOr32:
    case IR::Opcode::GlobalAtomicXor32:
    case IR::Opcode::GlobalAtomicExchange32:
    case IR::Opcode::GlobalAtomicIAdd64:
    case IR::Opcode::GlobalAtomicSMin64:
    case IR::Opcode::GlobalAtomicUMin64:
    case IR::Opcode::GlobalAtomicSMax64:
    case IR::Opcode::GlobalAtomicUMax64:
    case IR::Opcode::GlobalAtomicAnd64:
    case IR::Opcode::GlobalAtomicOr64:
    case IR::Opcode::GlobalAtomicXor64:
    case IR::Opcode::GlobalAtomicExchange64:
    case IR::Opcode::GlobalAtomicIAdd32x2:
    case IR::Opcode::GlobalAtomicSMin32x2:
    case IR::Opcode::GlobalAtomicUMin32x2:
    case IR::Opcode::GlobalAtomicSMax32x2:
    case IR::Opcode::GlobalAtomicUMax32x2:
    case IR::Opcode::GlobalAtomicAnd32x2:
    case IR::Opcode::GlobalAtomicOr32x2:
    case IR::Opcode::GlobalAtomicXor32x2:
    case IR::Opcode::GlobalAtomicExchange32x2:
    case IR::Opcode::GlobalAtomicAddF32:
    case IR::Opcode::GlobalAtomicAddF16x2:
    case IR::Opcode::GlobalAtomicAddF32x2:
    case IR::Opcode::GlobalAtomicMinF16x2:
    case IR::Opcode::GlobalAtomicMinF32x2:
    case IR::Opcode::GlobalAtomicMaxF16x2:
    case IR::Opcode::GlobalAtomicMaxF32x2:
        return ReplaceAtomic(block, inst, storage_index, offset);
    default:
        throw InvalidArgument("Invalid global memory opcode {}", inst.GetOpcode());
    }
}
} // Anonymous namespace

void GlobalMemoryToStorageBufferPass(Environment& env, IR::Program& program,
                                     const HostTranslateInfo& host_info) {
    StorageInfo info;
    for (IR::Block* const block : program.post_order_blocks) {
        for (IR::Inst& inst : block->Instructions()) {
            if (!IsGlobalMemory(inst)) {
                continue;
            }
            CollectStorageBuffers(*block, inst, info, env, program.stage);
        }
    }
    // Written ones first, should there be more than fit
    std::ranges::stable_partition(info.global_pointers, [&](const StorageBufferAddr& pointer) {
        return info.global_writes.contains(pointer);
    });
    for (const StorageBufferAddr& pointer : info.global_pointers) {
        auto& descriptors{program.info.global_pointer_descriptors};
        if (info.set.contains(pointer) || descriptors.size() == descriptors.capacity()) {
            continue;
        }
        descriptors.push_back({
            .cbuf_index = pointer.index,
            .cbuf_offset = pointer.offset,
            .indirect_offset = pointer.indirect,
            .is_written = info.global_writes.contains(pointer),
        });
    }
    for (const StorageBufferAddr& storage_buffer : info.set) {
        program.info.storage_buffers_descriptors.push_back({
            .cbuf_index = storage_buffer.index,
            .cbuf_offset = storage_buffer.offset,
            .count = 1,
            .is_written = info.writes.contains(storage_buffer),
            .indirect_offset = storage_buffer.indirect,
        });
    }
    for (const StorageInst& storage_inst : info.to_replace) {
        const StorageBufferAddr storage_buffer{storage_inst.storage_buffer};
        const auto it{info.set.find(storage_inst.storage_buffer)};
        const IR::U32 index{IR::Value{static_cast<u32>(info.set.index_of(it))}};
        IR::Block* const block{storage_inst.block};
        IR::Inst* const inst{storage_inst.inst};
        const IR::U32 offset{
            StorageOffset(*block, *inst, storage_buffer, u32(host_info.min_ssbo_alignment))};
        Replace(*block, *inst, index, offset);
    }
}

void JoinStorageInfo(Info& base, Info& source) {
    auto& descriptors = base.storage_buffers_descriptors;
    for (auto& desc : source.storage_buffers_descriptors) {
        auto it{std::ranges::find_if(descriptors, [&desc](const auto& existing) {
            return desc.cbuf_index == existing.cbuf_index &&
                   desc.cbuf_offset == existing.cbuf_offset && desc.count == existing.count &&
                   desc.indirect_offset == existing.indirect_offset;
        })};
        if (it != descriptors.end()) {
            it->is_written |= desc.is_written;
            continue;
        }
        descriptors.push_back(desc);
    }
}

} // namespace Shader::Optimization
