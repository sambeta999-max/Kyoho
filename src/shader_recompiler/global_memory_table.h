// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>

#include "common/common_types.h"

namespace Shader {

/// The guest's global memory accesses take raw GPU addresses, which the console's MMU translates:
/// a shader can store through a pointer it loaded from memory (Donkey Kong Bananza resets the
/// indirect draw records of its GPU culling that way, one pointer per vertex). Accesses that no
/// storage buffer of the program covers look the address up in this table instead, which maps
/// guest GPU pages to the device addresses of the host buffers holding them.
///
/// Open addressing: entry GlobalMemoryTableSlot(page) and the GLOBAL_MEMORY_TABLE_PROBES after it
/// (wrapping). Entries are only ever added or changed in place, so a shader reading the table
/// while the host updates it sees an entry as it was or as it will be (the host clears `end`
/// before changing an entry and sets `tag` last when adding one).
struct GlobalMemoryTableEntry {
    u64 tag;   ///< Guest GPU page + 1; 0 for an empty entry
    u64 base;  ///< Device address of the start of the guest page in a host buffer
    u32 begin; ///< Bytes of the page the host buffer holds: [begin, end)
    u32 end;
    u32 padding[2];
};
static_assert(sizeof(GlobalMemoryTableEntry) == 32);
static_assert(offsetof(GlobalMemoryTableEntry, base) == 8);
static_assert(offsetof(GlobalMemoryTableEntry, begin) == 16);
static_assert(offsetof(GlobalMemoryTableEntry, end) == 20);

constexpr u32 GLOBAL_MEMORY_PAGE_BITS = 12;
constexpr u32 GLOBAL_MEMORY_PAGE_SIZE = 1U << GLOBAL_MEMORY_PAGE_BITS;
constexpr u32 GLOBAL_MEMORY_TABLE_PROBES = 8;
constexpr u32 GLOBAL_MEMORY_HASH_MULTIPLIER = 0x9E3779B1U;

/// First entry to probe for a guest GPU page, in a table of 2^bits entries
[[nodiscard]] constexpr u32 GlobalMemoryTableSlot(u64 page, u32 bits) {
    return ((static_cast<u32>(page) * GLOBAL_MEMORY_HASH_MULTIPLIER) ^
            static_cast<u32>(page >> 32)) >>
           (32 - bits);
}

} // namespace Shader
