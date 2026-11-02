// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: 2021 yuzu Emulator Project
// SPDX-FileCopyrightText: 2021 Skyline Team and Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstring>
#include <limits>
#include <utility>

#include "common/alignment.h"
#include "common/assert.h"
#include "common/common_types.h"
#include "common/logging.h"
#include "core/core.h"
#include "common/div_ceil.h"
#include "core/device_memory.h"
#include "core/hle/kernel/board/nintendo/nx/k_system_control.h"
#include "core/hle/kernel/k_memory_manager.h"
#include "core/hle/kernel/k_page_group.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/kernel.h"
#include "core/hle/service/nvdrv/devices/gpu_address_space_layout.h"
#include "core/hle/service/nvdrv/core/container.h"
#include "core/hle/service/nvdrv/core/nvmap.h"
#include "core/hle/service/nvdrv/devices/ioctl_serialization.h"
#include "core/hle/service/nvdrv/devices/nvhost_as_gpu.h"
#include "core/hle/service/nvdrv/devices/native_gpu_kind.h"
#include "core/hle/service/nvdrv/devices/nvhost_gpu.h"
#include "core/hle/service/nvdrv/nvdrv.h"
#include "video_core/control/channel_state.h"
#include "video_core/gpu.h"
#include "video_core/host1x/host1x.h"
#include "video_core/memory_manager.h"
#include "video_core/rasterizer_interface.h"

namespace Service::Nvidia::Devices {

nvhost_as_gpu::nvhost_as_gpu(Core::System& system_, Module& module_, NvCore::Container& core)
    : nvdevice{system_, Type::GpuAddressSpace}, module{module_}, container{core}, nvmap{core.GetNvMapFile()}, vm{},
      gmmu{} {}

namespace {
// The address spaces with GPU-only regions, by device range, for the device memory manager's
// unmapped-write handler
struct GpuOnlyRanges {
    std::mutex mutex;
    std::map<DAddr, std::pair<u64, nvhost_as_gpu*>> ranges; ///< Device address -> (size, owner)
    bool handler_set{};
};
GpuOnlyRanges& GetGpuOnlyRanges() {
    static GpuOnlyRanges ranges;
    return ranges;
}
constexpr u64 GpuOnlyBlockSize = 4ULL << 20;

// Blocks of the device memory after the guest's DRAM (GpuOnlyMemorySize), shared by all address
// spaces
struct GpuOnlyPool {
    std::mutex mutex;
    u64 next = Kernel::Board::Nintendo::Nx::KSystemControl::Init::GetIntendedMemorySize();
    std::vector<u64> free;
};
GpuOnlyPool& GetGpuOnlyPool() {
    static GpuOnlyPool pool;
    return pool;
}

/// Offset in device memory of a free block, or 0 when there is none
u64 AllocateGpuOnlyBlock() {
    auto& pool = GetGpuOnlyPool();
    const std::scoped_lock lock{pool.mutex};
    if (!pool.free.empty()) {
        const u64 offset = pool.free.back();
        pool.free.pop_back();
        return offset;
    }
    const u64 end = Kernel::Board::Nintendo::Nx::KSystemControl::Init::GetIntendedMemorySize() +
                    Core::GpuOnlyMemorySize;
    if (pool.next + GpuOnlyBlockSize > end) {
        return 0;
    }
    const u64 offset = pool.next;
    pool.next += GpuOnlyBlockSize;
    return offset;
}

void FreeGpuOnlyBlock(u64 offset) {
    auto& pool = GetGpuOnlyPool();
    const std::scoped_lock lock{pool.mutex};
    pool.free.push_back(offset);
}

bool BackGpuOnlyWrite(DAddr address, size_t size) {
    auto& gpu_only = GetGpuOnlyRanges();
    nvhost_as_gpu* owner{};
    {
        const std::scoped_lock lock{gpu_only.mutex};
        auto it = gpu_only.ranges.upper_bound(address);
        if (it == gpu_only.ranges.begin()) {
            return false;
        }
        --it;
        if (address >= it->first + it->second.first) {
            return false;
        }
        owner = it->second.second;
    }
    return owner->BackGpuOnly(address, size);
}
} // namespace

nvhost_as_gpu::~nvhost_as_gpu() {
    if (!m_native_sparse.empty()) {
        auto& smmu = system.Host1x().MemoryManager();
        {
            auto& ranges = GetGpuOnlyRanges();
            const std::scoped_lock lock{ranges.mutex};
            for (auto& [address, region] : m_native_sparse) {
                ranges.ranges.erase(region.device);
            }
        }
        for (auto& [address, region] : m_native_sparse) {
            if (region.device == 0) {
                continue;
            }
            smmu.Unmap(region.device, region.size);
            smmu.Free(region.device, region.size);
            ReleaseGpuOnly(region);
        }
    }
    if (m_owner_process) { m_owner_process->Close(system.Kernel()); }
}

// Gives memory to the blocks of a GPU-only region that [address, +size) reaches, cleared, from
// the device memory after the guest's DRAM. Runs before a write lands there, usually on the GPU
// thread.
bool nvhost_as_gpu::BackGpuOnly(DAddr address, size_t size) {
    const std::scoped_lock lock{m_gpu_only_mutex};
    auto& smmu = system.Host1x().MemoryManager();
    bool backed = false;
    for (auto& [gpu_address, region] : m_native_sparse) {
        if (region.device == 0 || address + size <= region.device ||
            address >= region.device + region.size) {
            continue;
        }
        const u64 first = ((std::max)(address, region.device) - region.device) / GpuOnlyBlockSize;
        const u64 last = ((std::min)(address + size, region.device + region.size) - 1 -
                          region.device) / GpuOnlyBlockSize;
        for (u64 block = first; block <= last; ++block) {
            if (region.blocks[block]) {
                backed = true;
                continue;
            }
            const u64 offset = block * GpuOnlyBlockSize;
            const u64 block_size = (std::min)(GpuOnlyBlockSize, region.size - offset);
            const u64 memory = AllocateGpuOnlyBlock();
            if (memory == 0) {
                static bool warned{};
                if (!std::exchange(warned, true)) {
                    LOG_WARNING(Service_NVDRV,
                                "No memory for GPU-only region {:#x}: writes to it are dropped",
                                gpu_address);
                }
                return backed;
            }
            std::memset(system.DeviceMemory().GetPointerFromRaw<u8>(memory), 0, block_size);
            smmu.MapPhysical(region.device + offset, memory, block_size);
            region.blocks[block] = memory;
            m_gpu_only_bytes += block_size;
            backed = true;
            LOG_DEBUG(Service_NVDRV, "GPU-only region {:#x}: block {:#x} has memory, {} MiB in use",
                      gpu_address, gpu_address + offset, m_gpu_only_bytes >> 20);
        }
    }
    return backed;
}

void nvhost_as_gpu::ReleaseGpuOnly(GpuOnlyRegion& region) {
    const std::scoped_lock lock{m_gpu_only_mutex};
    for (u64 block = 0; block < region.blocks.size(); ++block) {
        if (region.blocks[block] != 0) {
            m_gpu_only_bytes -= (std::min)(GpuOnlyBlockSize, region.size - block * GpuOnlyBlockSize);
            FreeGpuOnlyBlock(region.blocks[block]);
            region.blocks[block] = 0;
        }
    }
}

NvResult nvhost_as_gpu::Ioctl1(DeviceFD fd, Ioctl command, std::span<const u8> input,
                               std::span<u8> output) {
    switch (command.group) {
    case 'A':
        switch (command.cmd) {
        case 0x1:
            return WrapFixed(this, &nvhost_as_gpu::BindChannel, input, output);
        case 0x2:
            return WrapFixed(this, &nvhost_as_gpu::AllocateSpace, input, output);
        case 0x3:
            return WrapFixed(this, &nvhost_as_gpu::FreeSpace, input, output);
        case 0x5:
            return WrapFixed(this, &nvhost_as_gpu::UnmapBuffer, input, output);
        case 0x6:
            return WrapFixed(this, &nvhost_as_gpu::MapBufferEx, input, output);
        case 0x8:
            return WrapFixed(this, &nvhost_as_gpu::GetVARegions1, input, output);
        case 0x9:
            return WrapFixed(this, &nvhost_as_gpu::AllocAsEx, input, output);
        case 0xb:
            return WrapFixed(this, &nvhost_as_gpu::GetSyncpointShimInfo, input, output);
        case 0xc:
            return WrapFixed(this, &nvhost_as_gpu::ModifyKind, input, output);
        case 0x14:
            return WrapVariable(this, &nvhost_as_gpu::Remap, input, output);
        default:
            break;
        }
        break;
    default:
        break;
    }

    UNIMPLEMENTED_MSG("Unimplemented ioctl={:08X}", command.raw);
    return NvResult::NotImplemented;
}

NvResult nvhost_as_gpu::Ioctl2(DeviceFD fd, Ioctl command, std::span<const u8> input,
                               std::span<const u8> inline_input, std::span<u8> output) {
    UNIMPLEMENTED_MSG("Unimplemented ioctl={:08X}", command.raw);
    return NvResult::NotImplemented;
}

NvResult nvhost_as_gpu::Ioctl3(DeviceFD fd, Ioctl command, std::span<const u8> input,
                               std::span<u8> output, std::span<u8> inline_output) {
    switch (command.group) {
    case 'A':
        switch (command.cmd) {
        case 0x8:
            return WrapFixedInlOut(this, &nvhost_as_gpu::GetVARegions3, input, output,
                                   inline_output);
        default:
            break;
        }
        break;
    default:
        break;
    }
    UNIMPLEMENTED_MSG("Unimplemented ioctl={:08X}", command.raw);
    return NvResult::NotImplemented;
}

void nvhost_as_gpu::OnOpen(NvCore::SessionId session_id, DeviceFD fd) {
    m_fd = fd;
    if (const auto* session = container.GetSession(session_id)) {
        m_owner_process = session->process;
        m_owner_process->Open(system.Kernel());
        m_extended_abi = m_owner_process->UsesPAuthReturns();
    }
}
void nvhost_as_gpu::OnClose(DeviceFD fd) {}

NvResult nvhost_as_gpu::MapNative4K(u32 handle_id, u64 address, u64 buffer_offset, u64 size, u8 kind) {
    if (!m_extended_abi || !size || !Common::IsAligned(address, VM::YUZU_PAGESIZE) ||
        !Common::IsAligned(size, VM::YUZU_PAGESIZE) ||
        !Common::IsAligned(buffer_offset, VM::YUZU_PAGESIZE)) {
        return NvResult::BadParameter;
    }
    const auto handle = nvmap.GetHandle(handle_id);
    if (!handle || buffer_offset > handle->orig_size || size > handle->orig_size - buffer_offset) {
        return NvResult::BadParameter;
    }
    {
        const std::scoped_lock lock{mutex};
        if (!vm.initialised || address < vm.va_range_start || address >= vm.va_range_split ||
            size > vm.va_range_split - address) {
            return NvResult::BadParameter;
        }
        const auto next = allocation_map.lower_bound(address);
        if ((next != allocation_map.end() && next->first < address + size) ||
            (next != allocation_map.begin() &&
             std::prev(next)->first + std::prev(next)->second.size > address)) {
            return NvResult::BadParameter;
        }
    }
    IoctlAllocSpace reservation{};
    reservation.pages = static_cast<u32>(size >> VM::PAGE_SIZE_BITS);
    reservation.page_size = VM::YUZU_PAGESIZE;
    reservation.flags = MappingFlags::Fixed;
    reservation.offset = address;
    const auto reserved = AllocateSpace(reservation);
    if (reserved != NvResult::Success) { return reserved; }
    IoctlMapBufferEx mapping{};
    mapping.flags = MappingFlags::Fixed;
    mapping.kind = kind;
    mapping.handle = handle_id;
    mapping.page_size = VM::YUZU_PAGESIZE;
    mapping.buffer_offset = static_cast<s64>(buffer_offset);
    mapping.mapping_size = size;
    mapping.offset = static_cast<s64>(address);
    const auto result = MapBufferEx(mapping);
    if (result != NvResult::Success) {
        nvmap.UnpinHandle(handle_id);
        IoctlFreeSpace release{};
        release.offset = address;
        release.pages = reservation.pages;
        release.page_size = VM::YUZU_PAGESIZE;
        FreeSpace(release);
    }
    return result;
}

u64 nvhost_as_gpu::MapNativeImage(u32 handle_id, u64 buffer_offset, u64 size, u8 kind) {
    if (!size || !Common::IsAligned(buffer_offset, VM::YUZU_PAGESIZE)) {
        return 0;
    }
    IoctlMapBufferEx mapping{};
    mapping.kind = kind;
    mapping.handle = handle_id;
    mapping.page_size = VM::YUZU_PAGESIZE;
    mapping.buffer_offset = static_cast<s64>(buffer_offset);
    mapping.mapping_size = Common::AlignUp(size, VM::YUZU_PAGESIZE);
    return MapBufferEx(mapping) == NvResult::Success ? static_cast<u64>(mapping.offset) : 0;
}

NvResult nvhost_as_gpu::UnmapNativeImage(u64 address) {
    IoctlUnmapBuffer release{};
    release.offset = static_cast<s64>(address);
    return UnmapBuffer(release);
}

NvResult nvhost_as_gpu::UnmapNative4K(u64 address, u64 size) {
    if (!m_extended_abi || !size || !Common::IsAligned(address, VM::YUZU_PAGESIZE) ||
        !Common::IsAligned(size, VM::YUZU_PAGESIZE)) {
        return NvResult::BadParameter;
    }
    IoctlFreeSpace release{};
    release.offset = address;
    release.pages = static_cast<u32>(size >> VM::PAGE_SIZE_BITS);
    release.page_size = VM::YUZU_PAGESIZE;
    const auto result = FreeSpace(release);
    if (result == NvResult::Success) {
        // Pages unmapped inside a sparse region return to it.
        const std::scoped_lock lock{mutex};
        const auto region = m_native_sparse.upper_bound(address);
        if (region != m_native_sparse.begin() &&
            address + size - std::prev(region)->first <= std::prev(region)->second.size) {
            const auto& [start, gpu_only] = *std::prev(region);
            if (gpu_only.device != 0) {
                gmmu->Map(address, gpu_only.device + (address - start), size,
                          Tegra::PTEKind::INVALID, false);
            } else {
                gmmu->MapSparse(address, size, false);
            }
        }
    }
    return result;
}

NvResult nvhost_as_gpu::CreateNativeSparse(u64 address, u64 size) {
    if (!m_extended_abi || !size || !Common::IsAligned(address, VM::YUZU_PAGESIZE) ||
        !Common::IsAligned(size, VM::YUZU_PAGESIZE)) {
        return NvResult::BadParameter;
    }
    const std::scoped_lock lock{mutex};
    if (!vm.initialised || address < vm.va_range_start || address >= vm.va_range_split ||
        size > vm.va_range_split - address) {
        return NvResult::BadParameter;
    }
    const auto next = m_native_sparse.lower_bound(address);
    if ((next != m_native_sparse.end() && next->first < address + size) ||
        (next != m_native_sparse.begin() &&
         std::prev(next)->first + std::prev(next)->second.size > address)) {
        return NvResult::BadParameter;
    }
    auto& smmu = system.Host1x().MemoryManager();
    const DAddr device = smmu.Allocate(size);
    if (device == 0) {
        // Without a device range the region reads as zero and drops writes
        LOG_ERROR(Service_NVDRV, "No device address space for GPU-only region {:#x}+{:#x}", address,
                  size);
        gmmu->MapSparse(address, size, false);
        m_native_sparse[address] = GpuOnlyRegion{size, 0, {}};
        return NvResult::Success;
    }
    {
        const std::scoped_lock gpu_only_lock{m_gpu_only_mutex};
        auto& region = m_native_sparse[address];
        region.size = size;
        region.device = device;
        region.blocks.resize(Common::DivCeil(size, GpuOnlyBlockSize));
    }
    {
        auto& ranges = GetGpuOnlyRanges();
        const std::scoped_lock ranges_lock{ranges.mutex};
        ranges.ranges[device] = {size, this};
        if (!std::exchange(ranges.handler_set, true)) {
            smmu.SetUnmappedWriteHandler(&BackGpuOnlyWrite);
        }
    }
    // Only the gaps between existing mappings reach the region's device range.
    u64 gap = address;
    if (auto it = allocation_map.lower_bound(address); it != allocation_map.begin()) {
        gap = (std::clamp)(std::prev(it)->first + std::prev(it)->second.size, address, address + size);
    }
    for (auto it = allocation_map.lower_bound(address);
         it != allocation_map.end() && it->first < address + size; ++it) {
        if (it->first > gap) {
            gmmu->Map(gap, device + (gap - address), it->first - gap, Tegra::PTEKind::INVALID, false);
        }
        gap = (std::max)(gap, it->first + it->second.size);
    }
    if (gap < address + size) {
        gmmu->Map(gap, device + (gap - address), address + size - gap, Tegra::PTEKind::INVALID,
                  false);
    }
    return NvResult::Success;
}

NvResult nvhost_as_gpu::DestroyNativeSparse(u64 address, u64 size) {
    const std::scoped_lock lock{mutex};
    const auto region = m_native_sparse.find(address);
    if (region == m_native_sparse.end() || region->second.size != size) {
        return NvResult::BadParameter;
    }
    u64 gap = address;
    if (auto it = allocation_map.lower_bound(address); it != allocation_map.begin()) {
        gap = (std::clamp)(std::prev(it)->first + std::prev(it)->second.size, address, address + size);
    }
    for (auto it = allocation_map.lower_bound(address);
         it != allocation_map.end() && it->first < address + size; ++it) {
        if (it->first > gap) {
            gmmu->Unmap(gap, it->first - gap);
        }
        gap = (std::max)(gap, it->first + it->second.size);
    }
    if (gap < address + size) {
        gmmu->Unmap(gap, address + size - gap);
    }
    auto& smmu = system.Host1x().MemoryManager();
    {
        auto& ranges = GetGpuOnlyRanges();
        const std::scoped_lock ranges_lock{ranges.mutex};
        ranges.ranges.erase(region->second.device);
    }
    if (region->second.device != 0) {
        smmu.Unmap(region->second.device, size);
        smmu.Free(region->second.device, size);
        ReleaseGpuOnly(region->second);
    }
    {
        const std::scoped_lock gpu_only_lock{m_gpu_only_mutex};
        m_native_sparse.erase(region);
    }
    return NvResult::Success;
}


NvResult nvhost_as_gpu::AllocAsEx(IoctlAllocAsEx& params) {
    LOG_DEBUG(Service_NVDRV, "called, flags={:#x}, as_fd={}, big_page_size={:#x}, reserved={:#x}, VA={:#x}/{:#x}/{:#x}",
              params.flags, params.as_fd, params.big_page_size, params.reserved,
              params.va_range_start, params.va_range_split, params.va_range_end);

    std::scoped_lock lock(mutex);

    if (vm.initialised) {
        ASSERT_MSG(false, "Cannot initialise an address space twice!");
        return NvResult::InvalidState;
    }

    std::shared_ptr<nvhost_as_gpu> shared_parent;
    if (m_extended_abi && params.as_fd != 0) {
        // A view of the parent's small-page region. Flags 5 leave the split at zero; flags 0 put
        // it at the end of the range (Welcome Tour) or inside it (Minecraft, 0x400000000 of
        // 0x2000000000). With a 4 KiB "big" page size both sides of the split hold 4 KiB pages,
        // so the view is treated as one small-page region whatever the split.
        const bool split_in_range = params.va_range_split <= params.va_range_end;
        if (params.as_fd == m_fd || (params.flags != 0 && params.flags != 5) ||
            params.big_page_size != 0x1000 || params.reserved != 0 || !split_in_range) {
            LOG_WARNING(Service_NVDRV, "Unsupported shared address space flags={:#x}, as_fd={}, split={:#x}",
                        params.flags, params.as_fd, params.va_range_split);
            return NvResult::BadParameter;
        }
        const auto device = module.GetDevice<nvdevice>(params.as_fd);
        if (!device || device->GetType() != Type::GpuAddressSpace) { return NvResult::BadParameter; }
        shared_parent = std::static_pointer_cast<nvhost_as_gpu>(device);
        if (!shared_parent->m_extended_abi || !shared_parent->vm.initialised ||
            shared_parent->m_owner_process != m_owner_process ||
            params.va_range_start < shared_parent->vm.va_range_start ||
            params.va_range_start >= params.va_range_end ||
            params.va_range_end > shared_parent->vm.va_range_split) { return NvResult::BadParameter; }
    }

    const u32 big_page_size = GetGpuAddressSpaceBigPageSize(m_extended_abi, params.big_page_size);
    if (big_page_size) {
        if (!std::has_single_bit(big_page_size)) {
            LOG_ERROR(Service_NVDRV, "Non power-of-2 big page size: {:#x}!", big_page_size);
            return NvResult::BadValue;
        }

        if ((big_page_size & VM::SUPPORTED_BIG_PAGE_SIZES) == 0) {
            LOG_ERROR(Service_NVDRV, "Unsupported big page size: {:#x}!", big_page_size);
            return NvResult::BadValue;
        }

        vm.big_page_size = big_page_size;
        vm.big_page_size_bits = static_cast<u32>(std::countr_zero(big_page_size));

        vm.va_range_start = big_page_size << VM::VA_START_SHIFT;
    }

    if (m_extended_abi && !params.va_range_start) {
        const auto layout = GetGpuAddressSpaceLayout(true);
        vm.va_range_start = layout.start;
        vm.va_range_split = layout.split;
        vm.va_range_end = layout.end;
    }

    // If this is unspecified then default values should be used
    if (params.va_range_start) {
        vm.va_range_start = params.va_range_start;
        vm.va_range_split = params.va_range_split;
        vm.va_range_end = params.va_range_end;
    }

    if (shared_parent) {
        // Native flags 5 expose the parent's small-page region as a separate
        // reservation view. GPU mappings must resolve through the same page table.
        vm.va_range_split = params.va_range_end;
    }

    const u64 max_big_page_bits = Common::Log2Ceil(vm.va_range_end);

    const auto start_pages{static_cast<u32>(vm.va_range_start >> VM::PAGE_SIZE_BITS)};
    const auto end_pages{static_cast<u32>(vm.va_range_split >> VM::PAGE_SIZE_BITS)};
    vm.small_page_allocator.emplace(start_pages, end_pages);

    const auto start_big_pages{static_cast<u32>(vm.va_range_split >> vm.big_page_size_bits)};
    const auto end_big_pages{
        static_cast<u32>(vm.va_range_end >> vm.big_page_size_bits)};
    vm.big_page_allocator.emplace(start_big_pages, end_big_pages);

    if (shared_parent) {
        m_shared_parent = std::move(shared_parent);
        gmmu = m_shared_parent->gmmu;
        LOG_DEBUG(Service_NVDRV, "native AS view fd={} shares page table with fd={}", m_fd, params.as_fd);
    } else {
        gmmu = std::make_shared<Tegra::MemoryManager>(system, max_big_page_bits, vm.va_range_split, vm.big_page_size_bits, VM::PAGE_SIZE_BITS);
        system.GPU().InitAddressSpace(*gmmu);
    }
    vm.initialised = true;

    return NvResult::Success;
}

NvResult nvhost_as_gpu::AllocateSpace(IoctlAllocSpace& params) {
    LOG_DEBUG(Service_NVDRV, "called, pages={:X}, page_size={:X}, flags={:X}", params.pages,
              params.page_size, params.flags);

    std::scoped_lock lock(mutex);

    if (!vm.initialised) {
        return NvResult::BadValue;
    }

    if (params.page_size != VM::YUZU_PAGESIZE && params.page_size != vm.big_page_size) {
        return NvResult::BadValue;
    }

    if (params.page_size != vm.big_page_size &&
        ((params.flags & MappingFlags::Sparse) != MappingFlags::None)) {
        UNIMPLEMENTED_MSG("Sparse small pages are not implemented!");
        return NvResult::NotImplemented;
    }

    const u32 page_size_bits{params.page_size == VM::YUZU_PAGESIZE ? VM::PAGE_SIZE_BITS
                                                                   : vm.big_page_size_bits};

    auto& allocator{params.page_size == VM::YUZU_PAGESIZE ? *vm.small_page_allocator
                                                          : *vm.big_page_allocator};

    if ((params.flags & MappingFlags::Fixed) != MappingFlags::None) {
        allocator.AllocateFixed(static_cast<u32>(params.offset >> page_size_bits), params.pages);
    } else {
        params.offset = static_cast<u64>(allocator.Allocate(params.pages)) << page_size_bits;
        if (!params.offset) {
            ASSERT_MSG(false, "Failed to allocate free space in the GPU AS!");
            return NvResult::InsufficientMemory;
        }
    }

    u64 size{static_cast<u64>(params.pages) * params.page_size};

    if ((params.flags & MappingFlags::Sparse) != MappingFlags::None) {
        gmmu->MapSparse(params.offset, size);
    }

    allocation_map[params.offset] = {
        .mappings{},
        .size = size,
        .page_size = params.page_size,
        .sparse = (params.flags & MappingFlags::Sparse) != MappingFlags::None,
        .big_pages = params.page_size != VM::YUZU_PAGESIZE,
    };

    return NvResult::Success;
}

bool nvhost_as_gpu::FreeMappingLocked(u64 offset) noexcept {
    if (auto const it = mapping_map.find(offset); it != mapping_map.end()) {
        auto const mapping = it->second;
        if (!mapping.fixed) {
            auto& allocator{mapping.big_page ? *vm.big_page_allocator : *vm.small_page_allocator};
            u32 page_size_bits{mapping.big_page ? vm.big_page_size_bits : VM::PAGE_SIZE_BITS};
            u32 page_size{mapping.big_page ? vm.big_page_size : VM::YUZU_PAGESIZE};
            u64 aligned_size{Common::AlignUp(mapping.size, page_size)};
            allocator.Free(u32(mapping.offset >> page_size_bits), u32(aligned_size >> page_size_bits));
        }
        nvmap.UnpinHandle(mapping.handle);
        // Sparse mappings shouldn't be fully unmapped, just returned to their sparse state
        // Only FreeSpace can unmap them fully
        if (mapping.sparse_alloc) {
            gmmu->MapSparse(offset, mapping.size, mapping.big_page);
        } else {
            gmmu->Unmap(offset, mapping.size);
        }
        mapping_map.erase(offset);
        map_buffer_offsets.erase(static_cast<s64>(offset));
        return true;
    }
    return false;
}

NvResult nvhost_as_gpu::FreeSpace(IoctlFreeSpace& params) {
    LOG_DEBUG(Service_NVDRV, "called, offset={:X}, pages={:X}, page_size={:X}", params.offset, params.pages, params.page_size);
    std::scoped_lock lock(mutex);
    if (!vm.initialised) {
        return NvResult::BadValue;
    }
    if (auto const it = allocation_map.find(params.offset); it != allocation_map.end()) {
        auto const allocation = it->second;
        if (allocation.page_size != params.page_size || allocation.size != (u64(params.pages) * params.page_size))
            return NvResult::BadValue;

        for (const auto mapping_offset : allocation.mappings)
            if (!FreeMappingLocked(mapping_offset))
                return NvResult::BadValue;

        // Unset sparse flag if required
        if (allocation.sparse)
            gmmu->Unmap(params.offset, allocation.size);

        auto& allocator{params.page_size == VM::YUZU_PAGESIZE ? *vm.small_page_allocator : *vm.big_page_allocator};
        u32 page_size_bits{params.page_size == VM::YUZU_PAGESIZE ? VM::PAGE_SIZE_BITS : vm.big_page_size_bits};

        allocator.Free(u32(params.offset >> page_size_bits), u32(allocation.size >> page_size_bits));
        allocation_map.erase(params.offset);
        return NvResult::Success;
    }
    return NvResult::BadValue;
}

NvResult nvhost_as_gpu::GetNativeSyncpointShim(u64& address, u32& stride, u32& count) {
    if (!m_extended_abi || !vm.initialised) { return NvResult::NotSupported; }
    address = vm.va_range_split - VM::YUZU_PAGESIZE;
    if (!m_syncpoint_shim_reserved) {
        IoctlAllocSpace allocation{};
        allocation.pages = 1;
        allocation.page_size = VM::YUZU_PAGESIZE;
        allocation.flags = MappingFlags::Fixed;
        allocation.offset = address;
        const auto result = AllocateSpace(allocation);
        if (result != NvResult::Success) { return result; }
        m_syncpoint_shim_reserved = true;
        gmmu->SetNativeSyncpointShim(address);
    }
    stride = sizeof(u32);
    count = 192;
    return NvResult::Success;
}

NvResult nvhost_as_gpu::GetSyncpointShimInfo(IoctlSyncpointShimInfo& params) {
    u64 address{};
    u32 stride{}, count{};
    const auto result = GetNativeSyncpointShim(address, stride, count);
    params.address = address;
    params.stride = stride;
    params.count = count;
    LOG_DEBUG(Service_NVDRV, "native syncpoint shim address={:#x}, stride={}, count={}, result={}",
              address, stride, count, static_cast<u32>(result));
    return result;
}

NvResult nvhost_as_gpu::ModifyNativeKind(u64 address, u64 size, u8 requested, u8 fallback) {
    IoctlModifyKind params{};
    params.compressed_kind = requested;
    params.uncompressed_kind = fallback;
    params.mapping_base = address;
    params.size = size;
    return ModifyKind(params);
}

NvResult nvhost_as_gpu::ModifyKind(IoctlModifyKind& params) {
    const u32 requested = params.compressed_kind;
    const u32 fallback = params.uncompressed_kind;
    const u64 base = params.mapping_base;
    const u64 offset = params.buffer_offset;
    const u64 size = params.size;
    LOG_DEBUG(Service_NVDRV, "native modify kind base={:#x}, offset={:#x}, size={:#x}, kinds={:#x}/{:#x}",
              base, offset, size, requested, fallback);
    if (!m_extended_abi || params.reserved != 0 || fallback > 255 ||
        (requested > 255 && requested != 0xffff) || !size ||
        offset > std::numeric_limits<u64>::max() - base ||
        !Common::IsAligned(base + offset, VM::YUZU_PAGESIZE) ||
        !Common::IsAligned(size, VM::YUZU_PAGESIZE)) {
        return NvResult::BadParameter;
    }
    const auto kind = SelectNativeMemoryKind(static_cast<u8>(requested == 0xffff ? fallback : requested),
                                             static_cast<u8>(fallback));
    if (!kind) { return NvResult::NotSupported; }
    const std::scoped_lock lock{mutex};
    const u64 address = base + offset;
    auto it = mapping_map.upper_bound(address);
    if (!vm.initialised || it == mapping_map.begin()) { return NvResult::BadParameter; }
    --it;
    const auto& mapping = it->second;
    const u64 delta = address - it->first;
    if (delta >= mapping.size || size > mapping.size - delta) { return NvResult::BadParameter; }
    gmmu->Map(address, mapping.ptr + delta, size, static_cast<Tegra::PTEKind>(kind->backend),
              mapping.big_page);
    return NvResult::Success;
}

NvResult nvhost_as_gpu::Remap(std::span<IoctlRemapEntry> entries) {
    LOG_DEBUG(Service_NVDRV, "called, num_entries={:#x}", entries.size());

    if (!vm.initialised) {
        return NvResult::BadValue;
    }

    for (const auto& entry : entries) {
        GPUVAddr virtual_address{static_cast<u64>(entry.as_offset_big_pages)
                                 << vm.big_page_size_bits};
        u64 size{static_cast<u64>(entry.big_pages) << vm.big_page_size_bits};

        auto alloc{allocation_map.upper_bound(virtual_address)};

        if (alloc-- == allocation_map.begin() ||
            (virtual_address - alloc->first) + size > alloc->second.size) {
            LOG_WARNING(Service_NVDRV, "Cannot remap into an unallocated region!");
            return NvResult::BadValue;
        }

        if (!alloc->second.sparse) {
            LOG_WARNING(Service_NVDRV, "Cannot remap a non-sparse mapping!");
            return NvResult::BadValue;
        }

        const bool use_big_pages = alloc->second.big_pages;
        if (!entry.handle) {
            gmmu->MapSparse(virtual_address, size, use_big_pages);
        } else {
            auto handle{nvmap.GetHandle(entry.handle)};
            if (!handle) {
                return NvResult::BadValue;
            }

            DAddr base = nvmap.PinHandle(entry.handle, false);
            DAddr device_address{static_cast<DAddr>(
                base + (static_cast<u64>(entry.handle_offset_big_pages) << vm.big_page_size_bits))};

            gmmu->Map(virtual_address, device_address, size,
                      static_cast<Tegra::PTEKind>(entry.kind), use_big_pages);
        }
    }

    return NvResult::Success;
}

NvResult nvhost_as_gpu::MapBufferEx(IoctlMapBufferEx& params) {
    LOG_DEBUG(Service_NVDRV,
              "called, flags={:X}, nvmap_handle={:X}, buffer_offset={}, mapping_size={}"
              ", offset={:#x}",
              params.flags, params.handle, params.buffer_offset, params.mapping_size,
              params.offset);

    std::scoped_lock lock(mutex);

    if (!vm.initialised) {
        return NvResult::BadValue;
    }

    // Remaps a subregion of an existing mapping to a different PA
    if ((params.flags & MappingFlags::Remap) != MappingFlags::None) {
        if (auto const it = mapping_map.find(params.offset); it != mapping_map.end()) {
            auto const mapping = it->second;
            if (mapping.size < params.mapping_size) {
                LOG_WARNING(Service_NVDRV, "Cannot remap a partially mapped GPU address space region: {:#x}", params.offset);
                return NvResult::BadValue;
            }
            u64 gpu_address = u64(params.offset + params.buffer_offset);
            VAddr device_address{mapping.ptr + params.buffer_offset};
            gmmu->Map(gpu_address, device_address, params.mapping_size, Tegra::PTEKind(params.kind), mapping.big_page);
            return NvResult::Success;
        } else {
            LOG_WARNING(Service_NVDRV, "Cannot remap an unmapped GPU address space region: {:#x}", params.offset);
            return NvResult::BadValue;
        }
    }

    auto handle{nvmap.GetHandle(params.handle)};
    if (!handle) {
        return NvResult::BadValue;
    }

    DAddr device_address = DAddr(nvmap.PinHandle(params.handle, false) + params.buffer_offset);
    u64 size{params.mapping_size ? params.mapping_size : handle->orig_size};

    bool big_page{[&]() {
        if (Common::IsAligned(handle->align, vm.big_page_size)) {
            return true;
        } else if (Common::IsAligned(handle->align, VM::YUZU_PAGESIZE)) {
            return false;
        } else {
            ASSERT(false);
            return false;
        }
    }()};

    if ((params.flags & MappingFlags::Fixed) != MappingFlags::None) {
        auto alloc{allocation_map.upper_bound(params.offset)};

        if (alloc-- == allocation_map.begin() ||
            (params.offset - alloc->first) + size > alloc->second.size) {
            ASSERT_MSG(false, "Cannot perform a fixed mapping into an unallocated region!");
            return NvResult::BadValue;
        }

        const bool use_big_pages = alloc->second.big_pages && big_page;
        gmmu->Map(params.offset, device_address, size, static_cast<Tegra::PTEKind>(params.kind), use_big_pages);

        alloc->second.mappings.push_back(params.offset);
        mapping_map.insert_or_assign(params.offset, Mapping(params.handle, device_address, params.offset, size, true, use_big_pages, alloc->second.sparse));
    } else {
        auto& allocator{big_page ? *vm.big_page_allocator : *vm.small_page_allocator};
        u32 page_size{big_page ? vm.big_page_size : VM::YUZU_PAGESIZE};
        u32 page_size_bits{big_page ? vm.big_page_size_bits : VM::PAGE_SIZE_BITS};

        params.offset = u64(allocator.Allocate(u32(Common::AlignUp(size, page_size) >> page_size_bits))) << page_size_bits;
        if (!params.offset) {
            ASSERT_MSG(false, "Failed to allocate free space in the GPU AS!");
            return NvResult::InsufficientMemory;
        }
        gmmu->Map(params.offset, device_address, Common::AlignUp(size, page_size), Tegra::PTEKind(params.kind), big_page);
        mapping_map.insert_or_assign(params.offset, Mapping(params.handle, device_address, params.offset, size, false, big_page, false));
    }

    map_buffer_offsets.insert(params.offset);

    return NvResult::Success;
}

NvResult nvhost_as_gpu::UnmapBuffer(IoctlUnmapBuffer& params) {
    std::scoped_lock lock(mutex);
    if (auto const offset_it = map_buffer_offsets.find(params.offset); offset_it != map_buffer_offsets.end()) {
        LOG_DEBUG(Service_NVDRV, "called, offset={:#x}", params.offset);
        if (!vm.initialised) {
            return NvResult::BadValue;
        }

        auto const it = mapping_map.find(params.offset);
        auto const mapping = it->second;
        if (!mapping.fixed) {
            auto& allocator{mapping.big_page ? *vm.big_page_allocator : *vm.small_page_allocator};
            u32 page_size_bits{mapping.big_page ? vm.big_page_size_bits : VM::PAGE_SIZE_BITS};
            allocator.Free(u32(mapping.offset >> page_size_bits), u32(mapping.size >> page_size_bits));
        }

        // Sparse mappings shouldn't be fully unmapped, just returned to their sparse state
        // Only FreeSpace can unmap them fully
        if (mapping.sparse_alloc) {
            gmmu->MapSparse(params.offset, mapping.size, mapping.big_page);
        } else {
            gmmu->Unmap(params.offset, mapping.size);
        }

        nvmap.UnpinHandle(mapping.handle);
        mapping_map.erase(params.offset);
        map_buffer_offsets.erase(params.offset);
    }
    return NvResult::Success;
}

NvResult nvhost_as_gpu::BindChannel(IoctlBindChannel& params) {
    LOG_DEBUG(Service_NVDRV, "called, fd={:X}", params.fd);

    const auto device = module.GetDevice<nvdevice>(params.fd);
    if (!device || device->GetType() != nvdevice::Type::GpuChannel) { return NvResult::BadParameter; }
    const auto gpu_channel_device = std::static_pointer_cast<nvhost_gpu>(device);
    if (!vm.initialised || !gmmu || (gpu_channel_device->channel_state->initialized &&
        gpu_channel_device->channel_state->memory_manager != gmmu.get())) { return NvResult::InvalidState; }
    gpu_channel_device->channel_state->memory_manager_owner =
        std::shared_ptr<Tegra::MemoryManager>(module.GetDevice<nvhost_as_gpu>(m_fd), gmmu.get());
    gpu_channel_device->channel_state->memory_manager = gmmu.get();
    return NvResult::Success;
}

void nvhost_as_gpu::GetVARegionsImpl(IoctlGetVaRegions& params) {
    const bool has_big_pages =
        vm.big_page_allocator->GetVALimit() > vm.big_page_allocator->GetVAStart();
    // As in nvgpu, an address space without big pages reports only its small page region.
    // Native views of a parent's small pages have none, and NVN builds an allocator for every
    // region it is given, which fails for an empty one.
    params.buf_size = (has_big_pages ? 2 : 1) * sizeof(VaRegion);

    params.regions = std::array<VaRegion, 2>{
        VaRegion{
            .offset = GpuPageAddress(vm.small_page_allocator->GetVAStart(), VM::PAGE_SIZE_BITS),
            .page_size = VM::YUZU_PAGESIZE,
            ._pad0_{},
            .pages = vm.small_page_allocator->GetVALimit() - vm.small_page_allocator->GetVAStart(),
        },
        VaRegion{
            .offset = GpuPageAddress(vm.big_page_allocator->GetVAStart(), vm.big_page_size_bits),
            .page_size = vm.big_page_size,
            ._pad0_{},
            .pages = vm.big_page_allocator->GetVALimit() - vm.big_page_allocator->GetVAStart(),
        },
    };
    if (!has_big_pages) {
        params.regions[1] = {};
    }
}

NvResult nvhost_as_gpu::GetVARegions1(IoctlGetVaRegions& params) {
    LOG_DEBUG(Service_NVDRV, "called, buf_addr={:X}, buf_size={:X}", params.buf_addr, params.buf_size);

    std::scoped_lock lock(mutex);

    if (!vm.initialised) {
        return NvResult::BadValue;
    }

    GetVARegionsImpl(params);

    return NvResult::Success;
}

NvResult nvhost_as_gpu::GetVARegions3(IoctlGetVaRegions& params, std::span<VaRegion> regions) {
    LOG_DEBUG(Service_NVDRV, "called, buf_addr={:X}, buf_size={:X}", params.buf_addr,
              params.buf_size);

    std::scoped_lock lock(mutex);

    if (!vm.initialised) {
        return NvResult::BadValue;
    }

    GetVARegionsImpl(params);

    const size_t num_regions =
        (std::min)(params.buf_size / sizeof(VaRegion), regions.size());
    for (size_t i = 0; i < num_regions; i++) {
        regions[i] = params.regions[i];
    }

    return NvResult::Success;
}

Kernel::KEvent* nvhost_as_gpu::QueryEvent(u32 event_id) {
    LOG_CRITICAL(Service_NVDRV, "Unknown AS GPU Event {}", event_id);
    return nullptr;
}

} // namespace Service::Nvidia::Devices
