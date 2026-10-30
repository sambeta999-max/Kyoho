// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <cstring>
#include "common/logging.h"
#include "core/core.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/service/nvdrv/nvdrv.h"
#include "core/hle/service/nvdrv/devices/nvhost_gpu.h"
#include "core/hle/service/nvdrv/devices/nvhost_tsg_gpu.h"

namespace Service::Nvidia::Devices {

nvhost_tsg_gpu::nvhost_tsg_gpu(Core::System& system, Module& module_, NvCore::Container& core_)
    : nvdevice{system, Type::GpuTsg}, module{module_}, core{core_} {}

nvhost_tsg_gpu::~nvhost_tsg_gpu() { OnClose(0); }

void nvhost_tsg_gpu::OnOpen(NvCore::SessionId id, DeviceFD fd) {
    group_fd = fd;
    if (const auto* session = core.GetSession(id)) {
        process = session->process;
        process->Open(system.Kernel());
    }
}

void nvhost_tsg_gpu::OnClose(DeviceFD fd) {
    for (const auto& [channel_fd, channel] : channels) { channel->UnbindFromTsg(group_fd); }
    channels.clear();
    if (process) {
        process->Close(system.Kernel());
        process = nullptr;
    }
}


NvResult nvhost_tsg_gpu::Ioctl1(DeviceFD fd, Ioctl command, std::span<const u8> input,
                                std::span<u8> output) {
    std::array<u32, 8> words{};
    if (!input.empty()) { std::memcpy(words.data(), input.data(), std::min(input.size(), sizeof(words))); }
    LOG_DEBUG(Service_NVDRV, "native TSG ioctl={:#x}, input={}, output={}, words={:#x},{:#x},{:#x},{:#x},{:#x},{:#x},{:#x},{:#x}",
              command.raw, input.size(), output.size(), words[0], words[1], words[2], words[3],
              words[4], words[5], words[6], words[7]);
    if (command.group != 'T') {
        return NvResult::NotImplemented;
    }
    if (!process) {
        return NvResult::BadParameter;
    }
    switch (command.cmd) {
    case 0x1:  // BIND_CHANNEL
    case 0xb:  // BIND_CHANNEL_EX
    {
        const size_t expected_size = (command.cmd == 0xb) ? 0x18 : 0x4;
        if (input.size() < expected_size || output.size() < expected_size) {
            return NvResult::BadParameter;
        }
        const DeviceFD channel_fd = static_cast<s32>(words[0]);
        if (channels.contains(channel_fd)) {
            return NvResult::Busy;
        }
        if (channels.size() >= 64) {
            return NvResult::InsufficientMemory;
        }
        const auto device = module.GetDevice<nvdevice>(channel_fd);
        if (!device || device->GetType() != Type::GpuChannel) {
            return NvResult::BadParameter;
        }
        const auto channel = std::static_pointer_cast<nvhost_gpu>(device);
        const auto result = channel->BindToTsg(fd, process);
        if (result != NvResult::Success) {
            return result;
        }
        channels.emplace(channel_fd, channel);
        std::copy(input.begin(), input.end(), output.begin());
        return NvResult::Success;
    }
    case 0x2:  // UNBIND_CHANNEL
    {
        if (input.size() < 4 || output.size() < 4) {
            return NvResult::BadParameter;
        }
        const DeviceFD channel_fd = static_cast<s32>(words[0]);
        const auto it = channels.find(channel_fd);
        if (it == channels.end()) {
            return NvResult::BadParameter;
        }
        it->second->UnbindFromTsg(fd);
        channels.erase(it);
        std::copy(input.begin(), input.end(), output.begin());
        return NvResult::Success;
    }
    case 0x3:  // ENABLE
    case 0x4:  // DISABLE
    case 0x5:  // PREEMPT
    case 0x7:  // SET_TIMESLICE
    case 0x8:  // SET_RUNLIST_INTERLEAVE
    case 0x9:  // READ_SINGLE_SM_ERROR_STATE
    case 0xd:  // SET_L2_MAX_WAYS_EVICT_LAST
        if (!output.empty() && input.size() == output.size()) {
            std::copy(input.begin(), input.end(), output.begin());
        }
        return NvResult::Success;
    case 0xa:  // GET_TIMEDOUT
        std::fill(output.begin(), output.end(), u8{0});
        return NvResult::Success;
    default:
        return NvResult::NotImplemented;
    }
}

NvResult nvhost_tsg_gpu::Ioctl2(DeviceFD fd, Ioctl command, std::span<const u8> input,
                                std::span<const u8> inline_input, std::span<u8> output) {
    return NvResult::NotImplemented;
}

NvResult nvhost_tsg_gpu::Ioctl3(DeviceFD fd, Ioctl command, std::span<const u8> input,
                                std::span<u8> output, std::span<u8> inline_output) {
    if (command.group == 'T' && command.cmd == 0xc) {
        std::fill(output.begin(), output.end(), u8{0});
        std::fill(inline_output.begin(), inline_output.end(), u8{0});
        return NvResult::Success;
    }
    return NvResult::NotImplemented;
}

} // namespace Service::Nvidia::Devices
