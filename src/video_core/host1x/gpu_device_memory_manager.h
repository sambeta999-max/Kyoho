// SPDX-FileCopyrightText: 2023 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "core/device_memory_manager.h"

namespace VideoCore {
class RasterizerInterface;
}

namespace Tegra {

struct MaxwellDeviceMethods;

struct MaxwellDeviceTraits {
    // Switch 2 titles give GPU-only regions device ranges of their own (Mario Kart World's reach
    // several GiB each), beyond the 16 GiB a Switch 1 SMMU space holds
    static constexpr size_t device_virtual_bits = 36;
    using DeviceInterface = typename VideoCore::RasterizerInterface;
    using DeviceMethods = MaxwellDeviceMethods;
};

using MaxwellDeviceMemoryManager = Core::DeviceMemoryManager<MaxwellDeviceTraits>;

} // namespace Tegra