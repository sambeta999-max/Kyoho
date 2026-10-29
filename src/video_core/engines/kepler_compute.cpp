// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2018 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <bitset>
#include "common/assert.h"
#include "common/logging.h"
#include "core/core.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/engines/qmd_v3.h"
#include "video_core/memory_manager.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/textures/decoders.h"

namespace Tegra::Engines {

namespace {

// AMPERE_COMPUTE_B method, as a word index
constexpr u32 SEND_SIGNALING_PCAS2_B = 0x02C0 / sizeof(u32);

// SEND_SIGNALING_PCAS2_B actions
enum class PcasAction : u32 {
    Nop = 0x0,
    Invalidate = 0x1,
    Schedule = 0x2,
    InvalidateCopySchedule = 0x3,
    IncrementPut = 0x6,
    DecrementDependence = 0x7,
    Prefetch = 0x8,
    PrefetchSchedule = 0x9,
    InvalidatePrefetchCopySchedule = 0xA,
    InvalidatePrefetchCopyForceRequireScheduling = 0xB,
    IncrementDependence = 0xC,
    IncrementCwdRefCounter = 0xD,
};

constexpr std::size_t MAX_PENDING_QMDS = 4096;

u64 Reduce(QmdV3::ReductionOp op, u64 old_value, u64 value, bool is_signed, bool is_64) {
    const u64 mask = is_64 ? ~u64{0} : u64{0xFFFFFFFF};
    const auto as_signed = [is_64](u64 x) {
        return is_64 ? static_cast<s64>(x) : s64{static_cast<s32>(static_cast<u32>(x))};
    };
    old_value &= mask;
    value &= mask;
    switch (op) {
    case QmdV3::ReductionOp::Add:
        return (old_value + value) & mask;
    case QmdV3::ReductionOp::Min:
        return is_signed ? (as_signed(old_value) < as_signed(value) ? old_value : value)
                         : (std::min)(old_value, value);
    case QmdV3::ReductionOp::Max:
        return is_signed ? (as_signed(old_value) > as_signed(value) ? old_value : value)
                         : (std::max)(old_value, value);
    case QmdV3::ReductionOp::Inc:
        return old_value >= value ? 0 : old_value + 1;
    case QmdV3::ReductionOp::Dec:
        return old_value == 0 || old_value > value ? value : old_value - 1;
    case QmdV3::ReductionOp::And:
        return old_value & value;
    case QmdV3::ReductionOp::Or:
        return old_value | value;
    case QmdV3::ReductionOp::Xor:
        return old_value ^ value;
    }
    return old_value;
}

void ReleaseSemaphore(VideoCore::RasterizerInterface& rasterizer, MemoryManager& memory_manager,
                      const QmdV3::Release& release) {
    if (!release.reduction && !release.payload_64) {
        VideoCommon::QueryPropertiesFlags flags{VideoCommon::QueryPropertiesFlags::IsAFence};
        if (release.structure_size == 0) {
            // Four words: the payload followed by a timestamp
            flags |= VideoCommon::QueryPropertiesFlags::HasTimeout;
        }
        rasterizer.Query(release.address, VideoCommon::QueryType::Payload, flags,
                         static_cast<u32>(release.payload), 0);
        return;
    }
    rasterizer.SignalFence([&memory_manager, release] {
        u64 value = release.payload;
        if (release.reduction) {
            const u64 old_value = release.payload_64 ? memory_manager.Read<u64>(release.address)
                                                     : memory_manager.Read<u32>(release.address);
            value = Reduce(*release.reduction, old_value, value, release.reduction_signed,
                           release.payload_64);
        }
        if (release.payload_64) {
            memory_manager.Write<u64>(release.address, value);
        } else {
            memory_manager.Write<u32>(release.address, static_cast<u32>(value));
        }
    });
}

} // Anonymous namespace

KeplerCompute::KeplerCompute(MemoryManager& memory_manager_)
    : memory_manager{memory_manager_}
    , upload_state{memory_manager, regs.upload}
{
    execution_mask.reset();
    execution_mask[KEPLER_COMPUTE_REG_INDEX(exec_upload)] = true;
    execution_mask[KEPLER_COMPUTE_REG_INDEX(data_upload)] = true;
    execution_mask[KEPLER_COMPUTE_REG_INDEX(launch)] = true;
    execution_mask[SEND_SIGNALING_PCAS2_B] = true;
}

KeplerCompute::~KeplerCompute() = default;

void KeplerCompute::BindRasterizer(VideoCore::RasterizerInterface* rasterizer_) {
    rasterizer = rasterizer_;
    upload_state.BindRasterizer(rasterizer);
}

void KeplerCompute::ConsumeSinkImpl(Core::System& system) {
    for (auto [method, value] : method_sink) {
        regs.reg_array[method] = value;
    }
    method_sink.clear();
}

void KeplerCompute::CallMethod(Core::System& system, u32 method, u32 method_argument, bool is_last_call) {
    ASSERT_MSG(method < Regs::NUM_REGS, "Invalid KeplerCompute register, increase the size of the Regs structure");

    regs.reg_array[method] = method_argument;

    switch (method) {
    case KEPLER_COMPUTE_REG_INDEX(exec_upload): {
        UploadInfo info{.upload_address = upload_address,
                        .exec_address = upload_state.ExecTargetAddress(),
                        .copy_size = upload_state.GetUploadSize(),
                        .was_dirty = upload_dirty};
        uploads.push_back(info);
        upload_state.ProcessExec(regs.exec_upload.linear != 0);
        break;
    }
    case KEPLER_COMPUTE_REG_INDEX(data_upload): {
        upload_address = current_dma_segment;
        upload_dirty = current_dirty;
        current_dirty = false;
        upload_state.ProcessData(method_argument, is_last_call);
        break;
    }
    case SEND_SIGNALING_PCAS2_B:
    case KEPLER_COMPUTE_REG_INDEX(launch): {
        if (method == SEND_SIGNALING_PCAS2_B && !is_ampere) {
            break;
        }
        const GPUVAddr launch_desc_loc = regs.launch_desc_loc.Address();

        for (auto& data : uploads) {
            const GPUVAddr offset = data.exec_address - launch_desc_loc;
            if (offset / sizeof(u32) == LAUNCH_REG_INDEX(grid_dim_x)) {
                const bool source_dirty = memory_manager.IsMemoryDirty(data.upload_address, data.copy_size);
                if (data.was_dirty || source_dirty) {
                    indirect_compute = {data.upload_address};
                }
            }
        }
        uploads.clear();
        if (!is_ampere) {
            ProcessLaunch();
        } else if (method == SEND_SIGNALING_PCAS2_B) {
            // The count of dependence actions is OFFSET_MINUS_ONE at [14..24) plus one
            SignalPcas(method_argument & 0xF, ((method_argument >> 14) & 0x3FF) + 1);
        } else if ((method_argument & 2) != 0) {
            // SEND_SIGNALING_PCAS_B with SCHEDULE
            SignalPcas(static_cast<u32>(PcasAction::Schedule), 1);
        }
        indirect_compute = std::nullopt;
        break;
    }
    default:
        break;
    }
}

void KeplerCompute::CallMultiMethod(Core::System& system, u32 method, const u32* base_start, u32 amount, u32 methods_pending) {
    switch (method) {
    case KEPLER_COMPUTE_REG_INDEX(data_upload):
        upload_address = current_dma_segment;
        upload_dirty = current_dirty;
        current_dirty = false;
        upload_state.ProcessData(base_start, amount);
        return;
    default:
        for (u32 i = 0; i < amount; i++) {
            CallMethod(system, method, base_start[i], methods_pending - i <= 1);
        }
        break;
    }
}

void KeplerCompute::ProcessLaunch() {
    const GPUVAddr launch_desc_loc = regs.launch_desc_loc.Address();
    memory_manager.ReadBlockUnsafe(launch_desc_loc, &launch_description,
                                   LaunchParams::NUM_LAUNCH_PARAMETERS * sizeof(u32));
    rasterizer->DispatchCompute();
}

void KeplerCompute::SignalPcas(u32 action, u32 count) {
    struct Request {
        GPUVAddr address;
        PcasAction action;
        u32 count;
    };
    // Executing a QMD can signal its dependent QMD, which is handled in the same loop
    std::vector<Request> requests{{regs.launch_desc_loc.Address(), PcasAction{action}, count}};
    while (!requests.empty()) {
        const Request request = requests.back();
        requests.pop_back();
        const auto read_qmd = [this, &request] {
            QmdV3 qmd;
            memory_manager.ReadBlockUnsafe(request.address, qmd.words.data(),
                                           sizeof(qmd.words));
            return qmd;
        };
        switch (request.action) {
        case PcasAction::Nop:
        case PcasAction::Invalidate:
        case PcasAction::IncrementPut:
        case PcasAction::Prefetch:
        case PcasAction::IncrementCwdRefCounter:
            continue;
        case PcasAction::Schedule:
        case PcasAction::InvalidateCopySchedule:
        case PcasAction::PrefetchSchedule:
        case PcasAction::InvalidatePrefetchCopySchedule:
        case PcasAction::InvalidatePrefetchCopyForceRequireScheduling: {
            auto& qmd_state = qmd_states[request.address];
            qmd_state.dependencies += static_cast<s32>(read_qmd().DependenceCounter());
            qmd_state.scheduled = true;
            break;
        }
        case PcasAction::DecrementDependence:
            qmd_states[request.address].dependencies -= static_cast<s32>(request.count);
            break;
        case PcasAction::IncrementDependence:
            qmd_states[request.address].dependencies += static_cast<s32>(request.count);
            break;
        default:
            LOG_ERROR(HW_GPU, "Unimplemented PCAS action {:#x}",
                      static_cast<u32>(request.action));
            continue;
        }
        const auto it = qmd_states.find(request.address);
        if (!it->second.scheduled || it->second.dependencies > 0) {
            if (qmd_states.size() > MAX_PENDING_QMDS) {
                LOG_ERROR(HW_GPU, "Too many QMDs waiting for dependencies, dropping them");
                qmd_states.clear();
            }
            continue;
        }
        qmd_states.erase(it);
        // Read again: releases may be patched while the QMD waits for its dependencies
        const QmdV3 qmd = read_qmd();
        ExecuteQmd(qmd);
        if (const auto dependent = qmd.GetDependent()) {
            switch (dependent->action) {
            case QmdV3::DependentAction::Schedule:
                requests.push_back({dependent->address, PcasAction::Schedule, 1});
                break;
            case QmdV3::DependentAction::InvalidateCopySchedule:
                requests.push_back({dependent->address, PcasAction::InvalidateCopySchedule, 1});
                break;
            case QmdV3::DependentAction::DecrementDependence:
                requests.push_back({dependent->address, PcasAction::DecrementDependence, 1});
                break;
            default:
                break;
            }
        }
        if (qmd.HasSecondDependent()) {
            LOG_ERROR(HW_GPU, "Second dependent QMD is not implemented");
        }
    }
}

void KeplerCompute::ExecuteQmd(const QmdV3& qmd) {
    if (qmd.MajorVersion() != 3) {
        LOG_ERROR(HW_GPU, "Unsupported QMD major version {}", qmd.MajorVersion());
        return;
    }
    if (qmd.HasWork()) {
        launch_description = qmd.ToLaunchParams();
        ampere_program_address = qmd.ProgramAddress();
        rasterizer->DispatchCompute();
    }
    for (u32 index = 0; index < QmdV3::NUM_RELEASES; ++index) {
        if (const auto release = qmd.GetRelease(index)) {
            ReleaseSemaphore(*rasterizer, memory_manager, *release);
        }
    }
}

Texture::TICEntry KeplerCompute::GetTICEntry(u32 tic_index) const {
    const GPUVAddr tic_address_gpu{regs.tic.Address() + tic_index * sizeof(Texture::TICEntry)};

    Texture::TICEntry tic_entry;
    memory_manager.ReadBlockUnsafe(tic_address_gpu, &tic_entry, sizeof(Texture::TICEntry));
    return tic_entry;
}

Texture::TSCEntry KeplerCompute::GetTSCEntry(u32 tsc_index) const {
    const GPUVAddr tsc_address_gpu{regs.tsc.Address() + tsc_index * sizeof(Texture::TSCEntry)};

    Texture::TSCEntry tsc_entry;
    memory_manager.ReadBlockUnsafe(tsc_address_gpu, &tsc_entry, sizeof(Texture::TSCEntry));
    return tsc_entry;
}

} // namespace Tegra::Engines
