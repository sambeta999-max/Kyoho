// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2018 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <bitset>
#include <cstring>
#include "common/assert.h"
#include "common/logging.h"
#include "core/core.h"
#include <cstdlib>
#include <string>

#include "video_core/engines/kepler_compute.h"
#include "video_core/gpu_split.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/engines/qmd_v3.h"
#include "video_core/engines/semaphore.h"
#include "video_core/memory_manager.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/textures/decoders.h"

namespace Tegra::Engines {

namespace {

// SEND_SIGNALING_PCAS2_B
constexpr u32 SEND_SIGNALING_PCAS2_B = 0x02C0 / sizeof(u32);
// SET_INLINE_QMD_ADDRESS_A/B and LOAD_INLINE_QMD_DATA(0..63) of AMPERE_COMPUTE_B
constexpr u32 SET_INLINE_QMD_ADDRESS_A = 0x0318 / sizeof(u32);
constexpr u32 SET_INLINE_QMD_ADDRESS_B = 0x031C / sizeof(u32);
constexpr u32 LOAD_INLINE_QMD_DATA_BEGIN = 0x0320 / sizeof(u32);
constexpr u32 LOAD_INLINE_QMD_DATA_END = LOAD_INLINE_QMD_DATA_BEGIN + 64;
// SET_REPORT_SEMAPHORE_PAYLOAD_LOWER, followed by PAYLOAD_UPPER, ADDRESS_LOWER, ADDRESS_UPPER and
// REPORT_SEMAPHORE_EXECUTE
constexpr u32 REPORT_SEMAPHORE = 0x0158 / sizeof(u32);
constexpr u32 REPORT_SEMAPHORE_EXECUTE = REPORT_SEMAPHORE + 4;
// Legacy SET_SEMAPHORE_A, SET_SEMAPHORE_B, SET_SEMAPHORE_PAYLOAD and SEMAPHORE_EXECUTE (0x1B00..0x1B0C)
constexpr u32 REPORT_SEMAPHORE_LEGACY = 0x1B00 / sizeof(u32);
constexpr u32 REPORT_SEMAPHORE_LEGACY_EXECUTE = REPORT_SEMAPHORE_LEGACY + 3;
// MME DMA methods of AMPERE_COMPUTE_B: SET_MME_MEM_ADDRESS_A to SET_MME_DATA_FIFO_CONFIG, and
// MME_DMA_WRITE_METHOD_BARRIER
constexpr u32 AMPERE_MME_DMA_BEGIN = 0x0550 / sizeof(u32);
constexpr u32 AMPERE_MME_DMA_END = 0x0578 / sizeof(u32);
constexpr u32 AMPERE_MME_DMA_WRITE_METHOD_BARRIER = 0x0DEC / sizeof(u32);
// LOAD_MME_INSTRUCTION_RAM_POINTER, LOAD_MME_INSTRUCTION_RAM, LOAD_MME_START_ADDRESS_RAM_POINTER
// and LOAD_MME_START_ADDRESS_RAM, at the offsets of the 3D class
constexpr u32 LOAD_MME_INSTRUCTION_RAM_POINTER = 0x0114 / sizeof(u32);
constexpr u32 LOAD_MME_INSTRUCTION_RAM = 0x0118 / sizeof(u32);
constexpr u32 LOAD_MME_START_ADDRESS_RAM_POINTER = 0x011C / sizeof(u32);
constexpr u32 LOAD_MME_START_ADDRESS_RAM = 0x0120 / sizeof(u32);
// CALL_MME_MACRO(j) and CALL_MME_DATA(j) of AMPERE_COMPUTE_B, from 0x3800
constexpr u32 MACRO_REGISTERS_START = 0x3800 / sizeof(u32);

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

/// What a Turing macro called on the compute class runs on
class ComputeMacroHost final : public TuringMacroEngine::Host {
public:
    ComputeMacroHost(Core::System& system_, KeplerCompute& compute_, MemoryManager& memory_manager_)
        : system{system_}, compute{compute_}, memory_manager{memory_manager_} {}

    u32 ReadState(u32 method) override {
        return method < KeplerCompute::Regs::NUM_REGS ? compute.regs.reg_array[method] : 0;
    }

    void CallMethod(u32 method, u32 value) override {
        if (method >= MACRO_REGISTERS_START) {
            LOG_ERROR(HW_GPU, "Unsupported method {:#x} emitted by a compute macro", method);
            return;
        }
        compute.CallMethod(system, method, value, true);
    }

    void ReadMemory(GPUVAddr address, std::span<u32> data) override {
        memory_manager.ReadBlock(address, data.data(), data.size_bytes());
    }

    void WriteMemory(GPUVAddr address, std::span<const u32> data) override {
        memory_manager.WriteBlock(address, data.data(), data.size_bytes());
    }

private:
    Core::System& system;
    KeplerCompute& compute;
    MemoryManager& memory_manager;
};

/// A compute macro at the front of a split GPU thread: the front's copy of the registers, with
/// method writes sent to the back
class FrontComputeMacroHost final : public TuringMacroEngine::Host {
public:
    FrontComputeMacroHost(Core::System* system_, KeplerCompute& compute_,
                          const KeplerCompute::Regs& regs_, MemoryManager& memory_manager_)
        : system{system_}, compute{compute_}, regs{regs_}, memory_manager{memory_manager_} {}

    u32 ReadState(u32 method) override {
        return method < KeplerCompute::Regs::NUM_REGS ? regs.reg_array[method] : 0;
    }

    void CallMethod(u32 method, u32 value) override {
        if (system == nullptr || method >= MACRO_REGISTERS_START) {
            LOG_ERROR(HW_GPU, "Unsupported method {:#x} emitted by a compute macro", method);
            return;
        }
        if (compute.IsFrontMethod(method)) {
            compute.FrontCallMethod(*system, method, value);
            return;
        }
        compute.FrontWrite(method, value);
        if (method == 0x6C || method == 0x6D) {
            VideoCommon::GpuSplit::Active()->PushEngineMethod(compute, method, value);
            return;
        }
        VideoCommon::GpuSplit::Active()->PushEngineStateMethod(compute, method, value);
    }

    void ReadMemory(GPUVAddr address, std::span<u32> data) override {
        VideoCommon::GpuSplit::Active()->Drain();
        memory_manager.ReadBlock(address, data.data(), data.size_bytes());
    }

    void WriteMemory(GPUVAddr address, std::span<const u32> data) override {
        VideoCommon::GpuSplit::Active()->Drain();
        memory_manager.WriteBlock(address, data.data(), data.size_bytes());
    }

private:
    Core::System* system;
    KeplerCompute& compute;
    const KeplerCompute::Regs& regs;
    MemoryManager& memory_manager;
};

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
    execution_mask[LOAD_INLINE_QMD_DATA_END - 1] = true;
    execution_mask[REPORT_SEMAPHORE_EXECUTE] = true;
    execution_mask[REPORT_SEMAPHORE_LEGACY_EXECUTE] = true;
    execution_mask[AMPERE_MME_DMA_WRITE_METHOD_BARRIER] = true;
    for (size_t method = AMPERE_MME_DMA_BEGIN; method < AMPERE_MME_DMA_END; ++method) {
        execution_mask[method] = true;
    }
    execution_mask[LOAD_MME_INSTRUCTION_RAM_POINTER] = true;
    execution_mask[LOAD_MME_INSTRUCTION_RAM] = true;
    execution_mask[LOAD_MME_START_ADDRESS_RAM_POINTER] = true;
    execution_mask[LOAD_MME_START_ADDRESS_RAM] = true;
    for (size_t method = MACRO_REGISTERS_START; method < execution_mask.size(); ++method) {
        execution_mask[method] = true;
    }
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

void KeplerCompute::EnableFrontMacros() {
    front_regs = std::make_unique<Regs>(regs);
    for (const auto& [method, value] : method_sink) {
        front_regs->reg_array[method] = value;
    }
    front_macros = true;
}

bool KeplerCompute::IsFrontMethod(u32 method) const {
    if (method >= MACRO_REGISTERS_START) {
        return true;
    }
    if ((method >= AMPERE_MME_DMA_BEGIN && method < AMPERE_MME_DMA_END) ||
        method == AMPERE_MME_DMA_WRITE_METHOD_BARRIER) {
        return true;
    }
    switch (method) {
    case LOAD_MME_INSTRUCTION_RAM_POINTER:
    case LOAD_MME_INSTRUCTION_RAM:
    case LOAD_MME_START_ADDRESS_RAM_POINTER:
    case LOAD_MME_START_ADDRESS_RAM:
        return true;
    default:
        return false;
    }
}

void KeplerCompute::FrontClassify(u32 method) {
    // What the method may write once the back runs it, from the registers before it: inline
    // uploads and reports write ranges the front knows, launches whatever their QMD names
    // (semaphore releases), the rest only registers. From Lichii e691cc3f2 (the compute
    // classification in src/video_core/engines/kepler_compute.cpp), adapted.
    using Effect = VideoCommon::GpuSplit::Effect;
    auto& split = *VideoCommon::GpuSplit::Active();
    const auto note = [&](GPUVAddr address, u64 size) {
        split.ClassifyGpuWrite(memory_manager, address, size);
    };
    if (method >= Regs::NUM_REGS || !execution_mask[method]) {
        split.Classify(Effect::None);
        return;
    }
    switch (method) {
    case KEPLER_COMPUTE_REG_INDEX(exec_upload):
        split.Classify(Effect::None);
        return;
    case KEPLER_COMPUTE_REG_INDEX(data_upload): {
        const auto [address, size] =
            Upload::DestinationRange(front_regs->upload, front_regs->exec_upload.linear != 0);
        note(address, size);
        return;
    }
    case REPORT_SEMAPHORE_EXECUTE:
        note((GPUVAddr{front_regs->reg_array[REPORT_SEMAPHORE + 3] & 0xFF} << 32) |
                 front_regs->reg_array[REPORT_SEMAPHORE + 2],
             16);
        return;
    case REPORT_SEMAPHORE_LEGACY_EXECUTE:
        note((GPUVAddr{front_regs->reg_array[REPORT_SEMAPHORE_LEGACY] & 0xFF} << 32) |
                 front_regs->reg_array[REPORT_SEMAPHORE_LEGACY + 1],
             16);
        return;
    case KEPLER_COMPUTE_REG_INDEX(launch):
    case SEND_SIGNALING_PCAS2_B:
    case LOAD_INLINE_QMD_DATA_END - 1:
        // Dispatches, with the releases of their QMD: GPU work, as the 3D engine's draws
        split.Classify(Effect::Gpu);
        return;
    default:
        split.Classify(Effect::Unknown);
        return;
    }
}

void KeplerCompute::FrontWrite(u32 method, u32 argument) {
    FrontClassify(method);
    if (method < Regs::NUM_REGS) {
        front_regs->reg_array[method] = argument;
    }
}

void KeplerCompute::FrontMultiWrite(u32 method, const u32* base_start, u32 amount) {
    // CallMultiMethod sends data uploads to the upload engine without storing them
    if (method == KEPLER_COMPUTE_REG_INDEX(data_upload)) {
        FrontClassify(method);
        return;
    }
    for (u32 i = 0; i < amount; ++i) {
        FrontWrite(method, base_start[i]);
    }
}

void KeplerCompute::FrontCallMethod(Core::System& system, u32 method, u32 argument) {
    if (method >= MACRO_REGISTERS_START) {
        FrontCallMultiMethod(system, method, &argument, 1);
        return;
    }
    front_regs->reg_array[method] = argument;
    switch (method) {
    case LOAD_MME_INSTRUCTION_RAM_POINTER:
        return TuringMacro().SetUploadPointer(argument);
    case LOAD_MME_INSTRUCTION_RAM:
        return TuringMacro().Upload(argument);
    case LOAD_MME_START_ADDRESS_RAM_POINTER:
        macro_position_pointer = argument;
        return;
    case LOAD_MME_START_ADDRESS_RAM:
        MacroPositions()[macro_position_pointer++ % MacroPositions().size()] = argument;
        return;
    default: {
        FrontComputeMacroHost host{nullptr, *this, *front_regs, memory_manager};
        TuringMacro().ProcessMethod(method, argument, host);
        return;
    }
    }
}

void KeplerCompute::FrontCallMultiMethod(Core::System& system, u32 method, const u32* base_start,
                                         u32 amount) {
    if (method < MACRO_REGISTERS_START) {
        for (u32 i = 0; i < amount; ++i) {
            FrontCallMethod(system, method, base_start[i]);
        }
        return;
    }
    if ((method - MACRO_REGISTERS_START) % 2 == 0) {
        FrontFlushMacro(system);
        executing_macro = method;
    } else if (executing_macro == 0) {
        LOG_ERROR(HW_GPU, "Compute macro data {:#x} without a macro call", method * sizeof(u32));
        return;
    }
    macro_params.insert(macro_params.end(), base_start, base_start + amount);
}

void KeplerCompute::FrontFlushMacro(Core::System& system) {
    if (executing_macro == 0) {
        return;
    }
    const u32 entry = ((executing_macro - MACRO_REGISTERS_START) >> 1) %
                      static_cast<u32>(MacroPositions().size());
    executing_macro = 0;
    FrontComputeMacroHost host{&system, *this, *front_regs, memory_manager};
    if (!TuringMacro().Execute(MacroPositions()[entry], macro_params, host)) {
        LOG_ERROR(HW_GPU, "Compute macro {:#x} at {:#x} failed", entry, MacroPositions()[entry]);
    }
    macro_params.clear();
    static const bool verify = std::getenv("EDEN_GPU_SPLIT_VERIFY") != nullptr;
    if (verify) {
        auto& split = *VideoCommon::GpuSplit::Active();
        split.PushTask([this, &system] { ConsumeSink(system); });
        split.Drain();
        static u64 failures{};
        u32 differences = 0;
        std::string listing;
        for (u32 index = 0; index < Regs::NUM_REGS; ++index) {
            if (IsFrontMethod(index) || front_regs->reg_array[index] == regs.reg_array[index]) {
                continue;
            }
            if (differences++ < 8) {
                listing += fmt::format(" {:#x}: front {:#x} back {:#x};", index,
                                       front_regs->reg_array[index], regs.reg_array[index]);
            }
        }
        if (differences != 0 && failures++ < 32) {
            LOG_ERROR(HW_GPU, "Split GPU thread: {} compute registers differ:{}", differences,
                      listing);
        }
    }
}

void KeplerCompute::CallMethod(Core::System& system, u32 method, u32 method_argument, bool is_last_call) {
    if (!front_macros && executing_macro != 0 && !IsMacroData(method)) {
        FlushPendingMacro(system);
    }
    if (method >= MACRO_REGISTERS_START && is_ampere) {
        ProcessMacro(system, method, &method_argument, 1, is_last_call);
        return;
    }
    if (method >= Regs::NUM_REGS) {
        LOG_ERROR(HW_GPU, "Invalid KeplerCompute register {:#x} = {:#x}", method * sizeof(u32), method_argument);
        return;
    }

    regs.reg_array[method] = method_argument;

    switch (method) {
    case LOAD_MME_INSTRUCTION_RAM_POINTER:
        if (is_ampere) {
            TuringMacro().SetUploadPointer(method_argument);
        }
        break;
    case LOAD_MME_INSTRUCTION_RAM:
        if (is_ampere) {
            TuringMacro().Upload(method_argument);
        }
        break;
    case LOAD_MME_START_ADDRESS_RAM_POINTER:
        macro_position_pointer = method_argument;
        break;
    case LOAD_MME_START_ADDRESS_RAM:
        if (is_ampere) {
            MacroPositions()[macro_position_pointer++ % MacroPositions().size()] = method_argument;
        }
        break;
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
    case REPORT_SEMAPHORE_EXECUTE:
        if (is_ampere) {
            ReportSemaphore::Decode(regs.reg_array[REPORT_SEMAPHORE],
                                    regs.reg_array[REPORT_SEMAPHORE + 1],
                                    regs.reg_array[REPORT_SEMAPHORE + 2],
                                    regs.reg_array[REPORT_SEMAPHORE + 3], method_argument)
                .Execute(*rasterizer, memory_manager);
        }
        break;
    case REPORT_SEMAPHORE_LEGACY_EXECUTE: {
        const u32 address_high = regs.reg_array[REPORT_SEMAPHORE_LEGACY];
        const u32 address_low = regs.reg_array[REPORT_SEMAPHORE_LEGACY + 1];
        const u32 payload = regs.reg_array[REPORT_SEMAPHORE_LEGACY + 2];
        const GPUVAddr sequence_address =
            (static_cast<GPUVAddr>(address_high & 0xFF) << 32) | address_low;
        const u32 operation = method_argument & 0x3;
        const bool reduction_enable = (method_argument & (1U << 3)) != 0;
        const bool short_query = (method_argument & ((1U << 12) | (1U << 28))) != 0;
        const u32 report = (method_argument >> 23) & 0x1F;
        const u32 subreport = (method_argument >> 5) & 0x7;
        if (operation == 0 || operation == 2) {
            if (report == 0 && reduction_enable) {
                const SemaphoreRelease release{
                    .address = sequence_address,
                    .payload = payload,
                    .payload_64 = false,
                    .timestamp = !short_query,
                    .reduction = static_cast<SemaphoreReduction>((method_argument >> 9) & 0x7),
                    .reduction_signed = ((method_argument >> 17) & 0x3) == 1,
                };
                ReleaseSemaphore(*rasterizer, memory_manager, release);
                break;
            }
            VideoCommon::QueryPropertiesFlags flags{};
            if (!short_query) {
                flags |= VideoCommon::QueryPropertiesFlags::HasTimeout;
            } else if (operation == 0) {
                flags |= VideoCommon::QueryPropertiesFlags::IsAFence;
            }
            rasterizer->Query(sequence_address, static_cast<VideoCommon::QueryType>(report), flags,
                              payload, subreport);
        } else {
            LOG_ERROR(HW_GPU, "Unimplemented KeplerCompute legacy report semaphore operation {}",
                      operation);
        }
        break;
    }
    case LOAD_INLINE_QMD_DATA_END - 1:
        if (is_ampere) {
            ConsumeSink(system);
            regs.reg_array[method] = method_argument;
            ExecuteInlineQmd();
        }
        break;
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
            SignalPcas(launch_desc_loc, method_argument & 0xF,
                       ((method_argument >> 14) & 0x3FF) + 1);
        } else if ((method_argument & 2) != 0) {
            // SEND_SIGNALING_PCAS_B with SCHEDULE
            SignalPcas(launch_desc_loc, static_cast<u32>(PcasAction::Schedule), 1);
        }
        indirect_compute = std::nullopt;
        break;
    }
    default:
        if (is_ampere && ((method >= AMPERE_MME_DMA_BEGIN && method < AMPERE_MME_DMA_END) ||
                          method == AMPERE_MME_DMA_WRITE_METHOD_BARRIER)) {
            ComputeMacroHost host{system, *this, memory_manager};
            TuringMacro().ProcessMethod(method, method_argument, host);
        }
        break;
    }
}

void KeplerCompute::CallMultiMethod(Core::System& system, u32 method, const u32* base_start, u32 amount, u32 methods_pending) {
    if (!front_macros && executing_macro != 0 && !IsMacroData(method)) {
        FlushPendingMacro(system);
    }
    if (method >= MACRO_REGISTERS_START && is_ampere) {
        ProcessMacro(system, method, base_start, amount, amount == methods_pending);
        return;
    }
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

void KeplerCompute::ProcessMacro(Core::System& system, u32 method, const u32* base_start, u32 amount,
                                 bool is_last_call) {
    // As in the 3D engine, the call method starts a macro, data methods feed its FIFO, and
    // the macro runs once a method other than data arrives.
    if ((method - MACRO_REGISTERS_START) % 2 == 0) {
        FlushPendingMacro(system);
        executing_macro = method;
    } else if (executing_macro == 0) {
        LOG_ERROR(HW_GPU, "Compute macro data {:#x} without a macro call", method * sizeof(u32));
        return;
    }
    macro_params.insert(macro_params.end(), base_start, base_start + amount);
}

bool KeplerCompute::IsMacroData(u32 method) const {
    return is_ampere && method >= MACRO_REGISTERS_START &&
           (method - MACRO_REGISTERS_START) % 2 == 1;
}

void KeplerCompute::FlushPendingMacro(Core::System& system) {
    if (front_macros) {
        FrontFlushMacro(system);
        return;
    }
    if (executing_macro == 0) {
        return;
    }
    const u32 entry = ((executing_macro - MACRO_REGISTERS_START) >> 1) %
                      static_cast<u32>(MacroPositions().size());
    executing_macro = 0;
    ConsumeSink(system);
    ComputeMacroHost host{system, *this, memory_manager};
    if (!TuringMacro().Execute(MacroPositions()[entry], macro_params, host)) {
        LOG_ERROR(HW_GPU, "Compute macro {:#x} at {:#x} failed", entry, MacroPositions()[entry]);
    }
    macro_params.clear();
}

void KeplerCompute::ProcessLaunch() {
    const GPUVAddr launch_desc_loc = regs.launch_desc_loc.Address();
    memory_manager.ReadBlockUnsafe(launch_desc_loc, &launch_description,
                                   LaunchParams::NUM_LAUNCH_PARAMETERS * sizeof(u32));
    rasterizer->DispatchCompute();
}

void KeplerCompute::SignalPcas(GPUVAddr address, u32 action, u32 count) {
    struct Request {
        GPUVAddr address;
        PcasAction action;
        u32 count;
    };
    auto& qmd_states = memory_manager.QmdStates();
    // Executing a QMD can signal its dependent QMD, which is handled in the same loop
    std::vector<Request> requests{{address, PcasAction{action}, count}};
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
        case PcasAction::InvalidatePrefetchCopyForceRequireScheduling:
            qmd_states[request.address].dependencies +=
                static_cast<s32>(read_qmd().DependenceCounter());
            qmd_states[request.address].scheduled = true;
            break;
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

void KeplerCompute::ExecuteInlineQmd() {
    QmdV3 qmd{};
    std::memcpy(qmd.words.data(), &regs.reg_array[LOAD_INLINE_QMD_DATA_BEGIN], sizeof(qmd.words));
    const GPUVAddr qmd_address =
        ((static_cast<GPUVAddr>(regs.reg_array[SET_INLINE_QMD_ADDRESS_A] & 0x1FFFFFF) << 32) |
         static_cast<GPUVAddr>(regs.reg_array[SET_INLINE_QMD_ADDRESS_B]))
        << 8;
    if (qmd_address != 0) {
        rasterizer->AccelerateInlineToMemory(
            qmd_address, sizeof(qmd.words),
            std::span<const u8>{reinterpret_cast<const u8*>(qmd.words.data()), sizeof(qmd.words)});
        SignalPcas(qmd_address, static_cast<u32>(PcasAction::InvalidateCopySchedule), 1);
        return;
    }
    ExecuteQmd(qmd);
    if (const auto dependent = qmd.GetDependent()) {
        switch (dependent->action) {
        case QmdV3::DependentAction::Schedule:
            SignalPcas(dependent->address, static_cast<u32>(PcasAction::Schedule), 1);
            break;
        case QmdV3::DependentAction::InvalidateCopySchedule:
            SignalPcas(dependent->address, static_cast<u32>(PcasAction::InvalidateCopySchedule), 1);
            break;
        case QmdV3::DependentAction::DecrementDependence:
            SignalPcas(dependent->address, static_cast<u32>(PcasAction::DecrementDependence), 1);
            break;
        default:
            break;
        }
    }
    if (qmd.HasSecondDependent()) {
        LOG_ERROR(HW_GPU, "Second dependent QMD is not implemented");
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
