// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2018 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <cstring>
#include <cstdlib>
#include <optional>
#include <string>
#include "common/assert.h"
#include "common/bit_util.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "video_core/dirty_flags.h"
#include "video_core/draw_capture.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/engines/semaphore.h"
#include "video_core/gpu.h"
#include "video_core/gpu_split.h"
#include "video_core/memory_manager.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/textures/texture.h"

namespace Tegra::Engines {

namespace {
// SET_REPORT_SEMAPHORE_PAYLOAD_LOWER of AMPERE_B, followed by PAYLOAD_UPPER, ADDRESS_LOWER,
// ADDRESS_UPPER and REPORT_SEMAPHORE_EXECUTE
constexpr u32 AMPERE_REPORT_SEMAPHORE = 0x0158 / sizeof(u32);
constexpr u32 AMPERE_REPORT_SEMAPHORE_EXECUTE = AMPERE_REPORT_SEMAPHORE + 4;

// MME DMA methods of AMPERE_B: SET_MME_MEM_ADDRESS_A to SET_MME_DATA_FIFO_CONFIG, and
// MME_DMA_WRITE_METHOD_BARRIER
constexpr u32 AMPERE_MME_DMA_BEGIN = 0x0550 / sizeof(u32);
constexpr u32 AMPERE_MME_DMA_END = 0x0578 / sizeof(u32);
constexpr u32 AMPERE_MME_DMA_WRITE_METHOD_BARRIER = 0x0DEC / sizeof(u32);

/// What a Turing macro runs on: the registers, methods and memory of the 3D engine
class TuringMacroHost final : public TuringMacroEngine::Host {
public:
    TuringMacroHost(Core::System* system_, Maxwell3D& maxwell3d_, MemoryManager& memory_manager_)
        : system{system_}, maxwell3d{maxwell3d_}, memory_manager{memory_manager_} {}

    u32 ReadState(u32 method) override {
        return method < Maxwell3D::Regs::NUM_REGS ? maxwell3d.regs.reg_array[method] : 0;
    }

    void CallMethod(u32 method, u32 value) override {
        if (system == nullptr || method >= 0xE00) {
            LOG_ERROR(HW_GPU, "Unsupported method {:#x} emitted by a macro", method);
            return;
        }
        maxwell3d.CallMethod(*system, method, value, true);
    }

    void ReadMemory(GPUVAddr address, std::span<u32> data) override {
        memory_manager.ReadBlock(address, data.data(), data.size_bytes());
    }

    void WriteMemory(GPUVAddr address, std::span<const u32> data) override {
        memory_manager.WriteBlock(address, data.data(), data.size_bytes());
    }

private:
    Core::System* system;
    Maxwell3D& maxwell3d;
    MemoryManager& memory_manager;
};

/// What a Turing macro runs on at the front of a split GPU thread: the front's copy of the
/// registers, with method writes sent to the back
class FrontMacroHost final : public TuringMacroEngine::Host {
public:
    FrontMacroHost(Core::System* system_, Maxwell3D& maxwell3d_, const Maxwell3D::Regs& regs_,
                   MemoryManager& memory_manager_)
        : system{system_}, maxwell3d{maxwell3d_}, regs{regs_}, memory_manager{memory_manager_} {}

    u32 ReadState(u32 method) override {
        return method < Maxwell3D::Regs::NUM_REGS ? regs.reg_array[method] : 0;
    }

    void CallMethod(u32 method, u32 value) override {
        if (system == nullptr || method >= 0xE00) {
            LOG_ERROR(HW_GPU, "Unsupported method {:#x} emitted by a macro", method);
            return;
        }
        if (maxwell3d.IsFrontMethod(method)) {
            maxwell3d.FrontCallMethod(*system, method, value);
            return;
        }
        maxwell3d.FrontWrite(method, value);
        VideoCommon::GpuSplit::Active()->PushEngineMethod(maxwell3d, method, value);
    }

    // Guest memory as of this point: what was sent before and may write it has to be executed
    void ReadMemory(GPUVAddr address, std::span<u32> data) override {
        if (!maxwell3d.FrontReadEarly(address, data)) {
            VideoCommon::GpuSplit::Active()->Drain();
            memory_manager.ReadBlock(address, data.data(), data.size_bytes());
        }
        if (auto& capture{VideoCommon::DrawCapture::Get()}; capture.Active()) {
            capture.Line(fmt::format("  macro read {:#x} {}", address,
                                     fmt::join(data.begin(), data.begin() + std::min<size_t>(data.size(), 8), ",")));
        }
    }

    void WriteMemory(GPUVAddr address, std::span<const u32> data) override {
        VideoCommon::GpuSplit::Active()->Drain();
        if (auto& capture{VideoCommon::DrawCapture::Get()}; capture.Active()) {
            capture.Line(fmt::format("  macro write {:#x} {}", address,
                                     fmt::join(data.begin(), data.begin() + std::min<size_t>(data.size(), 8), ",")));
        }
        memory_manager.WriteBlock(address, data.data(), data.size_bytes());
    }

private:
    Core::System* system;
    Maxwell3D& maxwell3d;
    const Maxwell3D::Regs& regs;
    MemoryManager& memory_manager;
};
} // Anonymous namespace

/// First register id that is actually a Macro call.
constexpr u32 MacroRegistersStart = 0xE00;

Maxwell3D::Maxwell3D(MemoryManager& memory_manager_)
    : draw_manager()
    , memory_manager{memory_manager_}
#ifdef ARCHITECTURE_x86_64
    , macro_engine(bool(Settings::values.disable_macro_jit))
#else
    , macro_engine(true)
#endif
    , upload_state{memory_manager, regs.upload}
{
    dirty.flags.flip();
    InitializeRegisterDefaults();
    execution_mask.reset();
    for (size_t i = 0; i < execution_mask.size(); i++)
        execution_mask[i] = IsMethodExecutable(u32(i));
}

Maxwell3D::~Maxwell3D() = default;

void Maxwell3D::BindRasterizer(VideoCore::RasterizerInterface* rasterizer_) {
    rasterizer = rasterizer_;
    upload_state.BindRasterizer(rasterizer_);
}

void Maxwell3D::InitializeRegisterDefaults() {
    // Initializes registers to their default values - what games expect them to be at boot. This is
    // for certain registers that may not be explicitly set by games.

    // Reset all registers to zero
    std::memset(&regs, 0, sizeof(regs));

    // Depth range near/far is not always set, but is expected to be the default 0.0f, 1.0f. This is
    // needed for ARMS.
    for (auto& viewport : regs.viewports) {
        viewport.depth_range_near = 0.0f;
        viewport.depth_range_far = 1.0f;
    }
    for (auto& viewport : regs.viewport_transform) {
        viewport.swizzle.x.Assign(Regs::ViewportSwizzle::PositiveX);
        viewport.swizzle.y.Assign(Regs::ViewportSwizzle::PositiveY);
        viewport.swizzle.z.Assign(Regs::ViewportSwizzle::PositiveZ);
        viewport.swizzle.w.Assign(Regs::ViewportSwizzle::PositiveW);
    }

    // Doom and Bomberman seems to use the uninitialized registers and just enable blend
    // so initialize blend registers with sane values
    regs.blend.color_op = Regs::Blend::Equation::Add_D3D;
    regs.blend.color_source = Regs::Blend::Factor::One_D3D;
    regs.blend.color_dest = Regs::Blend::Factor::Zero_D3D;
    regs.blend.alpha_op = Regs::Blend::Equation::Add_D3D;
    regs.blend.alpha_source = Regs::Blend::Factor::One_D3D;
    regs.blend.alpha_dest = Regs::Blend::Factor::Zero_D3D;
    for (auto& blend : regs.blend_per_target) {
        blend.color_op = Regs::Blend::Equation::Add_D3D;
        blend.color_source = Regs::Blend::Factor::One_D3D;
        blend.color_dest = Regs::Blend::Factor::Zero_D3D;
        blend.alpha_op = Regs::Blend::Equation::Add_D3D;
        blend.alpha_source = Regs::Blend::Factor::One_D3D;
        blend.alpha_dest = Regs::Blend::Factor::Zero_D3D;
    }
    regs.stencil_front_op.fail = Regs::StencilOp::Op::Keep_D3D;
    regs.stencil_front_op.zfail = Regs::StencilOp::Op::Keep_D3D;
    regs.stencil_front_op.zpass = Regs::StencilOp::Op::Keep_D3D;
    regs.stencil_front_op.func = Regs::ComparisonOp::Always_GL;
    regs.stencil_front_func_mask = 0xFFFFFFFF;
    regs.stencil_front_mask = 0xFFFFFFFF;
    regs.stencil_two_side_enable = 1;
    regs.stencil_back_op.fail = Regs::StencilOp::Op::Keep_D3D;
    regs.stencil_back_op.zfail = Regs::StencilOp::Op::Keep_D3D;
    regs.stencil_back_op.zpass = Regs::StencilOp::Op::Keep_D3D;
    regs.stencil_back_op.func = Regs::ComparisonOp::Always_GL;
    regs.stencil_back_func_mask = 0xFFFFFFFF;
    regs.stencil_back_mask = 0xFFFFFFFF;

    regs.depth_test_func = Regs::ComparisonOp::Always_GL;
    regs.gl_front_face = Regs::FrontFace::CounterClockWise;
    regs.gl_cull_face = Regs::CullFace::Back;

    // TODO(Rodrigo): Most games do not set a point size. I think this is a case of a
    // register carrying a default value. Assume it's OpenGL's default (1).
    regs.point_size = 1.0f;

    // TODO(bunnei): Some games do not initialize the color masks (e.g. Sonic Mania). Assuming a
    // default of enabled fixes rendering here.
    for (auto& color_mask : regs.color_mask) {
        color_mask.R.Assign(1);
        color_mask.G.Assign(1);
        color_mask.B.Assign(1);
        color_mask.A.Assign(1);
    }

    for (auto& format : regs.vertex_attrib_format) {
        format.constant.Assign(1);
    }

    // NVN games expect these values to be enabled at boot
    regs.rasterize_enable = 1;
    regs.color_target_mrt_enable = 1;
    regs.framebuffer_srgb = 1;
    regs.line_width_aliased = 1.0f;
    regs.line_width_smooth = 1.0f;
    regs.gl_front_face = Maxwell3D::Regs::FrontFace::ClockWise;
    regs.polygon_mode_back = Maxwell3D::Regs::PolygonMode::Fill;
    regs.polygon_mode_front = Maxwell3D::Regs::PolygonMode::Fill;

    shadow_state = regs;
}

bool Maxwell3D::IsMethodExecutable(u32 method) {
    if (method >= MacroRegistersStart) {
        return true;
    }
    if (method >= AMPERE_MME_DMA_BEGIN && method < AMPERE_MME_DMA_END) {
        return true;
    }
    switch (method) {
    case MAXWELL3D_REG_INDEX(draw.end):
    case MAXWELL3D_REG_INDEX(draw.begin):
    case MAXWELL3D_REG_INDEX(vertex_buffer.first):
    case MAXWELL3D_REG_INDEX(vertex_buffer.count):
    case MAXWELL3D_REG_INDEX(index_buffer.first):
    case MAXWELL3D_REG_INDEX(index_buffer.count):
    case MAXWELL3D_REG_INDEX(draw_inline_index):
    case MAXWELL3D_REG_INDEX(index_buffer32_subsequent):
    case MAXWELL3D_REG_INDEX(index_buffer16_subsequent):
    case MAXWELL3D_REG_INDEX(index_buffer8_subsequent):
    case MAXWELL3D_REG_INDEX(index_buffer32_first):
    case MAXWELL3D_REG_INDEX(index_buffer16_first):
    case MAXWELL3D_REG_INDEX(index_buffer8_first):
    case MAXWELL3D_REG_INDEX(inline_index_2x16.even):
    case MAXWELL3D_REG_INDEX(inline_index_4x8.index0):
    case MAXWELL3D_REG_INDEX(vertex_array_instance_first):
    case MAXWELL3D_REG_INDEX(vertex_array_instance_subsequent):
    case MAXWELL3D_REG_INDEX(draw_texture.src_y0):
    case MAXWELL3D_REG_INDEX(wait_for_idle):
    case MAXWELL3D_REG_INDEX(shadow_ram_control):
    case MAXWELL3D_REG_INDEX(load_mme.instruction_ptr):
    case MAXWELL3D_REG_INDEX(load_mme.instruction):
    case MAXWELL3D_REG_INDEX(load_mme.start_address):
    case MAXWELL3D_REG_INDEX(falcon[4]):
    case MAXWELL3D_REG_INDEX(const_buffer.buffer):
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 1:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 2:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 3:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 4:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 5:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 6:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 7:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 8:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 9:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 10:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 11:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 12:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 13:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 14:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 15:
    case MAXWELL3D_REG_INDEX(bind_groups[0].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[1].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[2].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[3].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[4].raw_config):
    case MAXWELL3D_REG_INDEX(topology_override):
    case MAXWELL3D_REG_INDEX(clear_surface):
    case MAXWELL3D_REG_INDEX(report_semaphore.query):
    case AMPERE_REPORT_SEMAPHORE_EXECUTE:
    case AMPERE_MME_DMA_WRITE_METHOD_BARRIER:
    case MAXWELL3D_REG_INDEX(render_enable.mode):
    case MAXWELL3D_REG_INDEX(clear_report_value):
    case MAXWELL3D_REG_INDEX(sync_info):
    case MAXWELL3D_REG_INDEX(launch_dma):
    case MAXWELL3D_REG_INDEX(inline_data):
    case MAXWELL3D_REG_INDEX(fragment_barrier):
    case MAXWELL3D_REG_INDEX(invalidate_texture_data_cache):
    case MAXWELL3D_REG_INDEX(tiled_cache_barrier):
        return true;
    default:
        return false;
    }
}

void Maxwell3D::ProcessMacro(Core::System& system, u32 method, const u32* base_start, u32 amount, bool is_last_call) {
    if (is_ampere) {
        // The call method starts a macro and every data method feeds its FIFO, whichever
        // macro index it names. The macro runs once a method other than data arrives.
        if ((method - MacroRegistersStart) % 2 == 0) {
            FlushPendingMacro(system);
            executing_macro = method;
        } else if (executing_macro == 0) {
            LOG_ERROR(HW_GPU, "Macro data {:#x} without a macro call", method * sizeof(u32));
            return;
        }
        macro_params.insert(macro_params.end(), base_start, base_start + amount);
        macro_segments.emplace_back(current_dma_segment, amount);
        current_macro_dirty |= current_dirty;
        current_dirty = false;
        return;
    }
    if (executing_macro == 0) {
        // A macro call must begin by writing the macro method's register, not its argument.
        ASSERT((method % 2) == 0 && "Can't start macro execution by writing to the ARGS register");
        executing_macro = method;
    }

    macro_params.insert(macro_params.end(), base_start, base_start + amount);
    macro_segments.emplace_back(current_dma_segment, amount);
    current_macro_dirty |= current_dirty;
    current_dirty = false;

    // Call the macro when there are no more parameters in the command buffer
    if (is_last_call) {
        ConsumeSink(system);
        CallMacroMethod(system, executing_macro, macro_params);
        macro_params.clear();
        macro_segments.clear();
        current_macro_dirty = false;
    }
}

bool Maxwell3D::IsMacroData(u32 method) const {
    return method >= MacroRegistersStart && (method - MacroRegistersStart) % 2 == 1;
}

void Maxwell3D::FlushPendingMacro(Core::System& system) {
    if (front_macros) {
        FrontFlushMacro(system);
        return;
    }
    if (executing_macro == 0) {
        return;
    }
    ConsumeSink(system);
    CallMacroMethod(system, executing_macro, macro_params);
    macro_params.clear();
    macro_segments.clear();
    current_macro_dirty = false;
}

void Maxwell3D::RefreshParametersImpl() {
    if (!Settings::IsGPULevelHigh()) {
        return;
    }
    size_t current_index = 0;
    for (auto& segment : macro_segments) {
        if (segment.first == 0) {
            current_index += segment.second;
            continue;
        }
        memory_manager.ReadBlock(segment.first, &macro_params[current_index],
                                 sizeof(u32) * segment.second);
        current_index += segment.second;
    }
}

u32 Maxwell3D::GetMaxCurrentVertices() {
    u32 num_vertices = 0;
    for (size_t index = 0; index < Regs::NumVertexArrays; ++index) {
        const auto& array = regs.vertex_streams[index];
        if (array.enable == 0) {
            continue;
        }
        const auto& attribute = regs.vertex_attrib_format[index];
        if (attribute.constant) {
            num_vertices = (std::max)(num_vertices, 1U);
            continue;
        }
        const auto& limit = regs.vertex_stream_limits[index];
        const GPUVAddr gpu_addr_begin = array.Address();
        const GPUVAddr gpu_addr_end = limit.Address() + 1;
        const u32 address_size = static_cast<u32>(gpu_addr_end - gpu_addr_begin);
        num_vertices = (std::max)(
            num_vertices, address_size / (std::max)(attribute.SizeInBytes(), array.stride.Value()));
        break;
    }
    return num_vertices;
}

size_t Maxwell3D::EstimateIndexBufferSize() {
    GPUVAddr start_address = regs.index_buffer.StartAddress();
    GPUVAddr end_address = regs.index_buffer.EndAddress();
    auto const byte_size = regs.index_buffer.FormatSizeInBytes();
    auto const max_size = 1ull << (byte_size * CHAR_BIT);
    auto const upper_cap = GetMaxCurrentVertices() * 4 * byte_size;
    auto const lower_cap = std::min<size_t>(size_t(end_address - start_address), upper_cap);
    return std::min<size_t>(memory_manager.GetMemoryLayoutSize(start_address, byte_size * max_size) / byte_size, lower_cap);
}

u32 Maxwell3D::ProcessShadowRam(u32 method, u32 argument) {
    // Keep track of the register value in shadow_state when requested.
    auto const c = shadow_state.shadow_ram_control;
    if (c == Regs::ShadowRamControl::Track || c == Regs::ShadowRamControl::TrackWithFilter)
        return shadow_state.reg_array[method] = argument;
    else if (c == Regs::ShadowRamControl::Replay)
        return shadow_state.reg_array[method];
    return argument;
}

void Maxwell3D::ConsumeSinkImpl(Core::System& system) {
    const auto control = shadow_state.shadow_ram_control;
    // Sunk methods are state writes only (not in execution_mask): skip unchanged values, as
    // CallMethod does
    const bool skip_same = SkipSameValueWrites();
    const auto process = [&](u32 method, u32 value) {
        if (!skip_same || regs.reg_array[method] != value) {
            ProcessDirtyRegisters(method, value);
        }
    };
    if (control == Regs::ShadowRamControl::Track || control == Regs::ShadowRamControl::TrackWithFilter) {
        for (auto [method, value] : method_sink) {
            shadow_state.reg_array[method] = value;
            process(method, value);
        }
    } else if (control == Regs::ShadowRamControl::Replay) {
        for (auto [method, value] : method_sink)
            process(method, shadow_state.reg_array[method]);
    } else {
        for (auto [method, value] : method_sink)
            process(method, value);
    }
    method_sink.clear();
}

void Maxwell3D::ProcessDirtyRegisters(u32 method, u32 argument) {
    // The instances after the first are drawn together once their loop ends, and they must see
    // the state they were begun with: NVN's draw macros reset the base instance (and its copy in
    // c[0][0x644]) right after the loop, before the macro returns
    if (draw_manager.HasDeferredDraw() && !DrawManager::IsInstanceLoopMethod(method)) [[unlikely]] {
        draw_manager.DrawDeferred(*this, true);
    }
    regs.reg_array[method] = argument;
    for (auto const& table : dirty.tables)
        dirty.flags[table[method]] = true;
    if (is_ampere) {
        SyncAmpereBufferLimits(method);
    }
}

void Maxwell3D::SyncAmpereBufferLimits(u32 method) {
    if (const auto index = SyncAmpereBufferLimits(regs, method)) {
        for (auto const& table : dirty.tables) {
            dirty.flags[table[*index]] = true;
        }
    }
}

std::optional<u32> Maxwell3D::SyncAmpereBufferLimits(Regs& regs, u32 method) {
    // AMPERE_B sets index and vertex buffer sizes (bits 39:32 in A, 31:0 in B) instead of
    // limit addresses. The limit registers are unused there, so they are kept at start + size.
    static constexpr u32 INDEX_BUFFER_SIZE_A = 0x238 / sizeof(u32);
    static constexpr u32 VERTEX_STREAM_SIZE_A = 0x600 / sizeof(u32);
    static constexpr u32 VERTEX_STREAM_SIZE_WORDS = 2;
    static constexpr u32 VERTEX_STREAM_WORDS = sizeof(Regs::VertexStream) / sizeof(u32);
    static constexpr u32 VERTEX_STREAMS = MAXWELL3D_REG_INDEX(vertex_streams);
    static constexpr u32 VERTEX_STREAM_LIMITS = MAXWELL3D_REG_INDEX(vertex_stream_limits);
    const auto size_at{[&regs](u32 index) {
        return (u64{regs.reg_array[index] & 0xFF} << 32) | regs.reg_array[index + 1];
    }};
    if (method == INDEX_BUFFER_SIZE_A || method == INDEX_BUFFER_SIZE_A + 1 ||
        method == MAXWELL3D_REG_INDEX(index_buffer.start_addr_high) ||
        method == MAXWELL3D_REG_INDEX(index_buffer.start_addr_low)) {
        const GPUVAddr end{regs.index_buffer.StartAddress() + size_at(INDEX_BUFFER_SIZE_A)};
        regs.index_buffer.limit_addr_high = static_cast<u32>(end >> 32);
        regs.index_buffer.limit_addr_low = static_cast<u32>(end);
        return static_cast<u32>(MAXWELL3D_REG_INDEX(index_buffer.limit_addr_high));
    }
    std::optional<u32> stream;
    if (method >= VERTEX_STREAM_SIZE_A &&
        method < VERTEX_STREAM_SIZE_A + Regs::NumVertexArrays * VERTEX_STREAM_SIZE_WORDS) {
        stream = (method - VERTEX_STREAM_SIZE_A) / VERTEX_STREAM_SIZE_WORDS;
    } else if (method >= VERTEX_STREAMS &&
               method < VERTEX_STREAMS + Regs::NumVertexArrays * VERTEX_STREAM_WORDS) {
        stream = (method - VERTEX_STREAMS) / VERTEX_STREAM_WORDS;
    }
    if (!stream) {
        return std::nullopt;
    }
    // Limits are inclusive; a zero size leaves an empty range
    const u64 size{size_at(VERTEX_STREAM_SIZE_A + *stream * VERTEX_STREAM_SIZE_WORDS)};
    const GPUVAddr limit{regs.vertex_streams[*stream].Address() + size - 1};
    auto& stream_limit{regs.vertex_stream_limits[*stream]};
    stream_limit.address_high = static_cast<u32>(limit >> 32);
    stream_limit.address_low = static_cast<u32>(limit);
    return static_cast<u32>(VERTEX_STREAM_LIMITS + *stream * VERTEX_STREAM_SIZE_WORDS);
}

void Maxwell3D::EnableFrontMacros() {
    front_regs = std::make_unique<Regs>(regs);
    front_shadow = std::make_unique<Regs>(shadow_state);
    // Writes the back has not applied yet
    for (const auto& [method, value] : method_sink) {
        front_regs->reg_array[method] = FrontShadowRam(method, value);
    }
    front_macros = true;
}

void Maxwell3D::FrontClassify(u32 method, u32 amount) {
    // What the method may write once the back runs it, from the registers before it. The draws
    // and state writes (which can issue a deferred draw) are GPU work; constant buffer data and
    // reports write ranges the front knows; uploads and syncpoints may write anything.
    using Effect = VideoCommon::GpuSplit::Effect;
    auto& split = *VideoCommon::GpuSplit::Active();
    constexpr u32 cb_data = MAXWELL3D_REG_INDEX(const_buffer.buffer);
    split.Classify(Effect::Gpu);
    if (method >= cb_data && method < cb_data + 16) {
        FrontClassifyWrite(front_regs->const_buffer.Address() + front_regs->const_buffer.offset,
                           u64{amount} * sizeof(u32));
        return;
    }
    switch (method) {
    case MAXWELL3D_REG_INDEX(report_semaphore.query):
        // Payload, then the timestamp of a long report
        FrontClassifyWrite(front_regs->report_semaphore.Address(), 16);
        return;
    case AMPERE_REPORT_SEMAPHORE_EXECUTE:
        FrontClassifyWrite(
            (GPUVAddr{front_regs->reg_array[AMPERE_REPORT_SEMAPHORE + 3] & 0xFF} << 32) |
                front_regs->reg_array[AMPERE_REPORT_SEMAPHORE + 2],
            16);
        return;
    case MAXWELL3D_REG_INDEX(wait_for_idle):
    case MAXWELL3D_REG_INDEX(invalidate_texture_data_cache):
        split.ClassifySync();
        return;
    case MAXWELL3D_REG_INDEX(inline_data): {
        const auto [address, size] = Upload::DestinationRange(
            front_regs->upload,
            front_regs->launch_dma.memory_layout.Value() == Regs::LaunchDMA::Layout::Pitch);
        FrontClassifyWrite(address, size);
        return;
    }
    case MAXWELL3D_REG_INDEX(sync_info):
        split.Classify(Effect::Unknown);
        return;
    default:
        return;
    }
}

void Maxwell3D::FrontClassifyWrite(GPUVAddr address, u64 size) {
    VideoCommon::GpuSplit::Active()->ClassifyGpuWrite(memory_manager, address, size);
}

bool Maxwell3D::FrontReadEarly(GPUVAddr address, std::span<u32> data) {
    // Draining made every MME DMA read wait for all queued work, and the front stopped running
    // ahead (Ocarina of Time outdoors: 20k reads a second, the front waiting most of its time).
    // The read only needs the queued work it depends on (GpuSplit::ReadTarget); the back then
    // waits between two entries while the front reads and flushes the caches as after a drain.
    // Pages GPU work ever wrote keep the drain. From Lichii e691cc3f2 (FrontDmaReadEarly in
    // src/video_core/engines/maxwell_3d.cpp), adapted.
    auto& split = *VideoCommon::GpuSplit::Active();
    const u64 size = data.size_bytes();
    if (size == 0) {
        return false;
    }
    constexpr GPUVAddr page_mask = VideoCommon::GpuSplit::PageSize - 1;
    u64 target = 0;
    std::array<std::pair<DAddr, u64>, 4> ranges;
    size_t num_ranges = 0;
    for (GPUVAddr page = address & ~page_mask; page < address + size; page += page_mask + 1) {
        const auto device = memory_manager.GpuToCpuAddress(page);
        if (!device || num_ranges == ranges.size()) {
            return false;
        }
        const DAddr device_page = *device & ~DAddr{page_mask};
        if (split.WasGpuWritten(device_page, page_mask + 1)) {
            return false;
        }
        // The words read: noted writes elsewhere in the page (reports next to the macro's data)
        // don't hold it up
        const GPUVAddr begin = (std::max)(page, address);
        const GPUVAddr last = (std::min)(page + page_mask + 1, address + size);
        target = (std::max)(target,
                            split.ReadTarget(*device + (begin - page), *device + (last - page)));
        ranges[num_ranges++] = {*device + (begin - page), last - begin};
    }
    // What the back runs after the target doesn't write these words (ReadTarget) but may still
    // flush them: a query reported before keeps its final value in the query cache until a
    // flush. Without one, guest memory holds the words and the back goes on meanwhile.
    split.WaitUntil(target);
    const bool has_queries = std::ranges::any_of(
        ranges.begin(), ranges.begin() + num_ranges,
        [this](const std::pair<DAddr, u64>& range) {
            return rasterizer->HasCachedQueries(range.first, range.second);
        });
    if (has_queries) {
        split.Hold(target);
        memory_manager.ReadBlock(address, data.data(), size);
        split.Release();
    } else {
        memory_manager.ReadBlockUnsafe(address, data.data(), size);
    }
    // EDEN_GPU_SPLIT_VERIFY_READS: drain, read again and log the words that differ
    static const bool verify = std::getenv("EDEN_GPU_SPLIT_VERIFY_READS") != nullptr;
    if (verify) [[unlikely]] {
        static u64 reads, differing;
        std::vector<u32> drained(data.size());
        split.Drain();
        memory_manager.ReadBlock(address, drained.data(), size);
        ++reads;
        if (!std::ranges::equal(drained, data)) {
            if (++differing <= 32) {
                LOG_ERROR(HW_GPU, "split early read at {:#x}: {} words, first {:#x} vs drained {:#x}",
                          address, data.size(), data[0], drained[0]);
            }
            std::ranges::copy(drained, data.begin());
        }
        if ((reads & 0xFFFF) == 0) {
            LOG_WARNING(HW_GPU, "split early reads {} differing {}", reads, differing);
        }
    }
    return true;
}

bool Maxwell3D::IsFrontMethod(u32 method) const {
    if (method >= MacroRegistersStart) {
        return true;
    }
    if ((method >= AMPERE_MME_DMA_BEGIN && method < AMPERE_MME_DMA_END) ||
        method == AMPERE_MME_DMA_WRITE_METHOD_BARRIER) {
        return true;
    }
    switch (method) {
    case MAXWELL3D_REG_INDEX(load_mme.instruction_ptr):
    case MAXWELL3D_REG_INDEX(load_mme.instruction):
    case MAXWELL3D_REG_INDEX(load_mme.start_address):
        return true;
    default:
        return false;
    }
}

u32 Maxwell3D::FrontShadowRam(u32 method, u32 argument) {
    auto const control = front_shadow->shadow_ram_control;
    if (control == Regs::ShadowRamControl::Track ||
        control == Regs::ShadowRamControl::TrackWithFilter) {
        return front_shadow->reg_array[method] = argument;
    }
    if (control == Regs::ShadowRamControl::Replay) {
        return front_shadow->reg_array[method];
    }
    return argument;
}

void Maxwell3D::FrontWrite(u32 method, u32 argument) {
    FrontClassify(method, 1);
    // CallMethod and ProcessMethodCall as far as they change registers
    const u32 value = FrontShadowRam(method, argument);
    front_regs->reg_array[method] = value;
    SyncAmpereBufferLimits(*front_regs, method);
    switch (method) {
    case MAXWELL3D_REG_INDEX(shadow_ram_control):
        front_shadow->shadow_ram_control = static_cast<Regs::ShadowRamControl>(argument);
        break;
    case MAXWELL3D_REG_INDEX(falcon[4]):
        front_regs->shadow_scratch[0] = 1;
        break;
    default:
        if (method >= MAXWELL3D_REG_INDEX(const_buffer.buffer) &&
            method < MAXWELL3D_REG_INDEX(const_buffer.buffer) + 16) {
            front_regs->const_buffer.offset += sizeof(u32);
        }
        break;
    }
}

void Maxwell3D::FrontMultiWrite(u32 method, const u32* base_start, u32 amount) {
    // CallMultiMethod as far as it changes registers
    if (method >= MAXWELL3D_REG_INDEX(const_buffer.buffer) &&
        method < MAXWELL3D_REG_INDEX(const_buffer.buffer) + 16) {
        FrontClassify(method, amount);
        front_regs->const_buffer.offset += amount * static_cast<u32>(sizeof(u32));
        return;
    }
    switch (method) {
    case MAXWELL3D_REG_INDEX(inline_data):
        FrontClassify(method, amount);
        return;
    case MAXWELL3D_REG_INDEX(draw_inline_index):
    case MAXWELL3D_REG_INDEX(inline_index_2x16.even):
    case MAXWELL3D_REG_INDEX(inline_index_4x8.index0):
        if (front_shadow->shadow_ram_control != Regs::ShadowRamControl::Replay) {
            FrontClassify(method, amount);
            if (amount != 0) {
                front_regs->reg_array[method] = FrontShadowRam(method, base_start[amount - 1]);
                SyncAmpereBufferLimits(*front_regs, method);
            }
            return;
        }
        break;
    default:
        break;
    }
    for (u32 i = 0; i < amount; ++i) {
        FrontWrite(method, base_start[i]);
    }
}

void Maxwell3D::FrontCallMethod(Core::System& system, u32 method, u32 argument) {
    if (method >= MacroRegistersStart) {
        FrontCallMultiMethod(system, method, &argument, 1);
        return;
    }
    const u32 value = FrontShadowRam(method, argument);
    front_regs->reg_array[method] = value;
    FrontMacroHost host{nullptr, *this, *front_regs, memory_manager};
    switch (method) {
    case MAXWELL3D_REG_INDEX(load_mme.instruction_ptr):
        return TuringMacro().SetUploadPointer(front_regs->load_mme.instruction_ptr);
    case MAXWELL3D_REG_INDEX(load_mme.instruction):
        return TuringMacro().Upload(value);
    case MAXWELL3D_REG_INDEX(load_mme.start_address):
        MacroPositions()[front_regs->load_mme.start_address_ptr++] = value;
        return;
    default:
        TuringMacro().ProcessMethod(method, value, host);
        return;
    }
}

void Maxwell3D::FrontCallMultiMethod(Core::System& system, u32 method, const u32* base_start,
                                     u32 amount) {
    if (method < MacroRegistersStart) {
        for (u32 i = 0; i < amount; ++i) {
            FrontCallMethod(system, method, base_start[i]);
        }
        return;
    }
    // ProcessMacro of AMPERE_B
    if ((method - MacroRegistersStart) % 2 == 0) {
        FrontFlushMacro(system);
        executing_macro = method;
    } else if (executing_macro == 0) {
        LOG_ERROR(HW_GPU, "Macro data {:#x} without a macro call", method * sizeof(u32));
        return;
    }
    macro_params.insert(macro_params.end(), base_start, base_start + amount);
    macro_segments.emplace_back(current_dma_segment, amount);
    current_macro_dirty |= current_dirty;
    current_dirty = false;
}

void Maxwell3D::FrontFlushMacro(Core::System& system) {
    if (executing_macro == 0) {
        return;
    }
    const u32 entry = ((executing_macro - MacroRegistersStart) >> 1) %
                      static_cast<u32>(MacroPositions().size());
    executing_macro = 0;
    // No RefreshParameters: the front read GPU-written command lists after the back had
    // written them back (SplitReadsGpuWritten), so the words it has are the ones to use. A
    // suspended channel can run its macro after the guest has reused that memory, and reading
    // it again gave macros command headers as parameters (Welcome Tour's loop count of 2^31,
    // then a host GPU timeout)
    if (auto& capture{VideoCommon::DrawCapture::Get()}; capture.Active()) {
        std::string segments;
        for (size_t i = 0; i < std::min<size_t>(macro_segments.size(), 3); ++i) {
            segments += fmt::format(" seg {:#x}+{}", macro_segments[i].first,
                                    macro_segments[i].second);
        }
        capture.Line(fmt::format("front macro {} at {:#x} params={}{}", entry,
                                 MacroPositions()[entry],
                                 fmt::join(macro_params.begin(),
                                           macro_params.begin() +
                                               std::min<size_t>(macro_params.size(), 10),
                                           ","),
                                 segments));
    }
    FrontMacroHost host{&system, *this, *front_regs, memory_manager};
    TuringMacro().Execute(MacroPositions()[entry], macro_params, host);
    auto& split = *VideoCommon::GpuSplit::Active();
    split.Classify(VideoCommon::GpuSplit::Effect::Gpu);
    split.PushCall([](void* engine) {
        auto& maxwell3d = *static_cast<Maxwell3D*>(engine);
        maxwell3d.draw_manager.DrawDeferred(maxwell3d);
    }, this);
    macro_params.clear();
    macro_segments.clear();
    current_macro_dirty = false;
    static const bool verify = std::getenv("EDEN_GPU_SPLIT_VERIFY") != nullptr;
    if (verify && (++front_macro_calls & 0xFFF) == 0) {
        split.PushTask([this, &system] { ConsumeSink(system); });
        split.Drain();
        VerifyFrontRegisters();
    }
}

void Maxwell3D::VerifyFrontRegisters() {
    static u64 checks{};
    static u64 failures{};
    ++checks;
    u32 differences = 0;
    std::string listing;
    for (u32 index = 0; index < Regs::NUM_REGS; ++index) {
        // Registers only the front handles
        if ((index >= MAXWELL3D_REG_INDEX(load_mme) &&
             index < MAXWELL3D_REG_INDEX(load_mme) + sizeof(Regs::LoadMME) / sizeof(u32)) ||
            (index >= AMPERE_MME_DMA_BEGIN && index < AMPERE_MME_DMA_END) ||
            index == AMPERE_MME_DMA_WRITE_METHOD_BARRIER) {
            continue;
        }
        if (front_regs->reg_array[index] != regs.reg_array[index]) {
            if (differences++ < 8) {
                listing += fmt::format(" {:#x}: front {:#x} back {:#x};", index,
                                       front_regs->reg_array[index], regs.reg_array[index]);
            }
        }
    }
    if (differences != 0 && failures++ < 32) {
        LOG_ERROR(HW_GPU, "Split GPU thread: {} registers differ after {} checks:{}", differences,
                  checks, listing);
    }
    if ((checks & 0x3F) == 0) {
        LOG_WARNING(HW_GPU, "Split GPU thread: {} register checks, {} with differences", checks,
                    failures);
    }
}

void Maxwell3D::ProcessMethodCall(u32 method, u32 argument, u32 nonshadow_argument, bool is_last_call) {
    switch (method) {
    case MAXWELL3D_REG_INDEX(wait_for_idle):
        return rasterizer->WaitForIdle();
    case MAXWELL3D_REG_INDEX(shadow_ram_control):
        shadow_state.shadow_ram_control = static_cast<Regs::ShadowRamControl>(nonshadow_argument);
        return;
    case MAXWELL3D_REG_INDEX(load_mme.instruction_ptr):
        if (is_ampere) {
            return TuringMacro().SetUploadPointer(regs.load_mme.instruction_ptr);
        }
        return macro_engine.ClearCode(regs.load_mme.instruction_ptr);
    case MAXWELL3D_REG_INDEX(load_mme.instruction):
        if (is_ampere) {
            return TuringMacro().Upload(argument);
        }
        return macro_engine.AddCode(regs.load_mme.instruction_ptr, argument);
    case MAXWELL3D_REG_INDEX(load_mme.start_address):
        return ProcessMacroBind(argument);
    case MAXWELL3D_REG_INDEX(falcon[4]):
        return ProcessFirmwareCall4();
    case AMPERE_REPORT_SEMAPHORE_EXECUTE:
        if (is_ampere) {
            ReportSemaphore::Decode(regs.reg_array[AMPERE_REPORT_SEMAPHORE],
                                    regs.reg_array[AMPERE_REPORT_SEMAPHORE + 1],
                                    regs.reg_array[AMPERE_REPORT_SEMAPHORE + 2],
                                    regs.reg_array[AMPERE_REPORT_SEMAPHORE + 3], argument)
                .Execute(*rasterizer, memory_manager);
        }
        return;
    case MAXWELL3D_REG_INDEX(const_buffer.buffer):
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 1:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 2:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 3:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 4:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 5:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 6:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 7:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 8:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 9:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 10:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 11:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 12:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 13:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 14:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 15:
        return ProcessCBData(argument);
    case MAXWELL3D_REG_INDEX(bind_groups[0].raw_config):
        return ProcessCBBind(0);
    case MAXWELL3D_REG_INDEX(bind_groups[1].raw_config):
        return ProcessCBBind(1);
    case MAXWELL3D_REG_INDEX(bind_groups[2].raw_config):
        return ProcessCBBind(2);
    case MAXWELL3D_REG_INDEX(bind_groups[3].raw_config):
        return ProcessCBBind(3);
    case MAXWELL3D_REG_INDEX(bind_groups[4].raw_config):
        return ProcessCBBind(4);
    case MAXWELL3D_REG_INDEX(report_semaphore.query):
        return ProcessQueryGet();
    case MAXWELL3D_REG_INDEX(render_enable.mode):
        return ProcessQueryCondition();
    case MAXWELL3D_REG_INDEX(clear_report_value):
        return ProcessCounterReset();
    case MAXWELL3D_REG_INDEX(sync_info):
        return ProcessSyncPoint();
    case MAXWELL3D_REG_INDEX(launch_dma):
        return upload_state.ProcessExec(regs.launch_dma.memory_layout.Value() == Regs::LaunchDMA::Layout::Pitch);
    case MAXWELL3D_REG_INDEX(inline_data):
        upload_state.ProcessData(argument, is_last_call);
        return;
    case MAXWELL3D_REG_INDEX(fragment_barrier):
        return rasterizer->FragmentBarrier();
    case MAXWELL3D_REG_INDEX(invalidate_texture_data_cache):
        rasterizer->InvalidateGPUCache();
        return rasterizer->WaitForIdle();
    case MAXWELL3D_REG_INDEX(tiled_cache_barrier):
        return rasterizer->TiledCacheBarrier();
    default:
        if (is_ampere && ((method >= AMPERE_MME_DMA_BEGIN && method < AMPERE_MME_DMA_END) ||
                          method == AMPERE_MME_DMA_WRITE_METHOD_BARRIER)) {
            TuringMacroHost host{nullptr, *this, memory_manager};
            TuringMacro().ProcessMethod(method, argument, host);
            return;
        }
        draw_manager.ProcessMethodCall(*this, method, argument);
        break;
    }
}

void Maxwell3D::CallMacroMethod(Core::System& system, u32 method, const std::vector<u32>& parameters) {
    // Reset the current macro.
    executing_macro = 0;

    // Lookup the macro offset
    const u32 entry =
        ((method - MacroRegistersStart) >> 1) % static_cast<u32>(MacroPositions().size());

    // Execute the current macro.
    if (is_ampere) {
        RefreshParameters();
        if (auto& capture{VideoCommon::DrawCapture::Get()}; capture.Active()) {
            capture.Line(fmt::format("macro {} at {:#x} dirty={} params={}", entry,
                                     MacroPositions()[entry], current_macro_dirty,
                                     fmt::join(parameters.begin(),
                                               parameters.begin() +
                                                   std::min<size_t>(parameters.size(), 10),
                                               ",")));
        }
        TuringMacroHost host{&system, *this, memory_manager};
        TuringMacro().Execute(MacroPositions()[entry], parameters, host);
    } else {
        macro_engine.Execute(system, *this, macro_positions[entry], parameters);
    }
    draw_manager.DrawDeferred(*this);
}

void Maxwell3D::CallMethod(Core::System& system, u32 method, u32 method_argument, bool is_last_call) {
    // It is an error to write to a register other than the current macro's ARG register before
    // it has finished execution.
    if (front_macros) {
        // The front handles macro state
    } else if (is_ampere && executing_macro != 0 && !IsMacroData(method)) {
        FlushPendingMacro(system);
    } else if (executing_macro != 0) {
        ASSERT(method == executing_macro + 1);
    }

    // Methods after 0xE00 are special, they're actually triggers for some microcode that was
    // uploaded to the GPU during initialization.
    if (method >= MacroRegistersStart) {
        ProcessMacro(system, method, &method_argument, 1, is_last_call);
        return;
    }

    ASSERT(method < Regs::NUM_REGS && "Invalid Maxwell3D register, increase the size of the Regs structure");
    const u32 argument = ProcessShadowRam(method, method_argument);
    // A state register written with the value it holds changes nothing: no dirty flags (they
    // would make the next draw check that state again) and no method processing, which acts only
    // on executable methods. Macros write the same values over and over. Methods with side
    // effects (execution_mask: draws, clears, constant buffer data and binds, semaphores, ...)
    // always run. EDEN_SKIP_SAME_WRITES=0 turns it off.
    // From Eglantine 15eef40dd1 and 25bd673899 (src/video_core/engines/maxwell_3d.cpp).
    if (!execution_mask[method]) {
        if (!SkipSameValueWrites() || regs.reg_array[method] != argument) {
            ProcessDirtyRegisters(method, argument);
        }
        return;
    }
    ProcessDirtyRegisters(method, argument);
    ProcessMethodCall(method, argument, method_argument, is_last_call);
}

void Maxwell3D::CallMultiMethod(Core::System& system, u32 method, const u32* base_start, u32 amount, u32 methods_pending) {
    if (!front_macros && HasPendingMacro() && !IsMacroData(method)) {
        FlushPendingMacro(system);
    }
    // Methods after 0xE00 are special, they're actually triggers for some microcode that was
    // uploaded to the GPU during initialization.
    if (method >= MacroRegistersStart) {
        ProcessMacro(system, method, base_start, amount, amount == methods_pending);
        return;
    }
    // As in ProcessDirtyRegisters: the paths below write without it
    if (draw_manager.HasDeferredDraw() && !DrawManager::IsInstanceLoopMethod(method)) [[unlikely]] {
        draw_manager.DrawDeferred(*this, true);
    }
    switch (method) {
    case MAXWELL3D_REG_INDEX(const_buffer.buffer):
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 1:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 2:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 3:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 4:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 5:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 6:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 7:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 8:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 9:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 10:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 11:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 12:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 13:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 14:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 15:
        ProcessCBMultiData(base_start, amount);
        break;
    case MAXWELL3D_REG_INDEX(inline_data): {
        ASSERT(methods_pending == amount);
        upload_state.ProcessData(base_start, amount);
        return;
    }
    case MAXWELL3D_REG_INDEX(draw_inline_index):
    case MAXWELL3D_REG_INDEX(inline_index_2x16.even):
    case MAXWELL3D_REG_INDEX(inline_index_4x8.index0):
        if (shadow_state.shadow_ram_control != Regs::ShadowRamControl::Replay) {
            ProcessInlineIndexMultiData(method, base_start, amount);
            break;
        }
        [[fallthrough]];
    default:
        for (u32 i = 0; i < amount; i++) {
            CallMethod(system, method, base_start[i], methods_pending - i <= 1);
        }
        break;
    }
}

void Maxwell3D::ProcessMacroUpload(u32 data) {
    macro_engine.AddCode(regs.load_mme.instruction_ptr++, data);
}

void Maxwell3D::ProcessMacroBind(u32 data) {
    MacroPositions()[regs.load_mme.start_address_ptr++] = data;
}

void Maxwell3D::ProcessFirmwareCall4() {
    LOG_DEBUG(HW_GPU, "(STUBBED) called");

    // Firmware call 4 is a blob that changes some registers depending on its parameters.
    // These registers don't affect emulation and so are stubbed by setting 0xd00 to 1.
    regs.shadow_scratch[0] = 1;
}

void Maxwell3D::StampQueryResult(Core::System& system, u64 payload, bool long_query) {
    const GPUVAddr sequence_address{regs.report_semaphore.Address()};
    if (long_query) {
        memory_manager.Write<u64>(sequence_address + sizeof(u64), system.GPU().GetTicks());
        memory_manager.Write<u64>(sequence_address, payload);
    } else {
        memory_manager.Write<u32>(sequence_address, static_cast<u32>(payload));
    }
}

void Maxwell3D::ProcessQueryGet() {
    VideoCommon::QueryPropertiesFlags flags{};
    if (regs.report_semaphore.query.short_query == 0) {
        flags |= VideoCommon::QueryPropertiesFlags::HasTimeout;
    }
    const GPUVAddr sequence_address{regs.report_semaphore.Address()};
    const VideoCommon::QueryType query_type =
        static_cast<VideoCommon::QueryType>(regs.report_semaphore.query.report.Value());
    const u32 payload = regs.report_semaphore.payload;
    const u32 subreport = regs.report_semaphore.query.sub_report;
    VideoCommon::DrawCapture::Get().EngineWrite("report", sequence_address, 16, payload, false);
    switch (regs.report_semaphore.query.operation) {
    case Regs::ReportSemaphore::Operation::Release:
    case Regs::ReportSemaphore::Operation::ReportOnly:
        if (query_type == VideoCommon::QueryType::Payload &&
            regs.report_semaphore.query.reduction_enable != 0) {
            const SemaphoreRelease release{
                .address = sequence_address,
                .payload = payload,
                .payload_64 = false,
                .timestamp = regs.report_semaphore.query.short_query == 0,
                .reduction = static_cast<SemaphoreReduction>(
                    regs.report_semaphore.query.reduction_op.Value()),
                .reduction_signed = regs.report_semaphore.query.format_signed.Value() == 1,
            };
            ReleaseSemaphore(*rasterizer, memory_manager, release);
            break;
        }
        if (regs.report_semaphore.query.operation == Regs::ReportSemaphore::Operation::Release &&
            regs.report_semaphore.query.short_query != 0) {
            flags |= VideoCommon::QueryPropertiesFlags::IsAFence;
        }
        rasterizer->Query(sequence_address, query_type, flags, payload, subreport);
        break;
    case Regs::ReportSemaphore::Operation::Acquire:
        // TODO(Blinkhawk): Under this operation, the GPU waits for the CPU to write a value that
        // matches the current payload.
        UNIMPLEMENTED_MSG("Unimplemented query operation ACQUIRE");
        break;
    case Regs::ReportSemaphore::Operation::Trap:
        UNIMPLEMENTED_MSG("Unimplemented query operation TRAP");
        break;
    default:
        UNIMPLEMENTED_MSG("Unknown query operation");
        break;
    }
}

void Maxwell3D::ProcessQueryCondition() {
    SCOPE_EXIT {
        if (auto& capture{VideoCommon::DrawCapture::Get()}; capture.Active()) {
            std::array<u32, 4> words{};
            memory_manager.ReadBlockUnsafe(regs.render_enable.Address(), words.data(),
                                           sizeof(words));
            capture.Line(fmt::format("render enable override={} mode={} address={:#x} "
                                     "guest=[{:#x},{:#x},{:#x},{:#x}] execute={}",
                                     static_cast<u32>(regs.render_enable_override),
                                     static_cast<u32>(regs.render_enable.mode),
                                     regs.render_enable.Address(), words[0], words[1], words[2],
                                     words[3], execute_on));
        }
    };
    if (rasterizer->AccelerateConditionalRendering()) {
        execute_on = true;
        return;
    }
    const GPUVAddr condition_address{regs.render_enable.Address()};
    switch (regs.render_enable_override) {
    case Regs::RenderEnable::Override::AlwaysRender:
        execute_on = true;
        break;
    case Regs::RenderEnable::Override::NeverRender:
        execute_on = false;
        break;
    case Regs::RenderEnable::Override::UseRenderEnable: {
        switch (regs.render_enable.mode) {
        case Regs::RenderEnable::Mode::True: {
            execute_on = true;
            break;
        }
        case Regs::RenderEnable::Mode::False: {
            execute_on = false;
            break;
        }
        case Regs::RenderEnable::Mode::Conditional: {
            Regs::ReportSemaphore::Compare cmp;
            memory_manager.ReadBlock(condition_address, &cmp, sizeof(cmp));
            execute_on = cmp.initial_sequence != 0U && cmp.initial_mode != 0U;
            break;
        }
        case Regs::RenderEnable::Mode::IfEqual: {
            Regs::ReportSemaphore::Compare cmp;
            memory_manager.ReadBlock(condition_address, &cmp, sizeof(cmp));
            execute_on = cmp.initial_sequence == cmp.current_sequence &&
                         cmp.initial_mode == cmp.current_mode;
            break;
        }
        case Regs::RenderEnable::Mode::IfNotEqual: {
            Regs::ReportSemaphore::Compare cmp;
            memory_manager.ReadBlock(condition_address, &cmp, sizeof(cmp));
            execute_on = cmp.initial_sequence != cmp.current_sequence ||
                         cmp.initial_mode != cmp.current_mode;
            break;
        }
        default: {
            UNIMPLEMENTED_MSG("Uninplemented Condition Mode!");
            execute_on = true;
            break;
        }
        }
        break;
    }
    }
}

void Maxwell3D::ProcessCounterReset() {
    const auto query_type = [clear_report = regs.clear_report_value]() {
        switch (clear_report) {
        case Tegra::Engines::Maxwell3D::Regs::ClearReport::ZPassPixelCount:
            return VideoCommon::QueryType::ZPassPixelCount64;
        case Tegra::Engines::Maxwell3D::Regs::ClearReport::StreamingPrimitivesSucceeded:
            return VideoCommon::QueryType::StreamingPrimitivesSucceeded;
        case Tegra::Engines::Maxwell3D::Regs::ClearReport::PrimitivesGenerated:
            return VideoCommon::QueryType::PrimitivesGenerated;
        case Tegra::Engines::Maxwell3D::Regs::ClearReport::VtgPrimitivesOut:
            return VideoCommon::QueryType::VtgPrimitivesOut;
        default:
            LOG_DEBUG(HW_GPU, "Unimplemented counter reset={}", clear_report);
            return VideoCommon::QueryType::Payload;
        }
    }();
    rasterizer->ResetCounter(query_type);
}

void Maxwell3D::ProcessSyncPoint() {
    const u32 sync_point = regs.sync_info.sync_point.Value();
    [[maybe_unused]] const u32 cache_flush = regs.sync_info.clean_l2.Value();
    rasterizer->SignalSyncPoint(sync_point);
}

void Maxwell3D::ProcessCBBind(size_t stage_index) {
    // Bind the buffer currently in CB_ADDRESS to the specified index in the desired shader
    // stage.
    const auto& bind_data = regs.bind_groups[stage_index];
    if (bind_data.shader_slot >= Regs::MaxConstBuffers) {
        return;
    }
    auto& buffer = state.shader_stages[stage_index].const_buffers[bind_data.shader_slot];
    buffer.enabled = bind_data.valid.Value() != 0;
    buffer.address = regs.const_buffer.Address();
    buffer.size = regs.const_buffer.size;

    const bool is_enabled = bind_data.valid.Value() != 0;
    if (!is_enabled) {
        rasterizer->DisableGraphicsUniformBuffer(stage_index, bind_data.shader_slot);
        return;
    }
    const GPUVAddr gpu_addr = regs.const_buffer.Address();
    const u32 size = regs.const_buffer.size;
    rasterizer->BindGraphicsUniformBuffer(stage_index, bind_data.shader_slot, gpu_addr, size);
}

void Maxwell3D::ProcessCBMultiData(const u32* start_base, u32 amount) {
    // Write the input value to the current const buffer at the current position.
    const GPUVAddr buffer_address = regs.const_buffer.Address();
    ASSERT(buffer_address != 0);

    // Only the whole words that fit in the buffer are written. A method's data can continue
    // across command list entries, so dropping just the words past the end keeps the contents
    // independent of where the list was split; the offset still advances by the full size, as
    // on the front of a split GPU thread. From Eglantine 217f0229ad
    const size_t copy_size = amount * sizeof(u32);
    const size_t room = regs.const_buffer.offset < regs.const_buffer.size
                            ? regs.const_buffer.size - regs.const_buffer.offset
                            : 0;
    const size_t write_size = std::min(copy_size, room / sizeof(u32) * sizeof(u32));
    if (write_size != 0) {
        const GPUVAddr address{buffer_address + regs.const_buffer.offset};
        memory_manager.WriteBlockCached(address, start_base, write_size);
    }
    if (write_size != copy_size) {
        static std::atomic<bool> warned{};
        if (!warned.exchange(true, std::memory_order_relaxed)) {
            LOG_ERROR(HW_GPU,
                      "Dropping constant buffer words past the end: offset={:#x}, size={:#x}, "
                      "bytes={:#x}, written={:#x}",
                      regs.const_buffer.offset, regs.const_buffer.size, copy_size, write_size);
        }
    }

    // Increment the current buffer position.
    regs.const_buffer.offset += static_cast<u32>(copy_size);
}

void Maxwell3D::ProcessCBData(u32 value) {
    ProcessCBMultiData(&value, 1);
}

void Maxwell3D::ProcessInlineIndexMultiData(u32 method, const u32* start_base, u32 amount) {
    if (amount == 0) {
        return;
    }
    const u32 argument = ProcessShadowRam(method, start_base[amount - 1]);
    ProcessDirtyRegisters(method, argument);
    draw_manager.SetInlineIndexBuffer(*this, method, start_base, amount);
}

Texture::TICEntry Maxwell3D::GetTICEntry(u32 tic_index) const {
    const GPUVAddr tic_address_gpu{regs.tex_header.Address() +
                                   tic_index * sizeof(Texture::TICEntry)};
    Texture::TICEntry tic_entry;
    memory_manager.ReadBlockUnsafe(tic_address_gpu, &tic_entry, sizeof(Texture::TICEntry));
    return tic_entry;
}

Texture::TSCEntry Maxwell3D::GetTSCEntry(u32 tsc_index) const {
    const GPUVAddr tsc_address_gpu{regs.tex_sampler.Address() +
                                   tsc_index * sizeof(Texture::TSCEntry)};
    Texture::TSCEntry tsc_entry;
    memory_manager.ReadBlockUnsafe(tsc_address_gpu, &tsc_entry, sizeof(Texture::TSCEntry));
    return tsc_entry;
}

u32 Maxwell3D::GetRegisterValue(u32 method) const {
    ASSERT(method < Regs::NUM_REGS && "Invalid Maxwell3D register");
    return regs.reg_array[method];
}

void Maxwell3D::SetHLEReplacementAttributeType(u32 bank, u32 offset,
                                               HLEReplacementAttributeType name) {
    const u64 key = (static_cast<u64>(bank) << 32) | offset;
    replace_table.emplace(key, name);
}

} // namespace Tegra::Engines
