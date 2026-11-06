// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2018 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>
#include "common/bit_field.h"
#include "common/common_funcs.h"
#include "common/common_types.h"
#include "video_core/engines/engine_interface.h"
#include "video_core/engines/engine_upload.h"
#include "video_core/macro_turing.h"
#include "video_core/textures/texture.h"

namespace Core {
class System;
}

namespace Tegra {
class MemoryManager;
}

namespace VideoCore {
class RasterizerInterface;
}

namespace Tegra::Engines {

/**
 * This Engine is known as GK104_Compute. Documentation can be found in:
 * https://github.com/envytools/envytools/blob/master/rnndb/graph/gk104_compute.xml
 * https://cgit.freedesktop.org/mesa/mesa/tree/src/gallium/drivers/nouveau/nvc0/nve4_compute.xml.h
 */

#define KEPLER_COMPUTE_REG_INDEX(field_name)                                                       \
    (offsetof(Tegra::Engines::KeplerCompute::Regs, field_name) / sizeof(u32))

#define LAUNCH_REG_INDEX(field_name)                                                               \
    (offsetof(Tegra::Engines::KeplerCompute::LaunchParams, field_name) / sizeof(u32))

struct QmdV3;

class KeplerCompute final : public EngineInterface {
public:
    explicit KeplerCompute(MemoryManager& memory_manager);
    ~KeplerCompute();

    /// Binds a rasterizer to this engine.
    void BindRasterizer(VideoCore::RasterizerInterface* rasterizer);

    static constexpr std::size_t NumConstBuffers = 8;

    struct Regs {
        // Up to the macro calls, which leaves room for SET_MME_SHADOW_SCRATCH of AMPERE_COMPUTE_B
        static constexpr std::size_t NUM_REGS = 0xE00;

        union {
            struct {
                INSERT_PADDING_WORDS_NOINIT(0x60);

                Upload::Registers upload;

                struct {
                    union {
                        BitField<0, 1, u32> linear;
                    };
                } exec_upload;

                u32 data_upload;

                INSERT_PADDING_WORDS_NOINIT(0x3F);

                struct {
                    u32 address;
                    GPUVAddr Address() const {
                        return GPUVAddr{address} << 8;
                    }
                } launch_desc_loc;

                INSERT_PADDING_WORDS_NOINIT(0x1);

                u32 launch;

                INSERT_PADDING_WORDS_NOINIT(0x4A7);

                struct {
                    u32 address_high;
                    u32 address_low;
                    u32 limit;
                    GPUVAddr Address() const {
                        return (GPUVAddr{address_high} << 32) | GPUVAddr{address_low};
                    }
                } tsc;

                INSERT_PADDING_WORDS_NOINIT(0x3);

                struct {
                    u32 address_high;
                    u32 address_low;
                    u32 limit;
                    GPUVAddr Address() const {
                        return (GPUVAddr{address_high} << 32) | GPUVAddr{address_low};
                    }
                } tic;

                INSERT_PADDING_WORDS_NOINIT(0x22);

                struct {
                    u32 address_high;
                    u32 address_low;
                    GPUVAddr Address() const {
                        return (GPUVAddr{address_high} << 32) | GPUVAddr{address_low};
                    }
                } code_loc;

                INSERT_PADDING_WORDS_NOINIT(0x3FE);

                u32 tex_cb_index;

                INSERT_PADDING_WORDS_NOINIT(0x47C);
            };
            std::array<u32, NUM_REGS> reg_array;
        };
    } regs{};

    struct LaunchParams {
        static constexpr std::size_t NUM_LAUNCH_PARAMETERS = 0x40;

        INSERT_PADDING_WORDS(0x8);

        u32 program_start;

        INSERT_PADDING_WORDS(0x2);

        BitField<30, 1, u32> linked_tsc;

        BitField<0, 31, u32> grid_dim_x;
        union {
            BitField<0, 16, u32> grid_dim_y;
            BitField<16, 16, u32> grid_dim_z;
        };

        INSERT_PADDING_WORDS(0x3);

        BitField<0, 18, u32> shared_alloc;

        BitField<16, 16, u32> block_dim_x;
        union {
            BitField<0, 16, u32> block_dim_y;
            BitField<16, 16, u32> block_dim_z;
        };

        union {
            BitField<0, 8, u32> const_buffer_enable_mask;
            BitField<29, 2, u32> cache_layout;
        };

        INSERT_PADDING_WORDS(0x8);

        struct ConstBufferConfig {
            u32 address_low;
            union {
                BitField<0, 8, u32> address_high;
                BitField<15, 17, u32> size;
            };
            GPUVAddr Address() const {
                return (GPUVAddr{address_high.Value()} << 32) | GPUVAddr{address_low};
            }
        };
        std::array<ConstBufferConfig, NumConstBuffers> const_buffer_config;

        union {
            BitField<0, 20, u32> local_pos_alloc;
            BitField<27, 5, u32> barrier_alloc;
        };

        union {
            BitField<0, 20, u32> local_neg_alloc;
            BitField<24, 5, u32> gpr_alloc;
        };

        union {
            BitField<0, 20, u32> local_crs_alloc;
            BitField<24, 5, u32> sass_version;
        };

        INSERT_PADDING_WORDS(0x10);
    } launch_description{};

    struct {
        u32 write_offset = 0;
        u32 copy_size = 0;
        std::vector<u8> inner_buffer;
    } state{};

    static_assert(sizeof(Regs) == Regs::NUM_REGS * sizeof(u32),
                  "KeplerCompute Regs has wrong size");

    static_assert(sizeof(LaunchParams) == LaunchParams::NUM_LAUNCH_PARAMETERS * sizeof(u32),
                  "KeplerCompute LaunchParams has wrong size");

    /// Write the value to the register identified by method.
    void CallMethod(Core::System& system, u32 method, u32 method_argument, bool is_last_call) override;

    /// Write multiple values to the register identified by method.
    void CallMultiMethod(Core::System& system, u32 method, const u32* base_start, u32 amount, u32 methods_pending) override;

    bool HasPendingMacro() const override {
        return executing_macro != 0;
    }
    bool IsMacroData(u32 method) const override;
    /// Runs a deferred native macro call with the parameters it has received.
    void FlushPendingMacro(Core::System& system) override;

    std::optional<GPUVAddr> GetIndirectComputeAddress() const {
        return indirect_compute;
    }

    /// Binds the engine as AMPERE_COMPUTE_B (0xC7C0) instead of KEPLER_COMPUTE_B
    void SetAmpere(bool is_ampere_) {
        is_ampere = is_ampere_;
    }

    [[nodiscard]] bool IsAmpere() const {
        return is_ampere;
    }

    bool SupportsFrontMacros() const override {
        return is_ampere;
    }
    void EnableFrontMacros() override;
    bool MacrosOnFront() const override {
        return front_macros;
    }
    bool IsFrontMethod(u32 method) const override;
    void FrontCallMethod(Core::System& system, u32 method, u32 argument) override;
    void FrontCallMultiMethod(Core::System& system, u32 method, const u32* base_start,
                              u32 amount) override;
    void FrontWrite(u32 method, u32 argument) override;
    void FrontMultiWrite(u32 method, const u32* base_start, u32 amount) override;
    /// Tells the split what the method the front sends next may write (GpuSplit::Classify)
    void FrontClassify(u32 method);

    /// Uses the macros of the TSG, which the 3D and compute classes share
    void SetNativeMacroContext(std::shared_ptr<NativeMacroContext> context) {
        native_macro_context = std::move(context);
    }

    /// Base address and start offset of the program of the current launch
    [[nodiscard]] std::pair<GPUVAddr, u32> ShaderProgramLocation() const {
        if (is_ampere) {
            return {ampere_program_address, 0};
        }
        return {regs.code_loc.Address(), launch_description.program_start};
    }

private:
    void ProcessLaunch();

    /// Handles a PCAS action of AMPERE_COMPUTE_B on a QMD address
    void SignalPcas(GPUVAddr address, u32 action, u32 count);

    /// Launches an inline QMD 3.0 loaded via LOAD_INLINE_QMD_DATA(0..63)
    void ExecuteInlineQmd();

    /// Launches a QMD 3.0 whose dependencies are met and performs its releases
    void ExecuteQmd(const QmdV3& qmd);

    void ConsumeSinkImpl(Core::System& system) override;

    /// Collects the parameters of a CALL_MME_MACRO and runs it after the last one
    void ProcessMacro(Core::System& system, u32 method, const u32* base_start, u32 amount,
                      bool is_last_call);

    TuringMacroEngine& TuringMacro() {
        return native_macro_context ? native_macro_context->engine : turing_macro;
    }
    std::array<u32, 0x80>& MacroPositions() {
        return native_macro_context ? native_macro_context->positions : macro_positions;
    }

    /// Retrieves information about a specific TIC entry from the TIC buffer.
    Texture::TICEntry GetTICEntry(u32 tic_index) const;

    /// Retrieves information about a specific TSC entry from the TSC buffer.
    Texture::TSCEntry GetTSCEntry(u32 tsc_index) const;

    MemoryManager& memory_manager;
    VideoCore::RasterizerInterface* rasterizer = nullptr;
    Upload::State upload_state;
    GPUVAddr upload_address;
    bool upload_dirty{};

    struct UploadInfo {
        GPUVAddr upload_address;
        GPUVAddr exec_address;
        u32 copy_size;
        bool was_dirty;
    };
    std::vector<UploadInfo> uploads;
    std::optional<GPUVAddr> indirect_compute{};
    bool is_ampere{};

    /// Program address of the last QMD 3.0 launch, which is absolute instead of relative to
    /// code_loc
    GPUVAddr ampere_program_address{};

    /// Macro engine of AMPERE_COMPUTE_B, used when the channel has no TSG context
    TuringMacroEngine turing_macro;
    std::array<u32, 0x80> macro_positions{};
    u32 macro_position_pointer{};
    std::shared_ptr<NativeMacroContext> native_macro_context;
    /// CALL_MME_MACRO method being fed parameters, or 0
    u32 executing_macro{};
    std::vector<u32> macro_params;

    void FrontFlushMacro(Core::System& system);
    bool front_macros{};
    std::unique_ptr<Regs> front_regs; ///< The registers as the front of a split GPU thread sent them
};

#define ASSERT_REG_POSITION(field_name, position)                                                  \
    static_assert(offsetof(KeplerCompute::Regs, field_name) == position * 4,                       \
                  "Field " #field_name " has invalid position")

#define ASSERT_LAUNCH_PARAM_POSITION(field_name, position)                                         \
    static_assert(offsetof(KeplerCompute::LaunchParams, field_name) == position * 4,               \
                  "Field " #field_name " has invalid position")

ASSERT_REG_POSITION(upload, 0x60);
ASSERT_REG_POSITION(exec_upload, 0x6C);
ASSERT_REG_POSITION(data_upload, 0x6D);
ASSERT_REG_POSITION(launch, 0xAF);
ASSERT_REG_POSITION(tsc, 0x557);
ASSERT_REG_POSITION(tic, 0x55D);
ASSERT_REG_POSITION(code_loc, 0x582);
ASSERT_REG_POSITION(tex_cb_index, 0x982);
ASSERT_LAUNCH_PARAM_POSITION(program_start, 0x8);
ASSERT_LAUNCH_PARAM_POSITION(grid_dim_x, 0xC);
ASSERT_LAUNCH_PARAM_POSITION(shared_alloc, 0x11);
ASSERT_LAUNCH_PARAM_POSITION(block_dim_x, 0x12);
ASSERT_LAUNCH_PARAM_POSITION(const_buffer_enable_mask, 0x14);
ASSERT_LAUNCH_PARAM_POSITION(const_buffer_config, 0x1D);

#undef ASSERT_REG_POSITION

} // namespace Tegra::Engines
