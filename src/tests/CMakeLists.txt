# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later

# SPDX-FileCopyrightText: 2018 yuzu Emulator Project
# SPDX-License-Identifier: GPL-2.0-or-later

add_executable(tests
    common/addon_config.cpp
    common/bit_field.cpp
    common/cityhash.cpp
    common/container_hash.cpp
    common/fibers.cpp
    common/host_memory.cpp
    common/param_package.cpp
    common/range_map.cpp
    common/ring_buffer.cpp
    common/scratch_buffer.cpp
    common/undefined_fix.cpp
    common/unique_function.cpp
    core/builtin_patches.cpp
    core/core_timing.cpp
    audio_core/batch_decoder.cpp
    core/graphics_supervisor.cpp
    core/native_gpu_characteristics.cpp
    core/nup_container.cpp
    core/program_metadata.cpp
    core/internal_network/network.cpp
    shader_recompiler/ampere_decode.cpp
    shader_recompiler/ampere_decode_compiler_vectors.inc
    shader_recompiler/ampere_decode_vectors.inc
    shader_recompiler/ampere_hardware_vectors.inc
    shader_recompiler/ampere_kernel_converge.inc
    shader_recompiler/ampere_kernel_texk.inc
    shader_recompiler/ampere_kernel_warp.inc
    shader_recompiler/ampere_corpus.cpp
    shader_recompiler/ampere_translate.cpp
    video_core/macro_turing.cpp
    video_core/memory_tracker.cpp
    video_core/native_swizzle.cpp
    video_core/qmd_v3.cpp
    video_core/semaphore.cpp
    video_core/texture_header.cpp
    input_common/calibration_configuration_job.cpp
)

create_target_directory_groups(tests)

target_link_libraries(tests PRIVATE common core audio_core input_common video_core frontend_common)
target_link_libraries(tests PRIVATE ${PLATFORM_LIBRARIES} Catch2::Catch2WithMain Threads::Threads)
# The NUP test writes containers with SHA-256 hashes, as the reader checks them
if (TARGET OpenSSL::Crypto)
    target_link_libraries(tests PRIVATE OpenSSL::Crypto)
endif()

add_test(NAME tests COMMAND tests)

# needed for vma
if (NOT MSVC)
    target_compile_options(tests PRIVATE
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wno-conversion>
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wno-unused-variable>
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wno-unused-parameter>
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wno-missing-field-initializers>)
endif()
