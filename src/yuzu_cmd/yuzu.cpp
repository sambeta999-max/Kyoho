// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: 2014 Citra Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <iostream>
#include <memory>
#include <mutex>
#include <regex>
#include <string>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <thread>
#include <vector>
#include "common/settings_enums.h"
#include "core/launch_params.h"
#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif
#define SDL_MAIN_USE_CALLBACKS 1
#include <SDL3/SDL_main.h>

#include <fmt/ostream.h>

#include "common/logging.h"
#include "common/scm_rev.h"
#include "common/settings.h"
#include "common/string_util.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "core/cpu_manager.h"
#include "core/crypto/key_manager.h"
#include "core/file_sys/registered_cache.h"
#include "core/file_sys/vfs/vfs_real.h"
#include "core/hle/service/am/applet_manager.h"
#include "core/hle/service/filesystem/filesystem.h"
#include "core/tools/renderdoc.h"
#include "core/loader/loader.h"
#include "frontend_common/config.h"
#include "core/frontend/framebuffer_layout.h"
#include "input_common/drivers/keyboard.h"
#include "input_common/main.h"
#include "video_core/renderer_base.h"
#include "network/network.h"
#include "sdl_config.h"
#include "video_core/draw_capture.h"
#include "video_core/gpu.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/renderer_base.h"
#include "yuzu_cmd/emu_window/emu_window_sdl3.h"
#ifdef HAS_OPENGL
#include "yuzu_cmd/emu_window/emu_window_sdl3_gl.h"
#endif
#include "yuzu_cmd/emu_window/emu_window_sdl3_null.h"
#include "yuzu_cmd/emu_window/emu_window_sdl3_vk.h"

#ifdef _WIN32
// windows.h needs to be included before shellapi.h
#include <windows.h>
#include <shellapi.h>
#include "common/windows/timer_resolution.h"
#endif

#ifdef _WIN32
extern "C" {
// tells Nvidia and AMD drivers to use the dedicated GPU by default on laptops with switchable
// graphics
__declspec(dllexport) unsigned long NvOptimusEnablement = 0x00000001;
__declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
}
#endif

static void OnStateChanged(const Network::RoomMember::State& state) {
    switch (state) {
    case Network::RoomMember::State::Idle:
        LOG_DEBUG(Network, "Network is idle");
        break;
    case Network::RoomMember::State::Joining:
        LOG_DEBUG(Network, "Connection sequence to room started");
        break;
    case Network::RoomMember::State::Joined:
        LOG_DEBUG(Network, "Successfully joined to the room");
        break;
    case Network::RoomMember::State::Moderator:
        LOG_DEBUG(Network, "Successfully joined the room as a moderator");
        break;
    default:
        break;
    }
}

static void OnNetworkError(const Network::RoomMember::Error& error) {
    switch (error) {
    case Network::RoomMember::Error::LostConnection:
        LOG_DEBUG(Network, "Lost connection to the room");
        break;
    case Network::RoomMember::Error::CouldNotConnect:
        LOG_ERROR(Network, "Error: Could not connect");
        exit(1);
        break;
    case Network::RoomMember::Error::NameCollision:
        LOG_ERROR(
            Network,
            "You tried to use the same nickname as another user that is connected to the Room");
        exit(1);
        break;
    case Network::RoomMember::Error::IpCollision:
        LOG_ERROR(Network, "You tried to use the same fake IP-Address as another user that is "
                           "connected to the Room");
        exit(1);
        break;
    case Network::RoomMember::Error::WrongPassword:
        LOG_ERROR(Network, "Room replied with: Wrong password");
        exit(1);
        break;
    case Network::RoomMember::Error::WrongVersion:
        LOG_ERROR(Network,
                  "You are using a different version than the room you are trying to connect to");
        exit(1);
        break;
    case Network::RoomMember::Error::RoomIsFull:
        LOG_ERROR(Network, "The room is full");
        exit(1);
        break;
    case Network::RoomMember::Error::HostKicked:
        LOG_ERROR(Network, "You have been kicked by the host");
        break;
    case Network::RoomMember::Error::HostBanned:
        LOG_ERROR(Network, "You have been banned by the host");
        break;
    case Network::RoomMember::Error::UnknownError:
        LOG_ERROR(Network, "UnknownError");
        break;
    case Network::RoomMember::Error::PermissionDenied:
        LOG_ERROR(Network, "PermissionDenied");
        break;
    case Network::RoomMember::Error::NoSuchUser:
        LOG_ERROR(Network, "NoSuchUser");
        break;
    }
}

static void OnMessageReceived(const Network::ChatEntry& msg) {
    std::cout << std::endl << msg.nickname << ": " << msg.message << std::endl << std::endl;
}

static void OnStatusMessageReceived(const Network::StatusMessageEntry& msg) {
    std::string message = [&]() {
        switch (msg.type) {
        case Network::IdMemberJoin:
            return fmt::format("{} has joined", msg.nickname);
        case Network::IdMemberLeave:
            return fmt::format("{} has left", msg.nickname);
        case Network::IdMemberKicked:
            return fmt::format("{} has been kicked", msg.nickname);
        case Network::IdMemberBanned:
            return fmt::format("{} has been banned", msg.nickname);
        case Network::IdAddressUnbanned:
            return fmt::format("{} has been unbanned", msg.nickname);
        default:
            return std::string{};
        }
    }();
    if (!message.empty())
        std::cout << std::endl << "* " << message << std::endl << std::endl;
}

struct SdlState {
    Core::System system{};
    InputCommon::InputSubsystem input_subsystem{};
    std::unique_ptr<EmuWindow_SDL3> emu_window;
    std::jthread script;
};

namespace {
// Writes a 32-bit BMP from rows of B, G, R, A bytes (the renderer's screenshot layout)
void WriteBmp(const std::string& path, const std::vector<u8>& pixels, u32 width, u32 height,
              bool invert_y) {
    std::ofstream file(path, std::ios::binary);
    const u32 image_size = width * height * 4;
    const auto put32 = [&](u32 v) { file.write(reinterpret_cast<const char*>(&v), 4); };
    const auto put16 = [&](u16 v) { file.write(reinterpret_cast<const char*>(&v), 2); };
    file.write("BM", 2);
    put32(54 + image_size);
    put32(0);
    put32(54);
    put32(40);
    put32(width);
    // A negative height stores the rows top to bottom
    put32(static_cast<u32>(invert_y ? static_cast<s32>(height) : -static_cast<s32>(height)));
    put16(1);
    put16(32);
    put32(0);
    put32(image_size);
    put32(2835);
    put32(2835);
    put32(0);
    put32(0);
    file.write(reinterpret_cast<const char*>(pixels.data()), image_size);
}

// EDEN_SCRIPT="k48,b52,s82,...": at each time (seconds after boot) press a button or take a
// screenshot, inside the emulator, so scripted runs neither need the window in front nor send
// keys to other windows. The buttons are the keyboard keys of the cli-key.ini mappings:
// k/b/x/y/p = A/B/X/Y/+, q/e/w/m = L/R/L+R/-, u/d/l/r = D-pad, n/v/f/h = left stick
// up/down/left/right (held 1.5 s).
// Screenshots go to EDEN_SCRIPT_OUT (a path prefix) as <prefix>_<sec>s.bmp; z takes a burst of
// consecutive frames (EDEN_SCRIPT_BURST).
void RunScript(std::stop_token stop, SdlState* state, std::string script, std::string out) {
    const auto start = std::chrono::steady_clock::now();
    size_t pos = 0;
    while (pos < script.size() && !stop.stop_requested()) {
        const size_t end = (std::min)(script.find(',', pos), script.size());
        const std::string action = script.substr(pos, end - pos);
        pos = end + 1;
        if (action.size() < 2) {
            continue;
        }
        const char kind = action[0];
        const double at = std::atof(action.c_str() + 1);
        while (!stop.stop_requested() &&
               std::chrono::steady_clock::now() - start < std::chrono::duration<double>(at)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        if (stop.stop_requested()) {
            return;
        }
        LOG_INFO(Frontend, "script: {}", action);
        if (kind == 'g') {
            // RenderDoc capture of the next presented frame (run under renderdoccmd with
            // renderdoc_hotkey enabled)
            state->system.GetRenderdocAPI().TriggerCapture(1);
            continue;
        }
        if (kind == 'c') {
            // Draw capture from the next frame on (video_core/draw_capture.h); EDEN_DRAW_FRAMES
            // sets how many frames
            const char* frames = std::getenv("EDEN_DRAW_FRAMES");
            VideoCommon::DrawCapture::Get().Request(
                frames ? static_cast<u32>((std::max)(1, std::atoi(frames))) : 1U);
            continue;
        }
        if (kind == 'z') {
            // EDEN_SCRIPT_BURST (default 8) screenshots of consecutive presented frames, as
            // <prefix>_<sec>s_<n>.bmp, to see flicker between frames
            const char* burst = std::getenv("EDEN_SCRIPT_BURST");
            const int count = burst ? (std::max)(1, std::atoi(burst)) : 8;
            const auto layout = Layout::DefaultFrameLayout(1280, 720);
            for (int i = 0; i < count && !stop.stop_requested(); ++i) {
                auto pixels =
                    std::make_shared<std::vector<u8>>(size_t{layout.width} * layout.height * 4);
                auto done = std::make_shared<std::atomic_bool>(false);
                const std::string path = fmt::format("{}_{}s_{}.bmp", out, action.substr(1), i);
                state->system.Renderer().RequestScreenshot(
                    pixels->data(),
                    [pixels, path, layout, done](bool invert_y) {
                        WriteBmp(path, *pixels, layout.width, layout.height, invert_y);
                        done->store(true);
                    },
                    layout);
                // With EDEN_SCREENSHOT_NOWAIT the next request goes in once this one is taken
                const bool nowait = std::getenv("EDEN_SCREENSHOT_NOWAIT") != nullptr;
                while (!stop.stop_requested() &&
                       (nowait ? state->system.Renderer().IsScreenshotPending() : !done->load())) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            }
            continue;
        }
        if (kind == 's') {
            const auto layout = Layout::DefaultFrameLayout(1280, 720);
            auto pixels = std::make_shared<std::vector<u8>>(size_t{layout.width} * layout.height * 4);
            const std::string path = fmt::format("{}_{}s.bmp", out, action.substr(1));
            state->system.Renderer().RequestScreenshot(
                pixels->data(),
                [pixels, path, layout](bool invert_y) {
                    WriteBmp(path, *pixels, layout.width, layout.height, invert_y);
                    LOG_INFO(Frontend, "script: screenshot {}", path);
                },
                layout);
            continue;
        }
        int scancode = 0;
        int scancode2 = 0; // pressed together with the first (w = L+R)
        int hold_ms = 200;
        switch (kind) {
        case 'q': scancode = 20; break; // L (Q)
        case 'e': scancode = 8; break;  // R (E)
        case 'w': scancode = 20; scancode2 = 8; break; // L+R
        case 'm': scancode = 16; break; // - (M)
        case 'k': scancode = 4; break;   // A key
        case 'b': scancode = 22; break;  // S
        case 'x': scancode = 27; break;  // X
        case 'y': scancode = 28; break;  // Y
        case 'p': scancode = 19; break;  // P
        case 'u': scancode = 12; break;  // I
        case 'd': scancode = 14; break;  // K
        case 'l': scancode = 13; break;  // J
        case 'r': scancode = 15; break;  // L
        case 'n': scancode = 23; hold_ms = 1500; break; // T
        case 'v': scancode = 10; hold_ms = 1500; break; // G
        case 'f': scancode = 9; hold_ms = 1500; break;  // F
        case 'h': scancode = 11; hold_ms = 1500; break; // H
        default: continue;
        }
        auto* keyboard = state->input_subsystem.GetKeyboard();
        keyboard->PressKey(scancode);
        if (scancode2) {
            keyboard->PressKey(scancode2);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(hold_ms));
        keyboard->ReleaseKey(scancode);
        if (scancode2) {
            keyboard->ReleaseKey(scancode2);
        }
    }
}
} // namespace

extern "C" SDL_AppResult SDL_AppInit(void **appstate, int argc, char **argv) {
    SdlState* state = new SdlState();
    *appstate = state;

#ifdef _WIN32
    _setmaxstdio(8192);
    if (AttachConsole(ATTACH_PARENT_PROCESS)) {
        freopen("CONOUT$", "wb", stdout);
        freopen("CONOUT$", "wb", stderr);
    }
#endif

    Common::Log::Initialize();
    Common::Log::SetColorConsoleBackendEnabled(true);
    Common::Log::Start();

    int option_index = 0;
#ifdef _WIN32
    int argc_w;
    auto argv_w = CommandLineToArgvW(GetCommandLineW(), &argc_w);
    if (argv_w == nullptr) {
        LOG_CRITICAL(Frontend, "Failed to get command line arguments");
        return SDL_APP_FAILURE;
    }
#else
    wchar_t **argv_w = nullptr;
#endif
    Core::LaunchParams lp = Core::ParseLaunchParams(state->system, argc, argv, argv_w);
#ifdef _WIN32
    LocalFree(argv_w);
#endif
    SdlConfig config{lp.config_path};

    Core::ApplyLaunchParams(lp);

    if (lp.filepath.empty()) {
        LOG_CRITICAL(Frontend, "Failed to load ROM: No ROM specified");
        return SDL_APP_FAILURE;
    }

    state->system.Initialize();

    // Apply the command line arguments
    state->system.ApplySettings();

    switch (Settings::values.renderer_backend.GetValue()) {
#ifdef HAS_OPENGL
    case Settings::RendererBackend::OpenGL_GLSL:
    case Settings::RendererBackend::OpenGL_GLASM:
    case Settings::RendererBackend::OpenGL_SPIRV:
        state->emu_window = std::make_unique<EmuWindow_SDL3_GL>(&state->input_subsystem, state->system, lp.fullscreen);
        break;
#endif
    case Settings::RendererBackend::Vulkan:
        state->emu_window = std::make_unique<EmuWindow_SDL3_VK>(&state->input_subsystem, state->system, lp.fullscreen);
        break;
    case Settings::RendererBackend::Null:
        state->emu_window = std::make_unique<EmuWindow_SDL3_Null>(&state->input_subsystem, state->system, lp.fullscreen);
        break;
    default:
        LOG_CRITICAL(Frontend, "Invalid renderer backend");
        return SDL_APP_FAILURE;
    }

#ifdef _WIN32
    Common::Windows::SetCurrentTimerResolutionToMaximum();
    state->system.CoreTiming().SetTimerResolutionNs(Common::Windows::GetCurrentTimerResolution());
#endif

    state->system.SetContentProvider(std::make_unique<FileSys::ContentProviderUnion>());
    state->system.SetFilesystem(std::make_shared<FileSys::RealVfsFilesystem>());
    state->system.GetFileSystemController().CreateFactories(*state->system.GetFilesystem());
    state->system.GetUserChannel().clear();

    Service::AM::FrontendAppletParameters load_parameters{
        .applet_id = Service::AM::AppletId::Application,
    };
    const Core::SystemResultStatus load_result = state->system.Load(*state->emu_window, lp.filepath, load_parameters);
    switch (load_result) {
    case Core::SystemResultStatus::Success:
        break; // Expected case
    case Core::SystemResultStatus::ErrorGetLoader:
        LOG_CRITICAL(Frontend, "Failed to obtain loader for {}!", lp.filepath);
        return SDL_APP_FAILURE;
    case Core::SystemResultStatus::ErrorLoader:
        LOG_CRITICAL(Frontend, "Failed to load ROM!");
        return SDL_APP_FAILURE;
    case Core::SystemResultStatus::ErrorNotInitialized:
        LOG_CRITICAL(Frontend, "CPUCore not initialized");
        return SDL_APP_FAILURE;
    case Core::SystemResultStatus::ErrorVideoCore:
        LOG_CRITICAL(Frontend, "Failed to initialize VideoCore!");
        return SDL_APP_FAILURE;
    default:
        const u16 loader_id = u16(Core::SystemResultStatus::ErrorLoader);
        const u16 error_id = u16(load_result) - loader_id;
        LOG_CRITICAL(Frontend,
            "While attempting to load the ROM requested, an error occurred. Please "
            "refer to the Eden wiki for more information or the Eden discord for "
            "additional help.\n\nError Code: {:04X}-{:04X}\nError Description: {}",
            loader_id, error_id, Loader::ResultStatus(error_id));
        return SDL_APP_FAILURE;
    }

    if (lp.use_multiplayer) {
        if (auto member = Network::GetRoomMember().lock()) {
            member->BindOnChatMessageReceived(OnMessageReceived);
            member->BindOnStatusMessageReceived(OnStatusMessageReceived);
            member->BindOnStateChanged(OnStateChanged);
            member->BindOnError(OnNetworkError);
            LOG_DEBUG(Network, "Start connection to {}:{} with nickname {}", lp.address, lp.port, lp.nickname);
            member->Join(lp.nickname, lp.address.c_str(), lp.port, 0, Network::NoPreferredIP, lp.password);
        } else {
            LOG_ERROR(Network, "Could not access RoomMember");
            return SDL_APP_FAILURE;
        }
    }

    // EDEN_DUMP_SHADERS=1 dumps guest shaders (the setting cannot be saved in the ini)
    if (std::getenv("EDEN_DUMP_SHADERS")) {
        Settings::values.dump_guest_shaders.SetValue(true);
    }

    // Core is loaded, start the GPU (makes the GPU contexts current to this thread)
    state->system.GPU().Start();
    state->system.GetCpuManager().OnGpuReady();

    if (Settings::values.use_disk_shader_cache.GetValue()) {
        state->system.Renderer().ReadRasterizer()->LoadDiskResources(
            state->system.GetApplicationProcessProgramID(), std::stop_token{},
            [](VideoCore::LoadCallbackStage, size_t value, size_t total) {});
    }

    // don't do anything, SDL3 already exists for us :D
    state->system.RegisterExitCallback([] {});

    // QLaunch launched applications (should) now reload shader cache.
    static std::mutex shader_cache_reload_mutex;
    state->system.RegisterApplicationChangedCallback([state](u64 changed_program_id) {
        if (!Settings::values.use_disk_shader_cache.GetValue()) {
            return;
        }

        std::scoped_lock lk{shader_cache_reload_mutex};

        LOG_INFO(Frontend, "Reloading disk shader cache for {:016X}", changed_program_id);

        state->system.Pause();
        state->system.GPU().WaitForIdle();
        state->system.GPU().ObtainContext();

        state->system.Renderer().ReadRasterizer()->LoadDiskResources(
            changed_program_id, std::stop_token{},
            [](VideoCore::LoadCallbackStage, size_t, size_t) {});

        state->system.GPU().ReleaseContext();
        state->system.Run();
    });

    void(state->system.Run());
    if (state->system.DebuggerEnabled())
        state->system.InitializeDebugger();
    if (const char* script = std::getenv("EDEN_SCRIPT")) {
        const char* out = std::getenv("EDEN_SCRIPT_OUT");
        state->script = std::jthread(RunScript, state, std::string{script},
                                     std::string{out ? out : "shot"});
    }
    return SDL_APP_CONTINUE;
}
extern "C" SDL_AppResult SDL_AppIterate(void *appstate) {
    SdlState *state = (SdlState *)appstate;
    return state->emu_window->IsOpen() ? SDL_APP_CONTINUE : SDL_APP_SUCCESS;
}
extern "C" SDL_AppResult SDL_AppEvent(void *appstate, SDL_Event *event) {
    SdlState *state = (SdlState *)appstate;
    state->emu_window->OnEvent(*event);
    return SDL_APP_CONTINUE;
}
extern "C" void SDL_AppQuit(void *appstate, SDL_AppResult result) {
    SdlState *state = (SdlState *)appstate;
    if (!state) {
        return;
    }
    state->system.DetachDebugger();
    void(state->system.Pause());
    state->system.ShutdownMainProcess();
    delete state;
}

#define VMA_IMPLEMENTATION
#include "video_core/vulkan_common/vma.h"
