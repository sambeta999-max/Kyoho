// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <atomic>
#include <bit>
#include <array>
#include "common/alignment.h"
#include "core/hle/service/nvdrv/core/nvmap.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "core/hle/service/nvnflinger/buffer_queue_producer.h"
#include "core/hle/service/nvnflinger/ui/graphic_buffer.h"
#include "core/hle/kernel/k_client_session.h"
#include "core/hle/kernel/k_scoped_resource_reservation.h"
#include "core/hle/kernel/k_session.h"
#include "core/hle/kernel/k_readable_event.h"
#include "core/hle/kernel/k_transfer_memory.h"
#include "core/hle/kernel/k_event.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/svc_results.h"
#include "core/hle/service/am/service/graphics_supervisor.h"
#include "core/hle/service/am/service/graphics_session.h"
#include "core/hle/service/am/service/gpu_session.h"
#include "core/hle/service/am/service/video_codec.h"
#include "video_core/host1x/host1x.h"
#include "video_core/pte_kind.h"
#include "video_core/textures/decoders.h"
#include "core/hle/service/am/service/graphics_supervisor_protocol.h"
#include "core/hle/service/am/service/graphics_syncpoint_cache.h"
#include "core/hle/service/am/service/native_image_layout.h"
#include "core/hle/service/ipc_helpers.h"
#include "core/hle/service/nvdrv/devices/nvhost_as_gpu.h"
#include "core/hle/service/nvdrv/nvdrv_interface.h"
#include "core/hle/service/nvdrv/core/container.h"
#include "core/hle/service/nvdrv/core/syncpoint_manager.h"
#include "core/hle/service/server_manager.h"
#include "core/hle/service/sm/sm.h"
#include "core/hle/service/vi/application_display_service.h"
#include "core/hle/service/vi/container.h"
#include "core/hle/service/vi/manager_root_service.h"
#include "core/hle/service/vi/vi_types.h"

namespace Service::AM {
struct NativeImageRegistry {
    struct Image {
        u32 id;
        u32 memory_id;
        std::array<u32, 16> preference;
        u64 size;
        bool window_pool{};
    };
    NativeImageRegistry(Kernel::KernelCore& kernel_, Kernel::KProcess* process_)
        : kernel{kernel_}, process{process_} {
        process->Open(kernel);
    }
    ~NativeImageRegistry() {
        if (module) {
            for (const auto& image : images) {
                ReleaseMemory(image.memory_id);
            }
        }
        if (memory_session) {
            module->GetContainer().CloseSession(*memory_session);
        }
        process->Close(kernel);
    }
    void ReleaseMemory(u32 id) {
        const auto info = module->GetContainer().GetNvMapFile().FreeHandle(id, true);
        if (info && info->can_unlock && info->address != 0) {
            process->GetPageTable().UnlockForDeviceAddressSpace(info->address, info->size);
        }
    }
    Kernel::KernelCore& kernel;
    Kernel::KProcess* process;
    std::mutex mutex;
    std::shared_ptr<Nvidia::Module> module;
    std::optional<Nvidia::NvCore::SessionId> memory_session;
    std::vector<Image> images;
};
using NativeImage = NativeImageRegistry::Image;


namespace {

// TIPC payloads follow a 32-bit result, so these handlers push result.raw.

// CMIF camera session returned by OpenCameraSession. No camera device is emulated, so
// opening one by name reports 2452-0006, the result applications accept for a missing camera.
class ICameraSession final : public ServiceFramework<ICameraSession> {
public:
    explicit ICameraSession(Core::System& system) : ServiceFramework{system, "ICameraSession"} {
        static const FunctionInfo functions[] = {
            {3, &ICameraSession::OpenCamera, "OpenCamera"},
        };
        RegisterHandlers(functions);
    }

private:
    static constexpr Result ResultCameraNotFound{static_cast<ErrorModule>(452), 6};

    void OpenCamera(HLERequestContext& ctx) {
        IPC::RequestParser rp{ctx};
        const auto name = rp.PopRaw<std::array<char, 0x10>>();
        LOG_INFO(Service_AM, "no camera named {}", std::string_view(name.data(), strnlen(name.data(), name.size())));
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(ResultCameraNotFound);
    }
};

// TIPC session of nn::chat::gsv, which exchanges GameChat video images. It opens with the
// notification protocol of IVi2Session (commands 1 and 11). The emulator never reports a chat,
// so no notification slots exist and the event is never signaled.
class IChatImageSession final : public ServiceFramework<IChatImageSession> {
public:
    explicit IChatImageSession(Core::System& system)
        : ServiceFramework{system, "IChatImageSession"}, m_context{system, "ChatImage"} {
        static const FunctionInfo functions[] = {
            {1, &IChatImageSession::GetNotificationLimit, "GetNotificationLimit"},
            {11, &IChatImageSession::GetNotificationEvent, "GetNotificationEvent"},
            {103, &IChatImageSession::OpenVirtualDisplay, "OpenVirtualDisplay"},
        };
        RegisterHandlersTipc(functions);
    }

    Result HandleSyncRequest(Kernel::KServerSession& session, HLERequestContext& ctx) override {
        if (!ctx.IsTipc() || handlers_tipc.contains(ctx.GetCommand())) {
            return ServiceFrameworkBase::HandleSyncRequest(session, ctx);
        }
        const auto guard = LockService();
        LOG_WARNING(Service_AM, "chat image unsupported command={}, words={:08x}", ctx.GetCommand(),
                    fmt::join(std::span<const u32>(ctx.CommandBuffer() + 2, std::min<u32>(ctx.CommandBuffer()[1], 32)), ","));
        InvokeRequestTipc(ctx);
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(Kernel::ResultNotImplemented.raw);
        return system.IsPoweredOn() ? ctx.WriteToOutgoingCommandBuffer() : ResultSuccess;
    }

    ~IChatImageSession() override {
        if (m_event) {
            m_context.CloseEvent(m_event);
        }
    }

private:
    void GetNotificationLimit(HLERequestContext& ctx) {
        const auto* packet = ctx.CommandBuffer();
        const Result result = packet[0] == 0x11 && packet[1] == 0
                                  ? ResultSuccess : Kernel::ResultInvalidCombination;
        // Inclusive upper notification index; the SDK rejects values above 0x3ff.
        IPC::ResponseBuilder rb{ctx, 3};
        rb.Push(result.raw);
        rb.Push<u32>(0);
    }

    void GetNotificationEvent(HLERequestContext& ctx) {
        const auto* packet = ctx.CommandBuffer();
        Result result = Kernel::ResultInvalidCombination;
        Kernel::KReadableEvent* readable{};
        if (packet[0] == 0x1b && packet[1] == 1 && packet[2] == 0) {
            if (!m_event) {
                m_event = m_context.CreateEvent("ChatImageNotification");
            }
            result = m_event ? ResultSuccess : Kernel::ResultOutOfResource;
            if (m_event) {
                readable = &m_event->GetReadableEvent();
                readable->Open(kernel);
            }
        }
        IPC::ResponseBuilder rb{ctx, 2, 0, 1};
        rb.Push(result.raw);
        ctx.AddMoveObject(readable);
    }

    // nn::chat::gsv::OpenVirtualDisplay: a display name of up to 8 bytes ("chat2") and the
    // notification slot that reports its images. The display never receives an image.
    void OpenVirtualDisplay(HLERequestContext& ctx) {
        const auto* packet = ctx.CommandBuffer();
        Result result = Kernel::ResultInvalidCombination;
        if (packet[0] == 0x77 && packet[1] == 3) {
            std::array<char, 8> name{};
            std::memcpy(name.data(), packet + 2, name.size());
            LOG_DEBUG(Service_AM, "chat virtual display {} notification={}",
                      std::string_view(name.data(), strnlen(name.data(), name.size())), packet[4]);
            result = ResultSuccess;
        }
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(result.raw);
    }

    KernelHelpers::ServiceContext m_context;
    Kernel::KEvent* m_event{};
};

// TIPC session behind nn::gs::sdk::QueryApplicationGpuTimeSpan. The SDK first binds a u64
// (command 0, which must be echoed back) and then asks for the GPU time used between two
// timestamps (command 11). The emulator does not account GPU time per application.
class IGpuTimeSession final : public ServiceFramework<IGpuTimeSession> {
public:
    explicit IGpuTimeSession(Core::System& system) : ServiceFramework{system, "IGpuTimeSession"} {
        static const FunctionInfo functions[] = {
            {0, &IGpuTimeSession::BindProcess, "BindProcess"},
            {11, &IGpuTimeSession::QueryApplicationGpuTimeSpan, "QueryApplicationGpuTimeSpan"},
        };
        RegisterHandlersTipc(functions);
    }

private:
    void BindProcess(HLERequestContext& ctx) {
        const auto* packet = ctx.CommandBuffer();
        const Result result = packet[0] == 0x10 && packet[1] == 2 ? ResultSuccess
                                                                  : Kernel::ResultInvalidCombination;
        // Read before the response overwrites the command buffer.
        const u64 process_id = u64{packet[2]} | (u64{packet[3]} << 32);
        IPC::ResponseBuilder rb{ctx, 4};
        rb.Push(result.raw);
        rb.Push<u64>(process_id);
    }

    void QueryApplicationGpuTimeSpan(HLERequestContext& ctx) {
        const auto* packet = ctx.CommandBuffer();
        const Result result = packet[0] == 0x1b && packet[1] == 4 ? ResultSuccess
                                                                  : Kernel::ResultInvalidCombination;
        IPC::ResponseBuilder rb{ctx, 4};
        rb.Push(result.raw);
        rb.Push<u64>(0);
    }
};

class ISyncpointSupervisor final : public ServiceFramework<ISyncpointSupervisor> {
public:
    ISyncpointSupervisor(Core::System& system, std::shared_ptr<Nvidia::Module> module)
        : ServiceFramework{system, "ISyncpointSupervisor"}, m_module{std::move(module)},
          m_context{system, "SyncpointSupervisor"} {
        static const FunctionInfo functions[] = {
            {1, &ISyncpointSupervisor::GetMaximumEventCount, "GetMaximumEventCount"},
            {21, &ISyncpointSupervisor::SetEventCount, "SetEventCount"},
            {22, &ISyncpointSupervisor::GetEvent, "GetEvent"},
            {201, &ISyncpointSupervisor::RequestEvent, "RequestEvent"},
            {202, &ISyncpointSupervisor::CancelEvent, "CancelEvent"},
            {103, &ISyncpointSupervisor::QueryCounter, "QueryCounter"},
        };
        RegisterHandlersTipc(functions);
    }

    ~ISyncpointSupervisor() override {
        for (u32 index = 0; index < MaximumEventCount; ++index) {
            CancelRequest(index);
        }
        for (auto* event : m_events) {
            if (event) {
                m_context.CloseEvent(event);
            }
        }
    }

    Result HandleSyncRequest(Kernel::KServerSession& session, HLERequestContext& ctx) override {
        if (!ctx.IsTipc() || handlers_tipc.contains(ctx.GetCommand())) {
            return ServiceFrameworkBase::HandleSyncRequest(session, ctx);
        }
        const auto guard = LockService();
        LOG_DEBUG(Service_AM, "syncpoint unsupported command={}, words={:08x}", ctx.GetCommand(),
                  fmt::join(std::span<const u32>(ctx.CommandBuffer() + 2, std::min<u32>(ctx.CommandBuffer()[1], 32)), ","));
        InvokeRequestTipc(ctx);
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(Kernel::ResultNotImplemented.raw);
        return system.IsPoweredOn() ? ctx.WriteToOutgoingCommandBuffer() : ResultSuccess;
    }

private:
    static constexpr u32 MaximumEventCount = 63;

    void GetMaximumEventCount(HLERequestContext& ctx) {
        const auto* packet = ctx.CommandBuffer();
        const Result result = packet[0] == 0x11 && packet[1] == 0
            ? ResultSuccess : Kernel::ResultInvalidCombination;
        IPC::ResponseBuilder rb{ctx, 3};
        rb.Push(result.raw);
        rb.Push(MaximumEventCount);
        LOG_DEBUG(Service_AM, "syncpoint maximum events={}", MaximumEventCount);
    }

    void SetEventCount(HLERequestContext& ctx) {
        const auto* packet = ctx.CommandBuffer();
        Result result = Kernel::ResultInvalidCombination;
        if (packet[0] == 0x25 && packet[1] == 1 && packet[2] <= MaximumEventCount) {
            m_event_count = packet[2];
            result = ResultSuccess;
        }
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(result.raw);
        LOG_DEBUG(Service_AM, "syncpoint event count={}, result={:#x}", m_event_count, result.raw);
    }

    // nn::syncpt event slots below the count set by command 21: a u32 index, and the event as a
    // move handle before the result
    void GetEvent(HLERequestContext& ctx) {
        const auto* packet = ctx.CommandBuffer();
        Result result = Kernel::ResultInvalidCombination;
        Kernel::KReadableEvent* readable{};
        if (packet[0] == 0x26 && packet[1] == 1) {
            const u32 index = packet[2];
            result = index < m_event_count ? ResultSuccess : Kernel::ResultOutOfRange;
            if (result.IsSuccess()) {
                auto*& event = m_events[index];
                if (!event) {
                    event = m_context.CreateEvent("SyncpointEvent");
                }
                readable = &event->GetReadableEvent();
                readable->Open(kernel);
            }
            LOG_DEBUG(Service_AM, "syncpoint event index={}, result={:#x}", index, result.raw);
        }
        IPC::ResponseBuilder rb{ctx, 2, 0, 1};
        rb.Push(result.raw);
        ctx.AddMoveObject(readable);
    }

    /// Drops the pending syncpoint action of an event slot
    void CancelRequest(u32 index) {
        if (auto& request = m_requests[index]) {
            system.Host1x().GetSyncpointManager().DeregisterGuestAction(request->first,
                                                                         request->second);
            request.reset();
        }
    }

    // nn::syncpt::sdk::RequestSyncPointEvent: the event slot, the counter of a NativeSyncHandle
    // (the inverted syncpoint id), its u64 value and a mode word. The slot's event is signaled
    // when the syncpoint reaches the value.
    void RequestEvent(HLERequestContext& ctx) {
        const auto* packet = ctx.CommandBuffer();
        Result result = Kernel::ResultInvalidCombination;
        if (packet[0] == 0xd9 && packet[1] == 5) {
            const u32 index = packet[2];
            const u32 id = ~packet[3];
            const u32 value = packet[4];
            auto& syncpoints = m_module->GetContainer().GetSyncpointManager();
            result = index < m_event_count && m_events[index] ? ResultSuccess
                                                               : Kernel::ResultOutOfRange;
            if (result.IsSuccess() && !syncpoints.IsSyncpointAllocated(id)) {
                result = Kernel::ResultNotFound;
            }
            if (result.IsSuccess()) {
                CancelRequest(index);
                auto* event = m_events[index];
                event->Clear(kernel);
                auto& host_syncpoints = system.Host1x().GetSyncpointManager();
                if (host_syncpoints.IsReadyGuest(id, value)) {
                    event->Signal(kernel);
                } else {
                    auto handle = host_syncpoints.RegisterGuestAction(
                        id, value, [this, event] { event->Signal(kernel); });
                    m_requests[index].emplace(id, handle);
                }
            }
            LOG_DEBUG(Service_AM, "syncpoint event index={} id={} value={}, result={:#x}", index,
                      id, value, result.raw);
        }
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(result.raw);
    }

    void CancelEvent(HLERequestContext& ctx) {
        const auto* packet = ctx.CommandBuffer();
        Result result = Kernel::ResultInvalidCombination;
        if (packet[0] == 0xda && packet[1] == 1) {
            result = packet[2] < MaximumEventCount ? ResultSuccess : Kernel::ResultOutOfRange;
            if (result.IsSuccess()) {
                CancelRequest(packet[2]);
            }
        }
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(result.raw);
    }

    void QueryCounter(HLERequestContext& ctx) {
        const auto* packet = ctx.CommandBuffer();
        Result result = Kernel::ResultInvalidCombination;
        u64 value{};
        if (packet[0] == 0x77 && packet[1] == 1) {
            const u32 id = ~packet[2];
            auto current = m_module->GetContainer().GetSyncpointManager().ReadAllocatedSyncpoint(id);
            result = current ? ResultSuccess : Kernel::ResultNotFound;
            value = current.value_or(0);
            LOG_DEBUG(Service_AM, "query syncpoint id={}, value={}, result={:#x}", id, value, result.raw);
        }
        IPC::ResponseBuilder rb{ctx, 5};
        rb.Push(result.raw);
        rb.Push(value);
        rb.Push<u32>(0); // Existing NVDRV counters use 32-bit modular comparisons.
    }

    std::shared_ptr<Nvidia::Module> m_module;
    KernelHelpers::ServiceContext m_context;
    u32 m_event_count{};
    std::array<Kernel::KEvent*, MaximumEventCount> m_events{};
    std::array<std::optional<std::pair<u32, Tegra::Host1x::SyncpointManager::ActionHandle>>,
               MaximumEventCount>
        m_requests{};
};

class IVi2Session final : public ServiceFramework<IVi2Session> {
    struct ClientState {
        u64 token{};
        u32 slots{};
        std::optional<u64> display;
        std::optional<u64> layer;
        u64 available{0x7fff};
        u32 retire_notification{};
        u64 output_encoding{0x14};
        u32 available_notification{};
        std::shared_ptr<android::BufferQueueProducer> producer;
        std::array<std::optional<s32>, 15> queued{};
        std::optional<size_t> player_display;
    };

public:
    explicit IVi2Session(Core::System& system, u64 aruid, std::shared_ptr<NativeImageRegistry> images) : ServiceFramework{system, "IVi2Session"}, m_images{std::move(images)}, m_aruid{aruid}, m_context{system, "Vi2Session"} {
        static const FunctionInfo functions[] = {
            {1, &IVi2Session::GetSessionLimit, "GetNotificationLimit"},
            {11, &IVi2Session::GetNotificationEvent, "GetNotificationEvent"},
            {13, &IVi2Session::ReleaseNotificationEvent, "ReleaseNotificationEvent"},
            {21, &IVi2Session::GetNotifications, "GetNotifications"},
            {31, &IVi2Session::GetRequestGroupEvent, "GetRequestGroupEvent"},
            {263, &IVi2Session::BindDisplayEvent, "BindDisplayEvent"},
            {264, &IVi2Session::UnbindDisplayEvent, "UnbindDisplayEvent"},
            {265, &IVi2Session::BindDisplayEvent, "BindDisplayEvent"},
            {266, &IVi2Session::UnbindDisplayEvent, "UnbindDisplayEvent"},
            {267, &IVi2Session::BindDisplayEvent, "BindDisplayEvent"},
            {268, &IVi2Session::UnbindDisplayEvent, "UnbindDisplayEvent"},
            {269, &IVi2Session::BindDisplayEvent, "BindDisplayEvent"},
            {270, &IVi2Session::UnbindDisplayEvent, "UnbindDisplayEvent"},
            {271, &IVi2Session::BindDisplayEvent, "BindDisplayEvent"},
            {272, &IVi2Session::UnbindDisplayEvent, "UnbindDisplayEvent"},
            {3, &IVi2Session::GetSessionLimit, "GetSessionLimit"},
            {5, &IVi2Session::GetSessionLimit, "GetRequestGroupLimit"},
            {51, &IVi2Session::ConfigureRequestGroups, "ConfigureRequestGroups"},
            {101, &IVi2Session::OpenDisplay, "OpenDisplay"},
            {102, &IVi2Session::CloseDisplay, "CloseDisplay"},
            {103, &IVi2Session::QueryDisplayRefreshRate, "QueryDisplayRefreshRate"},
            {111, &IVi2Session::OpenEndpoint, "OpenEndpoint"},
            {112, &IVi2Session::CloseEndpoint, "CloseEndpoint"},
            {201, &IVi2Session::RegisterClientState, "RegisterClientState"},
            {202, &IVi2Session::UnregisterClientState, "UnregisterClientState"},
            {212, &IVi2Session::GetClientAvailability, "GetClientAvailability"},
            {229, &IVi2Session::PresentClientBuffer, "PresentClientBuffer"},
            {228, &IVi2Session::CancelClientBuffer, "CancelClientBuffer"},
            {351, &IVi2Session::SetClientOutputEncoding, "SetClientOutputEncoding"},
            {251, &IVi2Session::BindClientDisplay, "BindClientDisplay"},
            {252, &IVi2Session::UnbindClientDisplay, "UnbindClientDisplay"},
        };
        RegisterHandlersTipc(functions);
    }

    ~IVi2Session() override {
        if (m_release_poll) system.CoreTiming().UnscheduleEvent(m_release_poll);
        if (m_vsync_poll) system.CoreTiming().UnscheduleEvent(m_vsync_poll);
        if (m_display_service) {
            for (auto& state : m_client_states) {
                if (state.layer) m_display_service->GetContainer()->DestroyManagedLayer(*state.layer);
            }
        }
        if (m_default_display) { m_display_service->CloseDisplay(*m_default_display); }
        m_context.CloseEvent(m_event);
        for (Kernel::KEvent* event : m_request_group_events) {
            if (event) m_context.CloseEvent(event);
        }
    }

    Result HandleSyncRequest(Kernel::KServerSession& session, HLERequestContext& ctx) override {
        if (!ctx.IsTipc() || handlers_tipc.contains(ctx.GetCommand())) {
            return ServiceFrameworkBase::HandleSyncRequest(session, ctx);
        }
        const auto guard = LockService();
        LOG_DEBUG(Service_AM, "VI2 unsupported command={}, words={:08x}", ctx.GetCommand(),
                  fmt::join(std::span<const u32>(ctx.CommandBuffer() + 2, std::min<u32>(ctx.CommandBuffer()[1], 32)), ","));
        InvokeRequestTipc(ctx);
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(Kernel::ResultNotImplemented.raw);
        return system.IsPoweredOn() ? ctx.WriteToOutgoingCommandBuffer() : ResultSuccess;
    }
private:
    static std::optional<size_t> GetPlayerDisplayIndex(u64 name) {
        if ((name & 0xffffffffffffULL) != 0x726579616c70ULL || (name >> 56) != 0) {
            return std::nullopt;
        }
        const u32 player = (name >> 48) & 0xff;
        if (player < '1' || player > '3') return std::nullopt;
        return player - '1';
    }
    Result OpenDefaultDisplay(u64 name, u32 flags) {
        if (const auto player = GetPlayerDisplayIndex(name)) {
            R_UNLESS(flags < m_notifications.size() * 32, Kernel::ResultInvalidCombination);
            R_UNLESS(m_player_display_references[*player] < 1024, Kernel::ResultOutOfResource);
            // GameShare's detached player displays retain their own lifetime;
            // they do not target the host's default display compositor.
            m_player_display_notifications[*player] = flags;
            ++m_player_display_references[*player];
            R_SUCCEED();
        }
        // The native display token is an eight-byte name; command 101 returns
        // only a result. Retain the real VI display ID for later operations.
        R_UNLESS(name == 0x00746c7561666564ULL, Kernel::ResultNotFound);
        R_UNLESS(flags == 1, Kernel::ResultNotImplemented);
        R_UNLESS(m_display_references < 1024, Kernel::ResultOutOfResource);
        if (!m_display_service) {
            const auto root = system.ServiceManager().GetService<VI::IManagerRootService>("vi:m", true);
            R_UNLESS(root, Kernel::ResultInvalidState);
            R_TRY(root->GetDisplayService(&m_display_service, VI::Policy::Compositor));
        }
        if (!m_default_display) {
            u64 id{};
            R_TRY(m_display_service->OpenDefaultDisplay(&id));
            m_default_display = id;
        }
        m_display_notification = flags;
        ++m_display_references;
        R_SUCCEED();
    }
    void OpenDisplay(HLERequestContext& ctx) {
        const auto* packet = ctx.CommandBuffer();
        const u64 name = u64{packet[2]} | (u64{packet[3]} << 32);
        const auto result = packet[0] == 0x75 && packet[1] == 3
            ? OpenDefaultDisplay(name, packet[4]) : Kernel::ResultInvalidCombination;
        LOG_DEBUG(Service_AM, "VI2 open display name={:#x}, flags={:#x}, references={}, result={:#x}",
                  name, packet[4], m_display_references, result.raw);
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(result.raw);
    }
    // nn::vi::QueryDisplayRefreshRate. The SDK maps 1 to 60 Hz, 2 to 120 Hz and
    // 0x8000000000000000 to an unknown rate; the emulated display runs at 60 Hz.
    void QueryDisplayRefreshRate(HLERequestContext& ctx) {
        const auto* packet = ctx.CommandBuffer();
        const u64 name = u64{packet[2]} | (u64{packet[3]} << 32);
        Result result = Kernel::ResultInvalidCombination;
        if (packet[0] == 0x77 && packet[1] == 2) {
            result = name == 0x00746c7561666564ULL || GetPlayerDisplayIndex(name)
                         ? ResultSuccess : Kernel::ResultNotFound;
        }
        IPC::ResponseBuilder rb{ctx, 4};
        rb.Push(result.raw);
        rb.Push<u64>(result.IsSuccess() ? 1 : 0);
    }
    void CloseDisplay(HLERequestContext& ctx) {
        const auto* packet = ctx.CommandBuffer();
        Result result = Kernel::ResultInvalidCombination;
        const u64 name = u64{packet[2]} | (u64{packet[3]} << 32);
        if (packet[0] == 0x76 && packet[1] == 2 && GetPlayerDisplayIndex(name)) {
            const auto player = *GetPlayerDisplayIndex(name);
            result = Kernel::ResultNotFound;
            if (m_player_display_references[player]) {
                if (--m_player_display_references[player] == 0) {
                    const auto notification = m_player_display_notifications[player];
                    m_notifications[notification / 32] |= 1U << (notification % 32);
                    if (m_event) m_event->Signal(kernel);
                }
                result = ResultSuccess;
            }
        } else if (packet[0] == 0x76 && packet[1] == 2 && name == 0x00746c7561666564ULL) {
            result = Kernel::ResultNotFound;
            if (m_default_display && m_display_references) {
                result = ResultSuccess;
                if (m_display_references == 1) {
                    result = m_display_service->CloseDisplay(*m_default_display);
                    if (result.IsSuccess()) { m_default_display.reset(); }
                }
                if (result.IsSuccess()) {
                    --m_display_references;
                    if (!m_display_references) {
                        m_notifications[m_display_notification / 32] |= 1U << (m_display_notification % 32);
                        if (m_event) m_event->Signal(kernel);
                        LOG_DEBUG(Service_AM, "VI2 display retired notification={}", m_display_notification);
                    }
                }
            }
        }
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(result.raw);
    }
    void RegisterClientState(HLERequestContext& ctx) {
        const auto* packet = ctx.CommandBuffer();
        const u64 token = u64{packet[2]} | (u64{packet[3]} << 32);
        const u32 first = packet[4];
        const u32 second = packet[5];
        Result result = Kernel::ResultInvalidCombination;
        if (packet[0] == 0xd9 && packet[1] == 4 && token != 0 &&
            first < 32 && second < 32 && first != second) {
            const u32 slots = (1U << first) | (1U << second);
            const auto existing = std::ranges::find(m_client_states, token, &ClientState::token);
            const auto free = std::ranges::find(m_client_states, u64{0}, &ClientState::token);
            result = Kernel::ResultInvalidState;
            if (existing == m_client_states.end() && !(m_client_slots & slots)) {
                result = Kernel::ResultOutOfResource;
                if (free != m_client_states.end()) {
                    // The SDK sends its object address as an opaque identity, not a
                    // service-accessible buffer. Keep the associated slots session-local.
                    *free = {token, slots};
                    free->retire_notification = first;
                    free->available_notification = second;
                    m_client_slots |= slots;
                    result = ResultSuccess;
                }
            }
        }
        LOG_DEBUG(Service_AM, "VI2 register client state token={:#x}, slots={}/{}, result={:#x}",
                  token, first, second, result.raw);
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(result.raw);
    }
    void UnregisterClientState(HLERequestContext& ctx) {
        const auto* packet = ctx.CommandBuffer();
        const u64 token = u64{packet[2]} | (u64{packet[3]} << 32);
        Result result = Kernel::ResultInvalidCombination;
        if (packet[0] == 0xda && packet[1] == 2 && token != 0) {
            const auto state = std::ranges::find(m_client_states, token, &ClientState::token);
            result = Kernel::ResultNotFound;
            if (state != m_client_states.end()) {
                if (m_default_endpoint == token) { m_default_endpoint.reset(); }
                if (state->layer) m_display_service->GetContainer()->DestroyManagedLayer(*state->layer);
                m_notifications[state->retire_notification / 32] |= 1U << (state->retire_notification % 32);
                if (m_event) m_event->Signal(kernel);
                LOG_DEBUG(Service_AM, "VI2 client retired token={:#x}, notification={}", token, state->retire_notification);
                m_client_slots &= ~state->slots;
                *state = {};
                result = ResultSuccess;
            }
        }
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(result.raw);
    }
    void GetClientAvailability(HLERequestContext& ctx) {
        RefreshReleasedBuffers();
        const auto* packet = ctx.CommandBuffer();
        const u64 token = u64{packet[2]} | (u64{packet[3]} << 32);
        Result result = Kernel::ResultInvalidCombination;
        u64 id{};
        if (packet[0] == 0xe4 && packet[1] == 2 && token) {
            auto state = std::ranges::find(m_client_states, token, &ClientState::token);
            result = Kernel::ResultNotFound;
            if (state != m_client_states.end() && m_default_display) {
                result = ResultSuccess;
                if (state->player_display) {
                    id = 0;
                } else if (!state->layer) {
                    result = m_display_service->GetContainer()->CreateManagedLayer(&id, *m_default_display, m_aruid);
                    if (result.IsSuccess()) state->layer = id;
                } else {
                    id = *state->layer;
                }
            }
        }
        const auto state = std::ranges::find(m_client_states, token, &ClientState::token);
        const u64 available = result.IsSuccess() ? state->available : 0;
        LOG_DEBUG(Service_AM, "VI2 client availability token={:#x}, layer={}, mask={:#x}, result={:#x}", token, id, available, result.raw);
        IPC::ResponseBuilder rb{ctx, 4};
        rb.Push(result.raw);
        rb.Push(available);
    }
    void CancelClientBuffer(HLERequestContext& ctx) {
        const auto* packet = ctx.CommandBuffer();
        const u64 token = u64{packet[2]} | (u64{packet[3]} << 32);
        const u32 slot = GraphicsSupervisorProtocol::DecodeBufferSlot(packet[4]);
        Result result = Kernel::ResultInvalidCombination;
        u64 available{};
        if (packet[0] == 0xf4 && packet[1] == 19 && slot < 15) {
            auto state = std::ranges::find(m_client_states, token, &ClientState::token);
            result = Kernel::ResultNotFound;
            if (token && state != m_client_states.end()) {
                // Unlike present (229), cancel has no image or input GPU fence.
                state->available |= u64{1} << slot;
                available = state->available;
                result = ResultSuccess;
            }
        }
        LOG_DEBUG(Service_AM, "VI2 cancel buffer token={:#x}, slot={}, mask={:#x}, result={:#x}", token, slot, available, result.raw);
        IPC::ResponseBuilder rb{ctx, 8};
        rb.Push(result.raw);
        rb.Push(available);
        for (u32 i = 0; i < 4; ++i) rb.Push<u32>(0);
    }
    void RefreshReleasedBuffers() {
        for (auto& state : m_client_states) {
            if (!state.producer || !m_images->module) continue;
            bool changed{};
            for (size_t slot = 0; slot < state.queued.size(); ++slot) {
                if (!state.queued[slot]) continue;
                const auto fence = state.producer->GetReleasedBufferFence(*state.queued[slot]);
                if (!fence) continue;
                bool complete = true;
                for (u32 i = 0; i < fence->num_fences; ++i) {
                    const auto& point = fence->fences[i];
                    if (point.id != ~u32{0}) {
                        const auto counter = m_images->module->GetContainer().GetSyncpointManager().ReadAllocatedSyncpoint(point.id);
                        if (!counter || static_cast<s32>(*counter - point.value) < 0) {
                            complete = false;
                        }
                    }
                }
                if (!complete) continue;
                state.queued[slot].reset();
                state.available |= u64{1} << slot;
                LOG_DEBUG(Service_AM, "Native display buffer retired: slot={}, mask={:#x}", slot, state.available);
                changed = true;
            }
            if (changed) {
                m_notifications[state.available_notification / 32] |= 1U << (state.available_notification % 32);
                if (m_event) m_event->Signal(kernel);
            }
        }
    }
    Result QueueNativeImage(ClientState& state, const u32* packet) {
        const u32 slot = GraphicsSupervisorProtocol::DecodeBufferSlot(packet[4]);
        if (state.player_display) {
            LOG_WARNING(Service_AM, "presentation to a player display is not implemented");
            return Kernel::ResultNotImplemented;
        }
        R_UNLESS(state.layer && state.display && slot < 15 && !state.queued[slot], Kernel::ResultInvalidState);
        const std::scoped_lock images_lock{m_images->mutex};
        const auto image = std::ranges::find(m_images->images, packet[5], &NativeImage::id);
        R_UNLESS(image != m_images->images.end() && image->window_pool && m_images->module, Kernel::ResultNotFound);
        // The observed presentation is a four-byte color image with the
        // backend's 16-GOB block height and no byte offset within its pool.
        // packet[7] packs the inclusive maximum x and y of the presented region,
        // which starts at the origin and may be smaller than the image.
        const u32 present_width = (packet[7] & 0xffffU) + 1;
        const u32 present_height = (packet[7] >> 16) + 1;
        const u32 image_width = (image->preference[2] & 0xffffU) + 1;
        const u32 image_height = (image->preference[2] >> 16) + 1;
        const u64 bpp = NativeImageBytesPerPixel(image->preference[6]);
        if (!(packet[6] == 0 && present_width <= image_width && present_height <= image_height &&
              packet[8] == 3 && image->preference[4] == 7 && bpp != 0 &&
              (state.output_encoding == 0x14 || state.output_encoding == 0x24 ||
               state.output_encoding == 0x38))) {
            LOG_WARNING(Service_AM, "unsupported presentation words={:08x} preference={:08x} encoding={:#x}",
                        fmt::join(std::span<const u32>(packet + 6, 16), ","),
                        fmt::join(image->preference, ","), state.output_encoding);
            return Kernel::ResultNotImplemented;
        }
        const u32 fence_id = ~packet[22];
        R_UNLESS(packet[23] == 0 && packet[25] == 0 &&
                 (fence_id == ~u32{0} || m_images->module->GetContainer().GetSyncpointManager().IsSyncpointAllocated(fence_id)),
                 Kernel::ResultInvalidCombination);
        if (!state.producer) {
            R_TRY(m_display_service->GetContainer()->GetLayerProducerHandle(&state.producer, *state.layer));
            android::QueueBufferOutput output{};
            R_UNLESS(state.producer->Connect({}, android::NativeWindowApi::Egl, false, &output) == android::Status::NoError,
                     Kernel::ResultInvalidState);
            R_UNLESS(state.producer->SetBufferCount(16) == android::Status::NoError, Kernel::ResultInvalidState);
            s32 binder{};
            R_TRY(m_display_service->GetContainer()->OpenLayer(&binder, *state.layer, m_aruid));
            R_TRY(m_display_service->GetContainer()->SetLayerVisibility(*state.layer, true));
        }
        auto native = std::make_shared<android::NvGraphicBuffer>();
        native->width = static_cast<s32>((image->preference[2] & 0xffffU) + 1);
        native->height = static_cast<s32>((image->preference[2] >> 16) + 1);
        native->stride = static_cast<s32>(image->preference[5] * (64 / bpp));
        switch (image->preference[6]) {
        case 3:
            native->format = android::PixelFormat::Bgra8888;
            break;
        case 4:
            native->format = android::PixelFormat::Rgb10a2;
            break;
        case 5:
        case 6:
            native->format = android::PixelFormat::RgbaFp16;
            break;
        default:
            native->format = android::PixelFormat::Rgba8888;
            break;
        }
        native->external_format = native->format;
        native->buffer_id = image->memory_id;
        auto buffer = std::make_shared<android::GraphicBuffer>(m_images->module->GetContainer().GetNvMapFile(), native);
        s32 queue_slot{};
        R_UNLESS(state.producer->AttachBuffer(&queue_slot, buffer) == android::Status::NoError, Kernel::ResultOutOfResource);
        android::QueueBufferInput input{};
        android::QueueBufferOutput output{};
        input.crop = {0, 0, static_cast<s32>(present_width), static_cast<s32>(present_height)};
        input.swap_interval = 1;
        input.fence = android::Fence::NoFence();
        if (fence_id != ~u32{0}) {
            input.fence.num_fences = 1;
            input.fence.fences[0] = {static_cast<s32>(fence_id), packet[24]};
        }
        if (state.producer->QueueBuffer(queue_slot, input, &output) != android::Status::NoError) {
            state.producer->CancelBuffer(queue_slot, input.fence);
            return Kernel::ResultInvalidState;
        }
        state.queued[slot] = queue_slot;
        state.available &= ~(u64{1} << slot);
        if (!m_release_poll) {
            m_release_poll = Core::Timing::CreateEvent("NativeViBufferRelease", [this](s64, std::chrono::nanoseconds) -> std::optional<std::chrono::nanoseconds> {
                const auto guard = LockService();
                RefreshReleasedBuffers();
                return std::nullopt;
            });
            system.CoreTiming().ScheduleLoopingEvent(std::chrono::milliseconds{1}, std::chrono::milliseconds{1}, m_release_poll);
        }
        LOG_INFO(Service_AM, "Native image queued for display: image={}, slot={}, fence={}/{}, dimensions={}x{}, crop={}x{}",
                 image->id, slot, fence_id, packet[24], native->width, native->height, present_width, present_height);
        R_SUCCEED();
    }
    void PresentClientBuffer(HLERequestContext& ctx) {
        const auto* packet = ctx.CommandBuffer();
        const u64 token = u64{packet[2]} | (u64{packet[3]} << 32);
        RefreshReleasedBuffers();
        Result result = Kernel::ResultInvalidCombination;
        u64 available{};
        if (packet[0] == 0xf5 && packet[1] == 24) {
            const auto state = std::ranges::find(m_client_states, token, &ClientState::token);
            result = Kernel::ResultNotFound;
            if (token && state != m_client_states.end()) {
                result = QueueNativeImage(*state, packet);
                available = state->available;
            }
        }
        LOG_DEBUG(Service_AM, "VI2 present token={:#x}, image={}, slot={}, mask={:#x}, result={:#x}", token, packet[5], packet[4], available, result.raw);
        IPC::ResponseBuilder rb{ctx, 8};
        rb.Push(result.raw);
        rb.Push(available);
        for (u32 i = 0; i < 4; ++i) rb.Push<u32>(0);
    }
    void SetClientOutputEncoding(HLERequestContext& ctx) {
        const auto* packet = ctx.CommandBuffer();
        const u64 token = u64{packet[2]} | (u64{packet[3]} << 32);
        const u64 encoding = u64{packet[4]} | (u64{packet[5]} << 32);
        Result result = Kernel::ResultInvalidCombination;
        if (packet[0] == 0x16f && packet[1] == 4) {
            const auto state = std::ranges::find(m_client_states, token, &ClientState::token);
            result = Kernel::ResultNotFound;
            if (token && state != m_client_states.end()) {
                result = Kernel::ResultNotImplemented;
                // 0x14 = sRGB, 0x24 = PQ (HDR10), 0x38 = scRGB linear.
                if (encoding == 0x14 || encoding == 0x24 || encoding == 0x38) {
                    state->output_encoding = encoding;
                    result = ResultSuccess;
                }
            }
        }
        LOG_DEBUG(Service_AM, "VI2 output encoding token={:#x}, encoding={:#x}, result={:#x}", token, encoding, result.raw);
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(result.raw);
    }
    void BindClientDisplay(HLERequestContext& ctx) {
        const auto* packet = ctx.CommandBuffer();
        const u64 token = u64{packet[2]} | (u64{packet[3]} << 32);
        const u64 name = u64{packet[4]} | (u64{packet[5]} << 32);
        Result result = Kernel::ResultInvalidCombination;
        if (packet[0] == 0x10b && packet[1] == 4) {
            const auto state = std::ranges::find(m_client_states, token, &ClientState::token);
            result = Kernel::ResultNotFound;
            if (token && state != m_client_states.end() && GetPlayerDisplayIndex(name)) {
                const auto player = *GetPlayerDisplayIndex(name);
                result = Kernel::ResultInvalidState;
                if (m_player_display_references[player] && !state->display) {
                    result = ResultSuccess;
                    if (state->layer) {
                        result = m_display_service->GetContainer()->DestroyManagedLayer(*state->layer);
                    }
                    if (result.IsSuccess()) {
                        state->layer.reset();
                        state->display = name;
                        state->player_display = player;
                    }
                }
            } else if (token && state != m_client_states.end() && name == 0x00746c7561666564ULL &&
                m_default_display && m_display_references) {
                result = Kernel::ResultInvalidState;
                if (!state->display) {
                    state->display = m_default_display;
                    result = ResultSuccess;
                }
            }
        }
        LOG_DEBUG(Service_AM, "VI2 bind client token={:#x}, display={:#x}, result={:#x}",
                  token, name, result.raw);
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(result.raw);
    }
    void UnbindClientDisplay(HLERequestContext& ctx) {
        const auto* packet = ctx.CommandBuffer();
        const u64 token = u64{packet[2]} | (u64{packet[3]} << 32);
        Result result = Kernel::ResultInvalidCombination;
        if (packet[0] == 0x10c && packet[1] == 2) {
            const auto state = std::ranges::find(m_client_states, token, &ClientState::token);
            result = Kernel::ResultNotFound;
            if (token && state != m_client_states.end() && state->display) {
                state->display.reset();
                state->player_display.reset();
                result = ResultSuccess;
            }
        }
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(result.raw);
    }
    void OpenEndpoint(HLERequestContext& ctx) {
        const auto* packet = ctx.CommandBuffer();
        const u64 name = u64{packet[2]} | (u64{packet[3]} << 32);
        const u64 display_name = u64{packet[4]} | (u64{packet[5]} << 32);
        const u64 token = u64{packet[6]} | (u64{packet[7]} << 32);
        Result result = Kernel::ResultInvalidCombination;
        if (packet[0] == 0x7f && packet[1] == 6) {
            const auto state = std::ranges::find(m_client_states, token, &ClientState::token);
            result = Kernel::ResultNotFound;
            if (token && state != m_client_states.end() && state->display &&
                m_default_display == state->display && name == 0x00746c7561666564ULL &&
                display_name == name) {
                result = Kernel::ResultInvalidState;
                if (!m_default_endpoint) {
                    m_default_endpoint = token;
                    result = ResultSuccess;
                }
            }
        }
        LOG_DEBUG(Service_AM, "VI2 open endpoint name={:#x}, display={:#x}, client={:#x}, result={:#x}",
                  name, display_name, token, result.raw);
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(result.raw);
    }
    void CloseEndpoint(HLERequestContext& ctx) {
        const auto* packet = ctx.CommandBuffer();
        const u64 name = u64{packet[2]} | (u64{packet[3]} << 32);
        const u64 display_name = u64{packet[4]} | (u64{packet[5]} << 32);
        Result result = Kernel::ResultInvalidCombination;
        if (packet[0] == 0x80 && packet[1] == 4 && name == 0x00746c7561666564ULL &&
            display_name == name) {
            result = Kernel::ResultNotFound;
            if (m_default_endpoint) {
                m_default_endpoint.reset();
                result = ResultSuccess;
            }
        }
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(result.raw);
    }
    void ConfigureRequestGroups(HLERequestContext& ctx) {
        const auto* packet = ctx.CommandBuffer();
        Result result = Kernel::ResultInvalidCombination;
        if (packet[0] == 0x43 && packet[1] == 1 && packet[2] < 8) {
            m_request_group_limit = packet[2];
            m_request_groups_initialized = true;
            result = ResultSuccess;
        }
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(result.raw);
    }
    void GetSessionLimit(HLERequestContext& ctx) {
        const auto* packet = ctx.CommandBuffer();
        const Result result = packet[0] == 16 + ctx.GetCommand() && packet[1] == 0
                                  ? ResultSuccess : Kernel::ResultInvalidCombination;
        // The SDK converts this inclusive upper group index into 32 client slots.
        IPC::ResponseBuilder rb{ctx, 3};
        rb.Push(result.raw);
        rb.Push<u32>(ctx.GetCommand() == 5 ? 7 : 0);
    }

    void GetNotifications(HLERequestContext& ctx) {
        Result result = Kernel::ResultInvalidSize;
        if (ctx.GetWriteBufferSize() >= sizeof(m_notifications)) {
            ctx.WriteBuffer(m_notifications);
            LOG_DEBUG(Service_AM, "VI2 notifications={:08x}", fmt::join(m_notifications, ","));
            m_notifications.fill(0);
            if (m_event) m_event->Clear(kernel);
            result = ResultSuccess;
        }
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(result.raw);
    }
    void ReleaseNotificationEvent(HLERequestContext& ctx) {
        const auto* packet = ctx.CommandBuffer();
        const auto result = packet[0] == 0x1d && packet[1] == 1 && packet[2] == 0
            ? ResultSuccess : Kernel::ResultInvalidCombination;
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(result.raw);
    }
    // Command 31 takes one index and returns a handle; next to the notification event commands
    // it is taken to be the event of a request group. This is inferred from the call made at
    // boot, not documented.
    void GetRequestGroupEvent(HLERequestContext& ctx) {
        const auto* packet = ctx.CommandBuffer();
        Result result = Kernel::ResultInvalidCombination;
        Kernel::KReadableEvent* readable{};
        const u32 index = packet[2];
        if (packet[0] == 0x2f && packet[1] == 1 && index < m_request_group_events.size()) {
            auto& event = m_request_group_events[index];
            if (!event) {
                event = m_context.CreateEvent("Vi2RequestGroup");
            }
            result = event ? ResultSuccess : Kernel::ResultOutOfResource;
            if (event) {
                readable = &event->GetReadableEvent();
                readable->Open(kernel);
            }
        }
        LOG_WARNING(Service_AM, "VI2 request group event index={}, result={:#x}", index, result.raw);
        IPC::ResponseBuilder rb{ctx, 2, 0, 1};
        rb.Push(result.raw);
        ctx.AddMoveObject(readable);
    }
    // Commands 263 to 272 take a display name and the index of a request group event (command
    // 31), and reply with a result only. They come in pairs; the odd command binds the event to
    // one kind of display notification and the even one unbinds it. This is inferred from the
    // requests made at boot and the proxies in the SDK; the kinds are not identified. The first
    // kind (263) is taken to be vsync and signalled at 60 Hz.
    void BindDisplayEvent(HLERequestContext& ctx) {
        const auto* packet = ctx.CommandBuffer();
        const u32 command = ctx.GetCommand();
        const u64 name = u64{packet[2]} | (u64{packet[3]} << 32);
        const u32 index = packet[4];
        const u32 kind = (command - 263) / 2;
        Result result = Kernel::ResultInvalidCombination;
        if (packet[0] == 16 + command && packet[1] == 3) {
            result = index < m_request_group_events.size() && m_request_group_events[index]
                         ? ResultSuccess
                         : Kernel::ResultNotFound;
        }
        if (result.IsSuccess() && kind == 0) {
            m_vsync_events |= 1U << index;
            if (!m_vsync_poll) {
                m_vsync_poll = Core::Timing::CreateEvent(
                    "NativeViVsync",
                    [this](s64, std::chrono::nanoseconds) -> std::optional<std::chrono::nanoseconds> {
                        const auto guard = LockService();
                        for (u32 i = 0; i < m_request_group_events.size(); ++i) {
                            if ((m_vsync_events >> i) & 1) m_request_group_events[i]->Signal(kernel);
                        }
                        return std::nullopt;
                    });
                constexpr auto period = std::chrono::nanoseconds{1'000'000'000 / 60};
                system.CoreTiming().ScheduleLoopingEvent(period, period, m_vsync_poll);
            }
        }
        LOG_WARNING(Service_AM, "VI2 bind display event command={}, kind={}, display={:#x}, index={}, result={:#x}",
                    command, kind, name, index, result.raw);
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(result.raw);
    }
    void UnbindDisplayEvent(HLERequestContext& ctx) {
        const auto* packet = ctx.CommandBuffer();
        const u32 command = ctx.GetCommand();
        const u32 index = packet[4];
        const u32 kind = (command - 264) / 2;
        Result result = Kernel::ResultInvalidCombination;
        if (packet[0] == 16 + command && packet[1] == 3) {
            result = ResultSuccess;
            if (kind == 0 && index < 32) {
                m_vsync_events &= ~(1U << index);
            }
        }
        LOG_WARNING(Service_AM, "VI2 unbind display event command={}, kind={}, index={}, result={:#x}",
                    command, kind, index, result.raw);
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(result.raw);
    }
    void GetNotificationEvent(HLERequestContext& ctx) {
        const auto* packet = ctx.CommandBuffer();
        Result result = Kernel::ResultInvalidCombination;
        Kernel::KReadableEvent* readable{};
        if (packet[0] == 0x1b && packet[1] == 1 && packet[2] == 0) {
            if (!m_event) {
                m_event = m_context.CreateEvent("Vi2Notification");
            }
            result = m_event ? ResultSuccess : Kernel::ResultOutOfResource;
            if (m_event) {
                if (std::ranges::any_of(m_notifications, [](u32 value) { return value != 0; })) m_event->Signal(kernel);
                readable = &m_event->GetReadableEvent();
                readable->Open(kernel);
            }
        }
        IPC::ResponseBuilder rb{ctx, 2, 0, 1};
        rb.Push(result.raw);
        ctx.AddMoveObject(readable);
    }

    std::shared_ptr<NativeImageRegistry> m_images;
    std::shared_ptr<Core::Timing::EventType> m_release_poll;
    std::array<ClientState, 16> m_client_states{};
    u32 m_client_slots{};
    std::shared_ptr<VI::IApplicationDisplayService> m_display_service;
    std::optional<u64> m_default_display;
    u32 m_display_references{};
    u32 m_display_notification{};
    std::array<u32, 3> m_player_display_references{};
    std::array<u32, 3> m_player_display_notifications{};
    bool m_request_groups_initialized{};
    u32 m_request_group_limit{};
    std::optional<u64> m_default_endpoint;
    std::array<u32, 32> m_notifications{};
    const u64 m_aruid;
    KernelHelpers::ServiceContext m_context;
    Kernel::KEvent* m_event{};
    std::array<Kernel::KEvent*, 8> m_request_group_events{};
    /// Request group events bound to display vsync (command 263)
    u32 m_vsync_events{};
    std::shared_ptr<Core::Timing::EventType> m_vsync_poll;
};

} // namespace


IGraphicsSupervisor::IGraphicsSupervisor(Core::System& system, Kernel::KProcess* process, u64 aruid)
    : ServiceFramework{system, "IGraphicsSupervisor"}, m_context{system, "GraphicsSupervisor"},
      m_syncpoint_cache_event{m_context}, m_process{process}, m_aruid{aruid} {
    m_process->Open(kernel);
    m_image_registry = std::make_shared<NativeImageRegistry>(kernel, m_process);
    static const FunctionInfo functions[] = {
        {4, &IGraphicsSupervisor::Initialize, "Initialize"},
        {11, &IGraphicsSupervisor::BindProcess, "BindProcess"},
        {101, &IGraphicsSupervisor::CompleteBootstrap, "CompleteBootstrap"},
        {115, &IGraphicsSupervisor::ConfigureNativeMemory, "ConfigureNativeMemory"},
        {111, &IGraphicsSupervisor::CreateNativeMemory, "CreateNativeMemory"},
        {112, &IGraphicsSupervisor::DestroyNativeMemory, "DestroyNativeMemory"},
        {114, &IGraphicsSupervisor::BindNativeMemory, "BindNativeMemory"},
        {201, &IGraphicsSupervisor::InitializeSyncpointContext, "InitializeSyncpointContext"},
        {202, &IGraphicsSupervisor::GetSyncpointSession, "GetSyncpointSession"},
        {204, &IGraphicsSupervisor::BindSyncpointMemory, "BindSyncpointMemory"},
        {301, &IGraphicsSupervisor::InitializeDisplayContext, "InitializeDisplayContext"},
        {305, &IGraphicsSupervisor::OpenDisplaySession, "OpenDisplaySession"},
        {314, &IGraphicsSupervisor::ConfigureNativeWindowImagePool, "DestroyNativeWindowAliasImagePool"},
        {315, &IGraphicsSupervisor::ConfigureNativeWindowImagePool, "CreateNativeWindowAliasImagePool"},
        {401, &IGraphicsSupervisor::InitializeGpuContext, "InitializeGpuContext"},
        {402, &IGraphicsSupervisor::InitializeGpuContext, "ActivateGpuContext"},
        {423, &IGraphicsSupervisor::BindNativeGpuMemory, "BindNativeGpuMemory"},
        {426, &IGraphicsSupervisor::MapNativeImage, "MapNativeImage"},
        {427, &IGraphicsSupervisor::UnmapNativeImage, "UnmapNativeImage"},
        {428, &IGraphicsSupervisor::QueryNativeImage, "QueryNativeImage"},
        {429, &IGraphicsSupervisor::CreateNativeImage, "CreateNativeImage"},
        {430, &IGraphicsSupervisor::DestroyNativeImage, "DestroyNativeImage"},
        {434, &IGraphicsSupervisor::RegisterGpuAddressSpace, "RegisterGpuAddressSpace"},
        {497, &IGraphicsSupervisor::OpenGpuSession, "OpenGpuSession"},
        {601, &IGraphicsSupervisor::InitializeCameraContext, "InitializeCameraContext"},
        {602, &IGraphicsSupervisor::OpenCameraSession, "OpenCameraSession"},
        {603, &IGraphicsSupervisor::FinalizeCameraContext, "FinalizeCameraContext"},
        {1111, &IGraphicsSupervisor::SetGraphicsFrameMode, "SetGraphicsFrameLengthMode"},
        {1112, &IGraphicsSupervisor::SetGraphicsFrameMode, "SetGraphicsFrameRestartMode"},
        {1113, &IGraphicsSupervisor::SetGraphicsFrameMode, "SetGraphicsFrameGpuTimesliceMode"},
        {1121, &IGraphicsSupervisor::OpenGpuTimeSession, "OpenGpuTimeSession"},
        {901, &IGraphicsSupervisor::InitializeVideoDecoderContext, "InitializeVideoDecoderContext"},
        {991, &IGraphicsSupervisor::CalculateDecoderImagePoolBuffer, "CalculateDecoderImagePoolBuffer"},
        {911, &IGraphicsSupervisor::CreateDecoderImagePool, "CreateDecoderImagePool"},
        {921, &IGraphicsSupervisor::DestroyDecoderImagePool, "DestroyDecoderImagePool"},
        {961, &IGraphicsSupervisor::GetDecoderImageDescriptor, "GetDecoderImageDescriptor"},
        {902, &IGraphicsSupervisor::OpenVideoDecoderSession, "OpenVideoDecoderSession"},
        {903, &IGraphicsSupervisor::FinalizeVideoDecoderContext, "FinalizeVideoDecoderContext"},
        {905, &IGraphicsSupervisor::OpenVideoDecoderSession, "OpenVideoDecoderSessionWithMode"},
        {1001, &IGraphicsSupervisor::InitializeVideoEncoderContext, "InitializeVideoEncoderContext"},
        {1002, &IGraphicsSupervisor::OpenVideoEncoderSession, "OpenVideoEncoderSession"},
        {1003, &IGraphicsSupervisor::FinalizeVideoEncoderContext, "FinalizeVideoEncoderContext"},
        {1201, &IGraphicsSupervisor::InitializeChatImageContext, "InitializeChatImageContext"},
        {1202, &IGraphicsSupervisor::OpenChatImageSession, "OpenChatImageSession"},
        {1203, &IGraphicsSupervisor::FinalizeChatImageContext, "FinalizeChatImageContext"},
        {1101, &IGraphicsSupervisor::InitializeFrameContext, "InitializeFrameContext"},
        {1102, &IGraphicsSupervisor::InitializeFrameContext, "ActivateFrameContext"},
        {1402, &IGraphicsSupervisor::GetNvidiaDriver, "GetNvidiaDriver"},
        {1501, &IGraphicsSupervisor::InitializeImageContext, "InitializeImageContext1501"},
        {435, &IGraphicsSupervisor::CalculateImageRequirements, "CalculateNativeImageSize"},
        {1510, &IGraphicsSupervisor::CalculateImageRequirements, "CalculateImageRequirements"},
        {1502, &IGraphicsSupervisor::InitializeImageContext, "InitializeImageContext1502"},
    };
    RegisterHandlersTipc(functions);
}

IGraphicsSupervisor::~IGraphicsSupervisor() {
    if (m_cache_poll) {
        system.CoreTiming().UnscheduleEvent(m_cache_poll);
    }
    if (m_syncpoint_rw) {
        m_syncpoint_rw->Close(kernel);
        m_syncpoint_ro->Close(kernel);
    }
    if (m_nv_module) {
        auto& nvmap = m_nv_module->GetContainer().GetNvMapFile();
        for (const auto id : m_native_memories) {
            if (const auto info = nvmap.FreeHandle(id, false); info && info->can_unlock && info->address != 0) {
                m_process->GetPageTable().UnlockForDeviceAddressSpace(info->address, info->size);
            }
        }
        if (m_native_memory_session) {
            m_nv_module->GetContainer().CloseSession(Nvidia::NvCore::SessionId{*m_native_memory_session});
        }
    }
    m_context.CloseEvent(m_syncpoint_request_event);
    m_process->Close(kernel);
}

Result IGraphicsSupervisor::HandleSyncRequest(Kernel::KServerSession& session,
                                               HLERequestContext& ctx) {
    if (!ctx.IsTipc() || handlers_tipc.contains(ctx.GetCommand())) {
        return ServiceFrameworkBase::HandleSyncRequest(session, ctx);
    }
    const auto guard = LockService();
    InvokeRequestTipc(ctx);
    // Unsupported TIPC must return an error instead of leaving request data as a reply.
    IPC::ResponseBuilder rb{ctx, 2};
    rb.Push(Kernel::ResultNotImplemented.raw);
    return system.IsPoweredOn() ? ctx.WriteToOutgoingCommandBuffer() : ResultSuccess;
}

void IGraphicsSupervisor::Initialize(HLERequestContext& ctx) {
    const auto request = GraphicsSupervisorProtocol::DecodeBootstrap(
        std::span<const u32>{ctx.CommandBuffer(), IPC::COMMAND_BUFFER_LENGTH});
    Result result = Kernel::ResultInvalidCombination;
    if (request) {
        auto process = ctx.GetObjectFromHandle<Kernel::KProcess>(request->process_handle);
        const bool empty_remote_memory = std::ranges::all_of(
            request->parameters, [](u64 value) { return value == 0; });
        if (process.IsNull() || process.GetPointerUnsafe() != m_process) {
            result = Kernel::ResultInvalidHandle;
        } else if (m_phase != Phase::Created) {
            result = Kernel::ResultInvalidState;
        } else if (request->applet_resource_user_id != m_aruid) {
            result = Kernel::ResultInvalidArgument;
        } else if (request->flags != 0x8000000000000005ULL || !empty_remote_memory) {
            result = Kernel::ResultNotImplemented;
        } else {
            m_phase = Phase::Initialized;
            result = ResultSuccess;
        }
        LOG_DEBUG(Service_AM, "graphics bootstrap aruid={:#x}, flags={:#x}, result={:#x}",
                  request->applet_resource_user_id, request->flags, result.raw);
    }
    IPC::ResponseBuilder rb{ctx, 2};
    rb.Push(result.raw);
}

void IGraphicsSupervisor::BindProcess(HLERequestContext& ctx) {
    Result result = Kernel::ResultInvalidCombination;
    const auto* packet = ctx.CommandBuffer();
    if (packet[0] == 0x1b && packet[1] == 0x80000000 && packet[2] == 2) {
        auto process = ctx.GetObjectFromHandle<Kernel::KProcess>(ctx.GetCopyHandle(0));
        if (process.IsNull() || process.GetPointerUnsafe() != m_process) {
            result = Kernel::ResultInvalidHandle;
        } else if (m_phase != Phase::Initialized) {
            result = Kernel::ResultInvalidState;
        } else {
            m_phase = Phase::ProcessBound;
            result = ResultSuccess;
        }
    }
    LOG_DEBUG(Service_AM, "graphics process binding result={:#x}", result.raw);
    IPC::ResponseBuilder rb{ctx, 2};
    rb.Push(result.raw);
}

void IGraphicsSupervisor::CompleteBootstrap(HLERequestContext& ctx) {
    Result result = Kernel::ResultInvalidCombination;
    const auto* packet = ctx.CommandBuffer();
    if (packet[0] == 0x75 && packet[1] == 0) {
        if (m_phase != Phase::ProcessBound) {
            result = Kernel::ResultInvalidState;
        } else {
            m_phase = Phase::Ready;
            result = ResultSuccess;
        }
    }
    LOG_DEBUG(Service_AM, "graphics bootstrap completion result={:#x}", result.raw);
    IPC::ResponseBuilder rb{ctx, 2};
    rb.Push(result.raw);
}

void IGraphicsSupervisor::InitializeImageContext(HLERequestContext& ctx) {
    Result result = Kernel::ResultInvalidCombination;
    const auto* packet = ctx.CommandBuffer();
    if (packet[0] == 16 + ctx.GetCommand() && packet[1] == 0) {
        if (m_phase != Phase::Ready) {
            result = Kernel::ResultInvalidState;
        } else if (ctx.GetCommand() == 1501) {
            m_image_context_initialized = true;
            result = ResultSuccess;
        } else if (m_image_context_initialized) {
            m_image_formats_initialized = true;
            result = ResultSuccess;
        } else {
            result = Kernel::ResultInvalidState;
        }
    }
    LOG_DEBUG(Service_AM, "empty image context command={}, result={:#x}",
              ctx.GetCommand(), result.raw);
    IPC::ResponseBuilder rb{ctx, 2};
    rb.Push(result.raw);
}

void IGraphicsSupervisor::OpenGpuSession(HLERequestContext& ctx) {
    const auto* packet = ctx.CommandBuffer();
    Result result = Kernel::ResultInvalidCombination;
    Kernel::KClientSession* client{};
    const u64 flags = u64{packet[2]} | (u64{packet[3]} << 32);
    if (packet[0] == 0x201 && packet[1] == 2 && flags == 4) {
        if (m_phase != Phase::Ready || !m_nv_module) {
            result = Kernel::ResultInvalidState;
        } else {
            auto handler = std::make_shared<IGpuSession>(system, m_nv_module, m_process);
            result = CreateGraphicsSession(system, ctx, handler, &client);
            if (result.IsSuccess()) {
                std::erase_if(m_gpu_sessions, [](const auto& item) { return item.expired(); });
                m_gpu_sessions.emplace_back(handler);
            }
        }
    }
    LOG_DEBUG(Service_AM, "GPU control session flags={:#x}, result={:#x}", flags, result.raw);
    // This SDK reads the move-handle slot before the result, including on failure.
    IPC::ResponseBuilder rb{ctx, 2, 0, 1};
    rb.Push(result.raw);
    ctx.AddMoveObject(client);
}

void IGraphicsSupervisor::RegisterGpuAddressSpace(HLERequestContext& ctx) {
    const auto* packet = ctx.CommandBuffer();
    Result result = Kernel::ResultInvalidCombination;
    if (packet[0] == 0x1c2 && packet[1] == 1) {
        result = m_gpu_context_ready ? Kernel::ResultNotFound : Kernel::ResultInvalidState;
        if (m_gpu_context_ready) {
            for (const auto& item : m_gpu_sessions) {
                if (auto handler = item.lock()) {
                    result = handler->RegisterAddressSpace(packet[2]);
                    if (result.IsSuccess()) {
                        break;
                    }
                }
            }
        }
    }
    LOG_DEBUG(Service_AM, "register GPU address space id={}, result={:#x}", packet[2], result.raw);
    IPC::ResponseBuilder rb{ctx, 2};
    rb.Push(result.raw);
}

void IGraphicsSupervisor::InitializeDisplayContext(HLERequestContext& ctx) {
    const auto* packet = ctx.CommandBuffer();
    Result result = Kernel::ResultInvalidCombination;
    if (packet[0] == 0x13d && packet[1] == 0) {
        result = m_phase == Phase::Ready ? ResultSuccess : Kernel::ResultInvalidState;
        if (result.IsSuccess()) {
            m_display_context_initialized = true;
        }
    }
    LOG_DEBUG(Service_AM, "display context initialization result={:#x}", result.raw);
    IPC::ResponseBuilder rb{ctx, 2};
    rb.Push(result.raw);
}

void IGraphicsSupervisor::OpenDisplaySession(HLERequestContext& ctx) {
    const auto* packet = ctx.CommandBuffer();
    Kernel::KClientSession* client{};
    const u64 flags = u64{packet[2]} | (u64{packet[3]} << 32);
    Result result = Kernel::ResultInvalidCombination;
    if (packet[0] == 0x141 && packet[1] == 2 && flags == 0x300020000ULL) {
        result = m_display_context_initialized ? ResultSuccess : Kernel::ResultInvalidState;
        if (result.IsSuccess()) {
            result = CreateGraphicsSession(system, ctx, std::make_shared<IVi2Session>(system, m_aruid, m_image_registry),
                                           &client);
        }
    }
    LOG_DEBUG(Service_AM, "VI2 session flags={:#x}, result={:#x}", flags, result.raw);
    IPC::ResponseBuilder rb{ctx, 2, 0, 1};
    rb.Push(result.raw);
    ctx.AddMoveObject(client);
}

void IGraphicsSupervisor::InitializeGpuContext(HLERequestContext& ctx) {
    const auto* packet = ctx.CommandBuffer();
    Result result = Kernel::ResultInvalidCombination;
    if (packet[0] == 16 + ctx.GetCommand() && packet[1] == 0) {
        if (m_phase != Phase::Ready) {
            result = Kernel::ResultInvalidState;
        } else if (ctx.GetCommand() == 401) {
            m_gpu_context_initialized = true;
            result = ResultSuccess;
        } else if (m_gpu_context_initialized) {
            m_gpu_context_ready = true;
            result = ResultSuccess;
        } else {
            result = Kernel::ResultInvalidState;
        }
    }
    LOG_DEBUG(Service_AM, "GPU context command={}, result={:#x}", ctx.GetCommand(), result.raw);
    IPC::ResponseBuilder rb{ctx, 2};
    rb.Push(result.raw);
}

void IGraphicsSupervisor::InitializeFrameContext(HLERequestContext& ctx) {
    const auto* packet = ctx.CommandBuffer();
    Result result = Kernel::ResultInvalidCombination;
    if (packet[0] == 16 + ctx.GetCommand() && packet[1] == 0) {
        if (m_phase != Phase::Ready) {
            result = Kernel::ResultInvalidState;
        } else if (ctx.GetCommand() == 1101) {
            m_frame_context_initialized = true;
            result = ResultSuccess;
        } else if (m_frame_context_initialized) {
            m_frame_context_ready = true;
            result = ResultSuccess;
        } else {
            result = Kernel::ResultInvalidState;
        }
    }
    LOG_DEBUG(Service_AM, "empty frame context command={}, result={:#x}", ctx.GetCommand(), result.raw);
    IPC::ResponseBuilder rb{ctx, 2};
    rb.Push(result.raw);
}

void IGraphicsSupervisor::ConfigureNativeMemory(HLERequestContext& ctx) {
    const auto* packet = ctx.CommandBuffer();
    Result result = Kernel::ResultInvalidCombination;
    if (packet[0] == 0x83 && packet[1] == 2 && packet[3] == 0 &&
        (packet[2] == 0 || packet[2] == 2)) {
        result = m_phase == Phase::Ready ? ResultSuccess : Kernel::ResultInvalidState;
    }
    // Clear the optional Gpuk-memory mode: Eden currently provides the NVDRV path.
    IPC::ResponseBuilder rb{ctx, 4};
    rb.Push(result.raw);
    rb.Push<u64>(0);
    LOG_DEBUG(Service_AM, "native memory uses NVDRV-compatible mode, result={:#x}", result.raw);
}

void IGraphicsSupervisor::InitializeSyncpointContext(HLERequestContext& ctx) {
    Result result = Kernel::ResultInvalidCombination;
    const auto* packet = ctx.CommandBuffer();
    if (packet[0] == 0xd9 && packet[1] == 0) {
        if (m_phase != Phase::Ready) {
            result = Kernel::ResultInvalidState;
        } else {
            m_syncpoint_context_initialized = true;
            result = ResultSuccess;
        }
    }
    LOG_DEBUG(Service_AM, "syncpoint context initialization result={:#x}", result.raw);
    IPC::ResponseBuilder rb{ctx, 2};
    rb.Push(result.raw);
}

void IGraphicsSupervisor::GetSyncpointSession(HLERequestContext& ctx) {
    const auto* packet = ctx.CommandBuffer();
    Result result = Kernel::ResultInvalidState;
    Kernel::KClientSession* client{};
    if (packet[0] == 0xda && packet[1] == 0 && m_syncpoint_context_initialized && m_nv_module) {
        result = CreateGraphicsSession(system, ctx,
            std::make_shared<ISyncpointSupervisor>(system, m_nv_module), &client);
    }
    if (result.IsError()) {
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(result.raw);
        return;
    }
    IPC::ResponseBuilder rb{ctx, 2, 0, 1};
    rb.Push(ResultSuccess.raw);
    rb.PushMoveObjects(ctx, client);
    LOG_DEBUG(Service_AM, "opened syncpoint-supervisor session");
}

// Camera (6xx) and GameChat image (12xx) clients share one protocol: command N+1 initializes
// the context, N+2 opens a CMIF session and returns it as a move handle, and N+3 finalizes the
// context, which the SDK also sends to undo N+1 when N+2 fails. Applications abort if these
// fail, even when no camera or chat is available.
void IGraphicsSupervisor::InitializeSubsystemContext(HLERequestContext& ctx, bool& initialized) {
    const auto* packet = ctx.CommandBuffer();
    Result result = Kernel::ResultInvalidCombination;
    if (packet[1] == 0) {
        result = initialized ? Kernel::ResultInvalidState : ResultSuccess;
        initialized = true;
    }
    LOG_DEBUG(Service_AM, "context initialization command={}, result={:#x}", ctx.GetCommand(),
              result.raw);
    IPC::ResponseBuilder rb{ctx, 2};
    rb.Push(result.raw);
}

void IGraphicsSupervisor::OpenSubsystemSession(HLERequestContext& ctx, bool initialized,
                                               u32 data_words,
                                               std::shared_ptr<SessionRequestHandler> handler) {
    const auto* packet = ctx.CommandBuffer();
    Result result = Kernel::ResultInvalidCombination;
    Kernel::KClientSession* client{};
    if (packet[1] == data_words) {
        result = initialized ? CreateGraphicsSession(system, ctx, std::move(handler), &client)
                             : Kernel::ResultInvalidState;
    }
    LOG_DEBUG(Service_AM, "session command={}, result={:#x}", ctx.GetCommand(), result.raw);
    // The SDK reads the move-handle slot before the result, as for OpenGpuSession.
    IPC::ResponseBuilder rb{ctx, 2, 0, 1};
    rb.Push(result.raw);
    ctx.AddMoveObject(client);
}

void IGraphicsSupervisor::FinalizeSubsystemContext(HLERequestContext& ctx, bool& initialized) {
    const auto* packet = ctx.CommandBuffer();
    Result result = Kernel::ResultInvalidCombination;
    if (packet[1] == 0) {
        result = initialized ? ResultSuccess : Kernel::ResultInvalidState;
        initialized = false;
    }
    IPC::ResponseBuilder rb{ctx, 2};
    rb.Push(result.raw);
}

void IGraphicsSupervisor::InitializeCameraContext(HLERequestContext& ctx) {
    InitializeSubsystemContext(ctx, m_camera_context_initialized);
}

void IGraphicsSupervisor::OpenCameraSession(HLERequestContext& ctx) {
    // The request carries one u64, 1 for the only observed session kind.
    OpenSubsystemSession(ctx, m_camera_context_initialized, 2,
                         std::make_shared<ICameraSession>(system));
}

void IGraphicsSupervisor::FinalizeCameraContext(HLERequestContext& ctx) {
    FinalizeSubsystemContext(ctx, m_camera_context_initialized);
}

// nn::codec::video::detail::ipc::Initialize: 901 and 1001 initialize the decoder and encoder
// contexts, 902 and 1002 open their root sessions. 905 is 902 with a u8 that lifts the default
// decode resolution limit (ShimLibraryGlobal::SetResolutionUnlimitedModeEnabled).
// nn::codec::CalculateDecoderImagePoolBuffer: a VideoDecoderImagePreference (16 words: the type
// and the inclusive maximum x and y, nothing else set) and an image count. Sizes come from
// VideoImageSize.
void IGraphicsSupervisor::CalculateDecoderImagePoolBuffer(HLERequestContext& ctx) {
    const auto* packet = ctx.CommandBuffer();
    Result result = Kernel::ResultInvalidCombination;
    u64 size{};
    constexpr u64 Alignment = 0x10000;
    if (packet[0] == 0x3ef && packet[1] == 17 && packet[2] == 0 &&
        VideoImageFormatFromPreference(packet[3])) {
        const u64 width = (packet[4] & 0xffffU) + 1;
        const u64 height = (packet[4] >> 16) + 1;
        const u32 count = packet[18];
        result = count != 0 && width <= 8192 && height <= 8192 ? ResultSuccess
                                                               : Kernel::ResultInvalidSize;
        size = VideoImageSize(static_cast<u32>(width), static_cast<u32>(height)) * count;
        LOG_DEBUG(Service_AM, "decoder image pool {}x{} count={}, size={:#x}", width, height,
                  count, size);
    }
    IPC::ResponseBuilder rb{ctx, 6};
    rb.Push(result.raw);
    rb.Push<u64>(size);
    rb.Push<u64>(Alignment);
}

// nn::codec::InitializeDecoderImagePool: the VideoDecoderImagePreference (16 words), the image
// count and the native memory holding the pool. The reply is the handle of image 0; image i has
// the handle plus i.
void IGraphicsSupervisor::CreateDecoderImagePool(HLERequestContext& ctx) {
    const auto* packet = ctx.CommandBuffer();
    Result result = Kernel::ResultInvalidCombination;
    u32 handle{};
    const auto format = VideoImageFormatFromPreference(packet[3]);
    if (packet[0] == 0x39f && packet[1] == 18 && packet[2] == 0 && format) {
        const u32 width = (packet[4] & 0xffffU) + 1;
        const u32 height = (packet[4] >> 16) + 1;
        const u32 count = packet[18];
        const u32 memory_id = packet[19];
        result = Kernel::ResultNotFound;
        if (m_nv_module && std::ranges::find(m_native_memories, memory_id) != m_native_memories.end()) {
            const auto memory = m_nv_module->GetContainer().GetNvMapFile().GetHandle(memory_id);
            result = Kernel::ResultInvalidSize;
            if (memory && memory->address && count &&
                VideoImageSize(width, height) * count <= memory->size) {
                handle = m_next_native_image_id;
                m_next_native_image_id += count;
                m_video_image_pools->Add({*format, handle, count, width, height,
                                          memory->address, memory->size, memory_id});
                result = ResultSuccess;
            }
        }
        LOG_DEBUG(Service_AM, "decoder image pool {}x{} count={}, memory={}, handle={:#x}, result={:#x}",
                  width, height, count, memory_id, handle, result.raw);
    }
    IPC::ResponseBuilder rb{ctx, 3};
    rb.Push(result.raw);
    rb.Push(handle);
}

void IGraphicsSupervisor::DestroyDecoderImagePool(HLERequestContext& ctx) {
    const auto* packet = ctx.CommandBuffer();
    Result result = Kernel::ResultInvalidCombination;
    if (packet[0] == 0x3a9 && packet[1] == 2) {
        result = m_video_image_pools->Remove(packet[2]) ? ResultSuccess : Kernel::ResultNotFound;
    }
    IPC::ResponseBuilder rb{ctx, 2};
    rb.Push(result.raw);
}

// The nn::gsv::CpuUserImageDescriptor through which nn::gs::sdk::ReadUserImage{Rgba,Nv12} copy
// a picture out of the pool. Format 8 is pitch-linear RGBA, offset at 8 and pitch (u64) at 0x10;
// format 0xa is pitch-linear NV12, adding the chroma offset at 0x18 and pitch at 0x20. Offsets
// are relative to the pool buffer.
void IGraphicsSupervisor::GetDecoderImageDescriptor(HLERequestContext& ctx) {
    const auto* packet = ctx.CommandBuffer();
    Result result = Kernel::ResultInvalidCombination;
    std::array<u32, 32> descriptor{};
    if (packet[0] == 0x3d1 && packet[1] == 1) {
        const auto image = m_video_image_pools->Find(packet[2]);
        result = image ? ResultSuccess : Kernel::ResultNotFound;
        if (image) {
            descriptor[0] = image->format == VideoImageFormat::Rgba ? 0x8 : 0xa;
            descriptor[2] = static_cast<u32>(image->layout.luma_offset);
            descriptor[4] = static_cast<u32>(image->layout.luma_pitch);
            descriptor[6] = static_cast<u32>(image->layout.chroma_offset);
            descriptor[8] = static_cast<u32>(image->layout.chroma_pitch);
        }
    }
    IPC::ResponseBuilder rb{ctx, 34};
    rb.Push(result.raw);
    for (const u32 word : descriptor) {
        rb.Push(word);
    }
}

// nn::gpu::sdk::QueryNativeImage, through which nvnNativeImageInitialize imports an image as a
// memory pool and nvnTextureInitializeFromNativeImage builds a texture on it. The reply has two
// reserved words, the u64 base address, the 0x80 byte nn::gsv::GpuOunceUserImageDescriptor and
// the GpuOunceUserImageFlags word. Descriptor: byte 0 is 1; word 2 holds width - 1 and height - 1
// as u16; the u64 at 0x20 has 1 in its low byte and the layout in its high word (1: RGBA, NVN
// formats RGBA8/BGRA8 and their sRGB forms; 2: YUV, formats 0x99-0x9c); the u64 at 0x28 holds the
// image offset from the base (low) and its size (high). The memory pool spans the pages of the
// image and the texture starts at the offset within the first page. No layout flag is passed to
// the texture builder, so the image is block linear with the default block height.
std::optional<std::array<u32, 32>> IGraphicsSupervisor::MakeGpuImageDescriptor(u32 handle,
                                                                               u64 offset) {
    const auto image = m_video_image_pools->Find(handle);
    if (!image || image->format != VideoImageFormat::Rgba) {
        return std::nullopt;
    }
    m_video_image_pools->SetBlockLinear(handle);
    std::array<u32, 32> descriptor{};
    descriptor[0] = 1;
    descriptor[2] = (image->width - 1) | ((image->height - 1) << 16);
    descriptor[8] = 1;
    descriptor[9] = 1;
    descriptor[10] = static_cast<u32>(offset);
    descriptor[11] = static_cast<u32>(Tegra::Texture::CalculateSize(
        true, 4, image->width, image->height, 1, VideoImageBlockHeight(image->height), 0));
    return descriptor;
}

void IGraphicsSupervisor::QueryNativeImage(HLERequestContext& ctx) {
    const auto* packet = ctx.CommandBuffer();
    Result result = Kernel::ResultInvalidCombination;
    u64 base{};
    std::array<u32, 32> descriptor{};
    if (packet[0] == 0x1bc && packet[1] == 1) {
        const auto image = m_video_image_pools->Find(packet[2]);
        const auto image_descriptor =
            image ? MakeGpuImageDescriptor(packet[2], image->layout.luma_offset) : std::nullopt;
        result = Kernel::ResultNotImplemented;
        if (image_descriptor) {
            base = image->address;
            descriptor = *image_descriptor;
            result = ResultSuccess;
        }
        LOG_DEBUG(Service_AM, "native image query {:#x}, result={:#x}", packet[2], result.raw);
    }
    IPC::ResponseBuilder rb{ctx, 39};
    rb.Push(result.raw);
    rb.Push<u32>(0);
    rb.Push<u32>(0);
    rb.Push<u64>(base);
    for (const u32 word : descriptor) {
        rb.Push(word);
    }
    rb.Push<u32>(0);
}

// nn::gpu::sdk::MapNativeImage: the image handle, a reserved 1, the GPU address space (an
// nvhost-as descriptor) and a permission byte. The reply is the GPU address of the image, the
// size of the mapping and the descriptor and flags of QueryNativeImage, the image offset in the
// descriptor counting from the returned address. UnmapNativeImage takes the image, the address
// space and the address and size from the map reply.
void IGraphicsSupervisor::MapNativeImage(HLERequestContext& ctx) {
    const auto* packet = ctx.CommandBuffer();
    Result result = Kernel::ResultInvalidCombination;
    u64 address{};
    u64 size{};
    std::array<u32, 32> descriptor{};
    if (packet[0] == 0x1ba && packet[1] == 4 && packet[3] == 1) {
        const auto image = m_video_image_pools->Find(packet[2]);
        const auto image_descriptor = image ? MakeGpuImageDescriptor(packet[2], 0) : std::nullopt;
        result = Kernel::ResultNotImplemented;
        if (image_descriptor && m_nv_module) {
            result = Kernel::ResultNotFound;
            if (const auto as =
                    m_nv_module->GetDevice<Nvidia::Devices::nvhost_as_gpu>(packet[4])) {
                size = Common::AlignUp((*image_descriptor)[11], 0x1000);
                address = as->MapNativeImage(image->memory_id, image->layout.luma_offset, size,
                                             static_cast<u8>(Tegra::PTEKind::GENERIC_16BX2));
                result = address ? ResultSuccess : Kernel::ResultOutOfAddressSpace;
                descriptor = *image_descriptor;
            }
        }
        LOG_DEBUG(Service_AM, "native image map {:#x} as={}, permission={}, address={:#x}, size={:#x}, result={:#x}",
                  packet[2], packet[4], packet[5] & 0xff, address, size, result.raw);
    }
    if (result.IsError()) {
        size = 0;
    }
    IPC::ResponseBuilder rb{ctx, 39};
    rb.Push(result.raw);
    rb.Push<u64>(address);
    rb.Push<u64>(size);
    for (const u32 word : descriptor) {
        rb.Push(word);
    }
    rb.Push<u32>(0);
}

void IGraphicsSupervisor::UnmapNativeImage(HLERequestContext& ctx) {
    const auto* packet = ctx.CommandBuffer();
    Result result = Kernel::ResultInvalidCombination;
    if (packet[0] == 0x1bb && packet[1] == 6) {
        const u64 address = u64{packet[4]} | (u64{packet[5]} << 32);
        result = Kernel::ResultNotFound;
        if (m_nv_module && m_video_image_pools->Find(packet[2])) {
            if (const auto as =
                    m_nv_module->GetDevice<Nvidia::Devices::nvhost_as_gpu>(packet[3])) {
                result = as->UnmapNativeImage(address) == Nvidia::NvResult::Success
                             ? ResultSuccess
                             : Kernel::ResultInvalidAddress;
            }
        }
        LOG_DEBUG(Service_AM, "native image unmap {:#x} as={}, address={:#x}, result={:#x}",
                  packet[2], packet[3], address, result.raw);
    }
    IPC::ResponseBuilder rb{ctx, 2};
    rb.Push(result.raw);
}

void IGraphicsSupervisor::InitializeVideoDecoderContext(HLERequestContext& ctx) {
    InitializeSubsystemContext(ctx, m_video_decoder_context_initialized);
}

void IGraphicsSupervisor::OpenVideoDecoderSession(HLERequestContext& ctx) {
    OpenSubsystemSession(ctx, m_video_decoder_context_initialized,
                         ctx.GetCommand() == 905 ? 1 : 0,
                         MakeVideoCodecRootSession(system, VideoCodecKind::Decoder, m_process,
                                                   m_nv_module, m_video_image_pools));
}

void IGraphicsSupervisor::FinalizeVideoDecoderContext(HLERequestContext& ctx) {
    FinalizeSubsystemContext(ctx, m_video_decoder_context_initialized);
}

void IGraphicsSupervisor::InitializeVideoEncoderContext(HLERequestContext& ctx) {
    InitializeSubsystemContext(ctx, m_video_encoder_context_initialized);
}

void IGraphicsSupervisor::OpenVideoEncoderSession(HLERequestContext& ctx) {
    OpenSubsystemSession(ctx, m_video_encoder_context_initialized, 0,
                         MakeVideoCodecRootSession(system, VideoCodecKind::Encoder, m_process,
                                                   m_nv_module, m_video_image_pools));
}

void IGraphicsSupervisor::FinalizeVideoEncoderContext(HLERequestContext& ctx) {
    FinalizeSubsystemContext(ctx, m_video_encoder_context_initialized);
}

void IGraphicsSupervisor::OpenGpuTimeSession(HLERequestContext& ctx) {
    // Opened on first use without a context command.
    OpenSubsystemSession(ctx, true, 0, std::make_shared<IGpuTimeSession>(system));
}

void IGraphicsSupervisor::InitializeChatImageContext(HLERequestContext& ctx) {
    InitializeSubsystemContext(ctx, m_chat_image_context_initialized);
}

void IGraphicsSupervisor::OpenChatImageSession(HLERequestContext& ctx) {
    OpenSubsystemSession(ctx, m_chat_image_context_initialized, 0,
                         std::make_shared<IChatImageSession>(system));
}

void IGraphicsSupervisor::FinalizeChatImageContext(HLERequestContext& ctx) {
    FinalizeSubsystemContext(ctx, m_chat_image_context_initialized);
}

// nn::gs::sdk::SetGraphicsFrame{Length,Restart,GpuTimeslice}Mode send one u32 each. The SDK
// translates length modes 0..3 to 0, 0x11e, 0x13c and 0x178 and checks restart modes below 3
// and timeslice modes below 2. Frame pacing is driven by the host, so the values are recorded.
void IGraphicsSupervisor::SetGraphicsFrameMode(HLERequestContext& ctx) {
    static constexpr std::array<u32, 4> FrameLengths{0, 0x11e, 0x13c, 0x178};
    const auto* packet = ctx.CommandBuffer();
    const u32 command = ctx.GetCommand();
    Result result = Kernel::ResultInvalidCombination;
    if (packet[1] == 1) {
        const u32 value = packet[2];
        const bool valid = command == 1111 ? std::ranges::find(FrameLengths, value) != FrameLengths.end()
                         : command == 1112 ? value < 3
                                           : value < 2;
        result = valid ? ResultSuccess : Kernel::ResultInvalidEnumValue;
        if (valid) {
            m_frame_modes[command - 1111] = value;
        }
    }
    LOG_DEBUG(Service_AM, "graphics frame mode command={}, value={:#x}, result={:#x}", command,
              packet[2], result.raw);
    IPC::ResponseBuilder rb{ctx, 2};
    rb.Push(result.raw);
}

void IGraphicsSupervisor::GetNvidiaDriver(HLERequestContext& ctx) {
    const auto Fail = [&](Result result) {
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(result.raw);
    };
    const auto* packet = ctx.CommandBuffer();
    if (packet[0] != 0x58a || packet[1] != 0) {
        Fail(Kernel::ResultInvalidCombination);
        return;
    }
    if (m_phase != Phase::Ready || !m_image_formats_initialized) {
        Fail(Kernel::ResultInvalidState);
        return;
    }
    auto source = system.ServiceManager().GetService<Nvidia::NVDRV>("nvdrv:s", true);
    if (!source) {
        Fail(Kernel::ResultNotFound);
        return;
    }
    m_nv_module = source->GetModule();
    Kernel::KClientSession* client{};
    const auto result = CreateGraphicsSession(system, ctx,
        std::make_shared<Nvidia::NVDRV>(system, m_nv_module, "nvdrv:gsv", m_process, m_aruid),
        &client);
    if (result.IsError()) {
        Fail(result);
        return;
    }
    LOG_DEBUG(Service_AM, "opened process-bound NVDRV session for aruid={:#x}", m_aruid);
    IPC::ResponseBuilder rb{ctx, 2, 0, 1};
    rb.Push(ResultSuccess.raw);
    rb.PushMoveObjects(ctx, client);
}

void IGraphicsSupervisor::BindSyncpointMemory(HLERequestContext& ctx) {
    using namespace GraphicsSyncpointCache;
    const auto Fail = [&](Result result) {
        IPC::ResponseBuilder rb{ctx, 2, 0, 2};
        rb.Push(result.raw);
        ctx.AddMoveObject(nullptr);
        ctx.AddMoveObject(nullptr);
    };
    const auto* packet = ctx.CommandBuffer();
    if (packet[0] != 0xdc || packet[1] != 0x80000006 || packet[2] != 4 ||
        packet[5] != 1 || packet[6] != 0x1ff || packet[7] != 0x1000 || packet[8] != 0 ||
        packet[9] != 0x2000 || packet[10] != 0) {
        Fail(Kernel::ResultInvalidCombination);
        return;
    }
    if (!m_syncpoint_context_initialized || !m_nv_module || m_syncpoint_rw) {
        Fail(Kernel::ResultInvalidState);
        return;
    }
    auto rw = ctx.GetObjectFromHandle<Kernel::KTransferMemory>(ctx.GetCopyHandle(0));
    auto ro = ctx.GetObjectFromHandle<Kernel::KTransferMemory>(ctx.GetCopyHandle(1));
    if (rw.IsNull() || ro.IsNull() || rw->GetOwner() != m_process || ro->GetOwner() != m_process ||
        rw.GetPointerUnsafe() == ro.GetPointerUnsafe()) {
        Fail(Kernel::ResultInvalidHandle);
        return;
    }
    if (rw->GetSize() != 0x1000 || ro->GetSize() != 0x2000) {
        Fail(Kernel::ResultInvalidSize);
        return;
    }
    auto* request_event = m_context.CreateEvent("SyncpointCacheRequest");
    if (request_event == nullptr) {
        Fail(Kernel::ResultOutOfResource);
        return;
    }
    m_syncpoint_request_event = request_event;
    m_syncpoint_rw = rw.GetPointerUnsafe();
    m_syncpoint_ro = ro.GetPointerUnsafe();
    m_syncpoint_rw->Open(kernel);
    m_syncpoint_ro->Open(kernel);
    auto& memory = m_process->GetMemory();
    const auto rw_base = rw->GetSourceAddress();
    const auto ro_base = ro->GetSourceAddress();
    const std::array<u8, 0x2000> zeros{};
    memory.WriteBlock(rw_base, zeros.data(), 0x1000);
    memory.WriteBlock(ro_base, zeros.data(), 0x2000);
    memory.WriteBlock(ro_base, Header.data(), sizeof(Header));
    for (u32 id = 0; id < CounterCount; ++id) {
        memory.Write64(ro_base + ValuesOffset + id * sizeof(u64), Unallocated);
    }
    m_cache_poll = Core::Timing::CreateEvent("GraphicsSyncpointCache",
        [this](s64, std::chrono::nanoseconds) -> std::optional<std::chrono::nanoseconds> {
            const auto guard = LockService();
            this->RefreshSyncpointCache();
            return std::nullopt;
        });
    system.CoreTiming().ScheduleLoopingEvent(std::chrono::milliseconds{1},
                                             std::chrono::milliseconds{1}, m_cache_poll);
    request_event->Open(kernel);
    auto* completed = m_syncpoint_cache_event.GetHandle();
    completed->Open(kernel);
    IPC::ResponseBuilder rb{ctx, 2, 0, 2};
    rb.Push(ResultSuccess.raw);
    ctx.AddMoveObject(request_event);
    ctx.AddMoveObject(completed);
    LOG_DEBUG(Service_AM, "bound syncpoint cache: {} real counters, rw={:#x}, ro={:#x}",
              CounterCount, GetInteger(rw_base), GetInteger(ro_base));
}

void IGraphicsSupervisor::RefreshSyncpointCache() {
    using namespace GraphicsSyncpointCache;
    if (!m_syncpoint_rw || !system.IsPoweredOn()) {
        return;
    }
    auto& memory = m_process->GetMemory();
    const auto rw = m_syncpoint_rw->GetSourceAddress();
    const auto ro = m_syncpoint_ro->GetSourceAddress();
    const u64 requests = memory.Read64(rw + GroupRequestOffset);
    const u64 acknowledgements = memory.Read64(ro + GroupAckOffset);
    u64 groups = (requests ^ acknowledgements) & ((1ULL << GroupCount) - 1);
    if (groups == 0) {
        return;
    }
    auto& syncpoints = m_nv_module->GetContainer().GetSyncpointManager();
    while (groups) {
        const u32 group = static_cast<u32>(std::countr_zero(groups));
        groups &= groups - 1;
        const u64 requested_slots = memory.Read64(rw + SlotRequestOffset + group * sizeof(u64));
        u64 slots = requested_slots ^ memory.Read64(ro + SlotAckOffset + group * sizeof(u64));
        while (slots) {
            const u32 bit = static_cast<u32>(std::countr_zero(slots));
            slots &= slots - 1;
            const u32 id = group * 64 + bit;
            const auto address = ro + ValuesOffset + id * sizeof(u64);
            const auto value = syncpoints.ReadAllocatedSyncpoint(id);
            const u64 updated = m_counter_values[id].Observe(value);
            memory.Write64(address, updated);
        }
        std::atomic_thread_fence(std::memory_order_release);
        memory.Write64(ro + SlotAckOffset + group * sizeof(u64), requested_slots);
    }
    std::atomic_thread_fence(std::memory_order_release);
    memory.Write64(ro + GroupAckOffset, requests);
    m_syncpoint_request_event->Clear(kernel);
    m_syncpoint_cache_event.Signal(kernel);
}

void IGraphicsSupervisor::CalculateImageRequirements(HLERequestContext& ctx) {
    const auto* packet = ctx.CommandBuffer();
    Result result = Kernel::ResultInvalidCombination;
    std::optional<NativeImageRequirements> requirements;
    const bool gpu_image = ctx.GetCommand() == 435;
    if (packet[0] == (gpu_image ? 0x1c3U : 0x5f6U) && packet[1] == 17 &&
        packet[3] == (gpu_image ? 1U : 2U)) {
        result = Kernel::ResultInvalidState;
        if (m_phase == Phase::Ready && m_image_formats_initialized) {
            requirements = CalculateNativeImageRequirements(std::span{packet + 2, 16}, packet[18]);
            result = requirements ? ResultSuccess : Kernel::ResultNotImplemented;
        }
    }
    LOG_DEBUG(Service_AM, "native image requirements dimensions={:#x}, block={}, pitch={}, size={:#x}, alignment={:#x}, result={:#x}",
              packet[4], packet[6], packet[7], requirements ? requirements->size : 0,
              requirements ? requirements->alignment : 0, result.raw);
    IPC::ResponseBuilder rb{ctx, gpu_image ? 4U : 6U};
    rb.Push(result.raw);
    rb.Push<u64>(requirements ? requirements->size : 0);
    if (!gpu_image) { rb.Push<u64>(requirements ? requirements->alignment : 0); }
}
void IGraphicsSupervisor::CreateNativeMemory(HLERequestContext& ctx) {
    const auto* packet = ctx.CommandBuffer();
    const u64 size = u64{packet[3]} | (u64{packet[4]} << 32);
    Result result = Kernel::ResultInvalidCombination;
    u32 id{};
    if (packet[0] == 0x7f && packet[1] == 3 && (packet[2] & 0xff) == 3) {
        result = Kernel::ResultInvalidState;
        if (m_phase == Phase::Ready && m_image_formats_initialized && m_nv_module) {
            result = Kernel::ResultInvalidSize;
            if (size && size <= (1ULL << 32) && Common::IsAligned(size, Kernel::PageSize)) {
                result = Kernel::ResultOutOfResource;
                if (m_native_memories.size() < 1024) {
                    auto& nvmap = m_nv_module->GetContainer().GetNvMapFile();
                    std::shared_ptr<Nvidia::NvCore::NvMap::Handle> handle;
                    if (nvmap.CreateHandle(size, handle) == Nvidia::NvResult::Success) {
                        id = handle->id;
                        m_native_memories.push_back(id);
                        result = ResultSuccess;
                    }
                }
            }
        }
    }
    LOG_DEBUG(Service_AM, "native memory create size={:#x}, id={}, result={:#x}", size, id, result.raw);
    IPC::ResponseBuilder rb{ctx, 3};
    rb.Push(result.raw);
    rb.Push(id);
}

Result IGraphicsSupervisor::BindNativeMemoryBacking(u32 id, u64 address, u64 size, u64 flags, u64 offset) {
    R_UNLESS(m_nv_module && std::ranges::find(m_native_memories, id) != m_native_memories.end(),
             Kernel::ResultNotFound);
    R_UNLESS((flags == 0 || flags == 2) && offset == 0, Kernel::ResultNotImplemented);
    auto handle = m_nv_module->GetContainer().GetNvMapFile().GetHandle(id);
    R_UNLESS(handle && !handle->allocated, Kernel::ResultInvalidState);
    R_UNLESS(size && size == handle->size && address != 0 && address + size > address,
             Kernel::ResultInvalidSize);
    R_UNLESS(Common::IsAligned(address, Kernel::PageSize), Kernel::ResultInvalidAddress);
    auto& page_table = m_process->GetPageTable();
    R_UNLESS(page_table.Contains(address, size), Kernel::ResultInvalidCurrentMemory);
    if (flags == 2) {
        for (u64 current = address; current < address + size;) {
            Kernel::KMemoryInfo info{};
            Kernel::Svc::PageInfo page{};
            R_TRY(page_table.QueryInfo(&info, &page, current));
            R_UNLESS(True(info.m_attribute & Kernel::KMemoryAttribute::GpuAccessible),
                     Kernel::ResultInvalidCurrentMemory);
            const u64 next = info.m_address + info.m_size;
            R_UNLESS(next > current, Kernel::ResultInvalidCurrentMemory);
            current = next;
        }
    }
    bool is_io{};
    R_TRY(page_table.LockForMapDeviceAddressSpace(&is_io, address, size,
                                                 Kernel::KMemoryPermission::None, true, true));
    if (is_io) {
        page_table.UnlockForDeviceAddressSpace(address, size);
        R_RETURN(Kernel::ResultNotImplemented);
    }
    if (!m_native_memory_session) {
        m_native_memory_session = m_nv_module->GetContainer().OpenSession(m_process).id;
    }
    const auto result = handle->Alloc({}, Kernel::PageSize, 0, address,
                                      Nvidia::NvCore::SessionId{*m_native_memory_session});
    if (result != Nvidia::NvResult::Success) {
        page_table.UnlockForDeviceAddressSpace(address, size);
        R_RETURN(Kernel::ResultOutOfResource);
    }
    R_SUCCEED();
}

void IGraphicsSupervisor::BindNativeMemory(HLERequestContext& ctx) {
    const auto* packet = ctx.CommandBuffer();
    const auto pair = [&](size_t i) { return u64{packet[i]} | (u64{packet[i + 1]} << 32); };
    const auto result = packet[0] == 0x82 && packet[1] == 9
        ? BindNativeMemoryBacking(packet[2], pair(3), pair(5), pair(7), pair(9))
        : Kernel::ResultInvalidCombination;
    LOG_DEBUG(Service_AM, "native memory bind id={}, address={:#x}, size={:#x}, flags={:#x}, result={:#x}",
              packet[2], pair(3), pair(5), pair(7), result.raw);
    IPC::ResponseBuilder rb{ctx, 2};
    rb.Push(result.raw);
}

Result IGraphicsSupervisor::ReleaseNativeMemory(u32 id) {
    const auto it = std::ranges::find(m_native_memories, id);
    R_UNLESS(m_nv_module && it != m_native_memories.end(), Kernel::ResultNotFound);
    auto& nvmap = m_nv_module->GetContainer().GetNvMapFile();
    auto handle = nvmap.GetHandle(id);
    R_UNLESS(handle, Kernel::ResultNotFound);
    {
        const std::scoped_lock lock{handle->mutex};
        R_UNLESS(handle->pins == 0 && handle->internal_dupes == 0 && handle->dupes == 1,
                 Kernel::ResultBusy);
    }
    const bool allocated = handle->allocated;
    handle.reset();
    const auto info = nvmap.FreeHandle(id, false);
    m_native_memories.erase(it);
    if (allocated) {
        R_UNLESS(info && info->can_unlock, Kernel::ResultInvalidState);
        R_TRY(m_process->GetPageTable().UnlockForDeviceAddressSpace(info->address, info->size));
    }
    R_SUCCEED();
}

void IGraphicsSupervisor::DestroyNativeMemory(HLERequestContext& ctx) {
    const auto* packet = ctx.CommandBuffer();
    const auto result = packet[0] == 0x80 && packet[1] == 1
        ? ReleaseNativeMemory(packet[2]) : Kernel::ResultInvalidCombination;
    LOG_DEBUG(Service_AM, "native memory destroy id={}, result={:#x}", packet[2], result.raw);
    IPC::ResponseBuilder rb{ctx, 2};
    rb.Push(result.raw);
}

void IGraphicsSupervisor::BindNativeGpuMemory(HLERequestContext& ctx) {
    const auto* packet = ctx.CommandBuffer();
    const auto pair = [&](size_t i) { return u64{packet[i]} | (u64{packet[i + 1]} << 32); };
    Result result = Kernel::ResultInvalidCombination;
    if (packet[0] == 0x1b7 && packet[1] == 12 && pair(10) == 0 && pair(12) == 0) {
        result = Kernel::ResultNotFound;
        if (m_nv_module) {
            auto& container = m_nv_module->GetContainer();
            const auto parent = container.GetNvMapFile().GetHandle(packet[7]);
            if (parent && parent->allocated) {
                const auto* session = container.GetSession(parent->session_id);
                const u64 offset = pair(8);
                const u64 address = pair(3);
                const u64 size = pair(5);
                if (session && session->process == m_process && offset <= parent->size &&
                    size <= parent->size - offset && parent->address + offset >= parent->address &&
                    address == parent->address + offset) {
                    result = BindNativeMemoryBacking(packet[2], address, size, 2, 0);
                }
            }
        }
    }
    LOG_DEBUG(Service_AM, "native GPU memory bind id={}, parent={}, address={:#x}, size={:#x}, offset={:#x}, result={:#x}",
              packet[2], packet[7], pair(3), pair(5), pair(8), result.raw);
    IPC::ResponseBuilder rb{ctx, 2};
    rb.Push(result.raw);
}

void IGraphicsSupervisor::CreateNativeImage(HLERequestContext& ctx) {
    const std::scoped_lock images_lock{m_image_registry->mutex};
    const auto* packet = ctx.CommandBuffer();
    Result result = Kernel::ResultInvalidCombination;
    u32 id{};
    if (packet[0] == 0x1bd && packet[1] == 18 && packet[3] == 1) {
        result = Kernel::ResultInvalidState;
        if (m_phase == Phase::Ready && m_gpu_context_ready && m_image_formats_initialized && m_nv_module) {
            const auto requirements = CalculateNativeImageRequirements(std::span{packet + 2, 16}, packet[18]);
            result = Kernel::ResultNotImplemented;
            if (requirements) {
                result = Kernel::ResultNotFound;
                auto& nvmap = m_nv_module->GetContainer().GetNvMapFile();
                const auto memory = nvmap.GetHandle(packet[19]);
                if (std::ranges::find(m_native_memories, packet[19]) != m_native_memories.end() &&
                    memory && memory->allocated) {
                    result = Kernel::ResultInvalidSize;
                    if (memory->size >= requirements->size) {
                        result = Kernel::ResultOutOfResource;
                        if (m_image_registry->images.size() < 1024 && m_next_native_image_id != 0 &&
                            memory->Duplicate(true) == Nvidia::NvResult::Success) {
                            NativeImage image{m_next_native_image_id++, packet[19], {}, requirements->size};
                            std::copy_n(packet + 2, image.preference.size(), image.preference.begin());
                            id = image.id;
                            if (!m_image_registry->module) {
                                m_image_registry->module = m_nv_module;
                                m_image_registry->memory_session = m_nv_module->GetContainer().OpenSession(m_process);
                            }
                            m_image_registry->images.push_back(image);
                            result = ResultSuccess;
                        }
                    }
                }
            }
        }
    }
    LOG_DEBUG(Service_AM, "native image create memory={}, count={}, id={}, result={:#x}",
              packet[19], packet[18], id, result.raw);
    IPC::ResponseBuilder rb{ctx, 3};
    rb.Push(result.raw);
    rb.Push(id);
}

void IGraphicsSupervisor::DestroyNativeImage(HLERequestContext& ctx) {
    const std::scoped_lock images_lock{m_image_registry->mutex};
    const auto* packet = ctx.CommandBuffer();
    Result result = Kernel::ResultInvalidCombination;
    if (packet[0] == 0x1be && packet[1] == 1) {
        const auto image = std::ranges::find(m_image_registry->images, packet[2], &NativeImage::id);
        result = Kernel::ResultNotFound;
        if (m_nv_module && image != m_image_registry->images.end()) {
            result = Kernel::ResultBusy;
            if (!image->window_pool) {
                m_image_registry->ReleaseMemory(image->memory_id);
                m_image_registry->images.erase(image);
                result = ResultSuccess;
            }
        }
    }
    LOG_DEBUG(Service_AM, "native image destroy id={}, result={:#x}", packet[2], result.raw);
    IPC::ResponseBuilder rb{ctx, 2};
    rb.Push(result.raw);
}

void IGraphicsSupervisor::ConfigureNativeWindowImagePool(HLERequestContext& ctx) {
    const std::scoped_lock images_lock{m_image_registry->mutex};
    const auto* packet = ctx.CommandBuffer();
    const bool create = ctx.GetCommand() == 315;
    Result result = Kernel::ResultInvalidCombination;
    if (packet[0] == (create ? 0x14bU : 0x14aU) && packet[1] == (create ? 2U : 1U)) {
        const auto image = std::ranges::find(m_image_registry->images, packet[2], &NativeImage::id);
        result = Kernel::ResultNotFound;
        if (image != m_image_registry->images.end()) {
            result = Kernel::ResultNotImplemented;
            if (!create || packet[3] == 1) {
                result = Kernel::ResultInvalidState;
                if (image->window_pool != create) {
                    image->window_pool = create;
                    result = ResultSuccess;
                }
            }
        }
    }
    LOG_DEBUG(Service_AM, "native window image pool id={}, create={}, result={:#x}",
              packet[2], create, result.raw);
    IPC::ResponseBuilder rb{ctx, 2};
    rb.Push(result.raw);
}

} // namespace Service::AM
