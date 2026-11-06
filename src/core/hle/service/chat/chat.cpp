// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "core/hle/kernel/k_event.h"
#include "core/hle/kernel/k_readable_event.h"
#include "core/hle/service/chat/chat.h"
#include "core/hle/service/cmif_serialization.h"
#include "core/hle/service/kernel_helpers.h"
#include "core/hle/service/server_manager.h"
#include "core/hle/service/service.h"

namespace Service::Chat {

// nn::chat::detail::IUserShimScopedObject. GameChat is never active in the emulator, so the
// chatting-state event is never signaled.
class IUserShimScopedObject final : public ServiceFramework<IUserShimScopedObject> {
public:
    explicit IUserShimScopedObject(Core::System& system_)
        : ServiceFramework{system_, "IUserShimScopedObject"}, service_context{system_, "chat"} {
        event = service_context.CreateEvent("chat:ChattingEvent");
        static const FunctionInfo functions[] = {
            {1, D<&IUserShimScopedObject::GetChattingEvent>, "GetChattingEvent"},
        };
        RegisterHandlers(functions);
    }

    ~IUserShimScopedObject() override {
        service_context.CloseEvent(event);
    }

private:
    Result GetChattingEvent(OutCopyHandle<Kernel::KReadableEvent> out_event) {
        *out_event = std::addressof(event->GetReadableEvent());
        R_SUCCEED();
    }

    KernelHelpers::ServiceContext service_context;
    Kernel::KEvent* event{};
};

// nn::chat::detail::IVirtualCameraShimScopedObject. Its commands 0-5 take no data and return
// only a result; without GameChat they do nothing.
class IVirtualCameraShimScopedObject final
    : public ServiceFramework<IVirtualCameraShimScopedObject> {
public:
    explicit IVirtualCameraShimScopedObject(Core::System& system_)
        : ServiceFramework{system_, "IVirtualCameraShimScopedObject"} {
        static const FunctionInfo functions[] = {
            {0, D<&IVirtualCameraShimScopedObject::NoOp>, "Unknown0"},
            {1, D<&IVirtualCameraShimScopedObject::NoOp>, "Unknown1"},
            {2, D<&IVirtualCameraShimScopedObject::NoOp>, "Unknown2"},
            {3, D<&IVirtualCameraShimScopedObject::NoOp>, "Unknown3"},
            {4, D<&IVirtualCameraShimScopedObject::NoOp>, "Unknown4"},
            {5, D<&IVirtualCameraShimScopedObject::NoOp>, "Unknown5"},
        };
        RegisterHandlers(functions);
    }

private:
    Result NoOp() {
        LOG_WARNING(Service, "(STUBBED) called");
        R_SUCCEED();
    }
};

// nn::chat::detail::IUserService, the GameChat interface of Switch 2 applications.
class IUserService final : public ServiceFramework<IUserService> {
public:
    explicit IUserService(Core::System& system_) : ServiceFramework{system_, "chat:u"} {
        // Commands whose proxies (sdk 0.x) take no data or one u32 and return only a result;
        // GameChat never runs here, so they succeed without doing anything. Unknown commands
        // would get no response and the client aborts with 2162-0005 (0x1a80a).
        static const FunctionInfo functions[] = {
            {0, D<&IUserService::CreateUserShimScopedObject>, "CreateUserShimScopedObject"},
            {3, D<&IUserService::CreateVirtualCameraShimScopedObject>,
             "CreateVirtualCameraShimScopedObject"},
            {11, D<&IUserService::GetChattingState>, "GetChattingState"},
            {22,D<&IUserService::NoOp>, "Unknown22"},
            {23, D<&IUserService::SetU32>, "Unknown23"},
            {61, D<&IUserService::NoOp>, "Unknown61"},
        };
        RegisterHandlers(functions);
    }

private:
    Result NoOp() {
        LOG_WARNING(Service, "(STUBBED) called");
        R_SUCCEED();
    }

    // nn::chat::GetChattingUid (sdk 0x6be460) passes a 0xf0-byte fixed-size pointer buffer and
    // takes the Uid at offset 0x18; Star Fox asks every frame. Nobody is chatting.
    Result GetChattingState(OutLargeData<std::array<u8, 0xf0>, BufferAttr_HipcPointer> out_state) {
        out_state->fill(0);
        R_SUCCEED();
    }

    // A u32 and a u64 (Nintendo Switch Sports Resort sends 1 and 0x3031) in, the object out
    Result CreateVirtualCameraShimScopedObject(
        Out<SharedPointer<IVirtualCameraShimScopedObject>> out_object, u32 kind, u64 parameter) {
        LOG_WARNING(Service, "(STUBBED) called, kind={}, parameter={:#x}", kind, parameter);
        *out_object = std::make_shared<IVirtualCameraShimScopedObject>(system);
        R_SUCCEED();
    }

    Result SetU32(u32 value) {
        LOG_WARNING(Service, "(STUBBED) called, value={:#x}", value);
        R_SUCCEED();
    }

    Result CreateUserShimScopedObject(OutInterface<IUserShimScopedObject> out_object) {
        *out_object = std::make_shared<IUserShimScopedObject>(system);
        R_SUCCEED();
    }
};

void LoopProcess(Core::System& system) {
    auto server_manager = std::make_unique<ServerManager>(system);
    server_manager->RegisterNamedService("chat:u", std::make_shared<IUserService>(system));
    ServerManager::RunServer(std::move(server_manager));
}

} // namespace Service::Chat
