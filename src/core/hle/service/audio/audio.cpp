// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2018 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <mutex>

#include "audio_core/audio_core.h"
#include "audio_core/sink/sink.h"
#include "audio_core/sink/sink_stream.h"
#include "core/core.h"
#include "core/memory.h"
#include "core/hle/service/kernel_helpers.h"
#include "core/hle/kernel/k_event.h"
#include "core/core_timing.h"
#include "core/hle/kernel/svc_results.h"
#include "core/hle/service/ipc_helpers.h"
#include "core/hle/service/audio/audio.h"
#include "core/hle/service/audio/audio_controller.h"
#include "core/hle/service/audio/audio_in_manager.h"
#include "core/hle/service/audio/audio_out_manager.h"
#include "core/hle/service/audio/audio_renderer_manager.h"
#include "core/hle/service/audio/final_output_recorder_manager.h"
#include "core/hle/service/audio/final_output_recorder_manager_for_applet.h"
#include "core/hle/service/audio/hardware_opus_decoder_manager.h"
#include "core/hle/service/cmif_serialization.h"
#include "core/hle/service/server_manager.h"
#include "core/hle/service/service.h"

namespace Service::Audio {

class INativeAudioManager final : public ServiceFramework<INativeAudioManager> {
public:
    INativeAudioManager(Core::System& system_, std::shared_ptr<IAudioInManager> input,
                        std::shared_ptr<IAudioOutManager> output,
                        std::shared_ptr<IAudioRendererManager> renderer)
        : ServiceFramework{system_, "aud:u"}, m_input{std::move(input)},
          m_output{std::move(output)}, m_renderer{std::move(renderer)} {
        static const FunctionInfo functions[] = {
            {0, &INativeAudioManager::GetAudioInManager, "GetAudioInManager"},
            {1, &INativeAudioManager::GetAudioOutManager, "GetAudioOutManager"},
            {2, &INativeAudioManager::GetAudioRendererManager, "GetAudioRendererManager"},
        };
        RegisterHandlers(functions);
    }

    Result HandleSyncRequest(Kernel::KServerSession& session, HLERequestContext& ctx) override {
        if (!ctx.IsTipc()) {
            return ServiceFrameworkBase::HandleSyncRequest(session, ctx);
        }
        const auto guard = LockService();
        LOG_WARNING(Service_Audio, "Native audio unsupported command={}, words={:08x}", ctx.GetCommand(),
                    fmt::join(std::span<const u32>(ctx.CommandBuffer() + 2,
                        std::min<u32>(ctx.CommandBuffer()[1], 32)), ","));
        InvokeRequestTipc(ctx);
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(Kernel::ResultNotImplemented.raw);
        return system.IsPoweredOn() ? ctx.WriteToOutgoingCommandBuffer() : ResultSuccess;
    }

private:
    void GetAudioInManager(HLERequestContext& ctx) {
        IPC::ResponseBuilder rb{ctx, 2, 0, 1};
        rb.Push(ResultSuccess);
        rb.PushIpcInterface(ctx, m_input);
    }
    void GetAudioOutManager(HLERequestContext& ctx) {
        IPC::ResponseBuilder rb{ctx, 2, 0, 1};
        rb.Push(ResultSuccess);
        rb.PushIpcInterface(ctx, m_output);
    }
    void GetAudioRendererManager(HLERequestContext& ctx) {
        LOG_DEBUG(Service_Audio, "Native user client opens existing audio renderer manager");
        IPC::ResponseBuilder rb{ctx, 2, 0, 1};
        rb.Push(ResultSuccess);
        rb.PushIpcInterface(ctx, m_renderer);
    }
    std::shared_ptr<IAudioInManager> m_input;
    std::shared_ptr<IAudioOutManager> m_output;
    std::shared_ptr<IAudioRendererManager> m_renderer;
};

// nn::audio::devmgr::IUserSession (Switch 2 audio device manager). The sdk's audio output path
// (nn::audio::GetDefaultVirtualOutputDevice, GetVirtualOutputDeviceProperties, OpenAudioRenderer)
// asks it for the virtual devices; there is one output device, "MainAudioOut" (the name
// OpenAudioRenderer checks), as functionality 1. Names are 0x100-byte NameStorage strings.
class IAudioDeviceManagerUserSession final
    : public ServiceFramework<IAudioDeviceManagerUserSession> {
public:
    explicit IAudioDeviceManagerUserSession(Core::System& system_)
        : ServiceFramework{system_, "IUserSession"} {
        static const FunctionInfo functions[] = {
            {0, D<&IAudioDeviceManagerUserSession::ListVirtualDeviceNames>, "ListVirtualDeviceNames"},
            {100, D<&IAudioDeviceManagerUserSession::GetDefaultVirtualDeviceName>, "GetDefaultVirtualDeviceName"},
            {200, D<&IAudioDeviceManagerUserSession::IsVirtualDeviceAvailable>, "IsVirtualDeviceAvailable"},
            {300, D<&IAudioDeviceManagerUserSession::IsVirtualDeviceStreamFormatSupported>, "IsVirtualDeviceStreamFormatSupported"},
            {401, D<&IAudioDeviceManagerUserSession::GetVirtualDeviceProfile>, "GetVirtualDeviceProfile"},
            {700, D<&IAudioDeviceManagerUserSession::AttachStream>, "AttachStream"},
            {800, D<&IAudioDeviceManagerUserSession::DetachStream>, "DetachStream"},
            {1000, D<&IAudioDeviceManagerUserSession::GetDeviceStreamFormat>, "GetDeviceStreamFormat"},
        };
        RegisterHandlers(functions);
    }

private:
    static constexpr u32 OutputFunctionality = 1;
    static constexpr char OutputDeviceName[] = "MainAudioOut";

    // nn::audio::StreamFormat: sample format (2 = PCM int16), channel count, sample rate
    struct StreamFormat {
        u32 sample_format;
        s32 channel_count;
        s32 sample_rate;
    };
    static_assert(sizeof(StreamFormat) == 0xc);
    static constexpr StreamFormat OutputFormat{2, 2, 48000};
    static constexpr u64 OutputLatencyNs = 5'000'000;
    static constexpr u32 OutputMixInput = 1;

    static void WriteName(std::span<u8> out) {
        std::ranges::fill(out, u8{0});
        std::memcpy(out.data(), OutputDeviceName,
                    std::min<size_t>(out.size(), sizeof(OutputDeviceName) - 1));
    }

    // A bool and the functionality in, the names out (0x100 bytes each)
    Result ListVirtualDeviceNames(Out<s32> out_count,
                                  OutBuffer<BufferAttr_HipcMapAlias> out_names, bool unknown,
                                  u32 functionality) {
        LOG_DEBUG(Service_Audio, "called, functionality={}, unknown={}", functionality, unknown);
        *out_count = 0;
        if (functionality == OutputFunctionality && out_names.size() >= 0x100) {
            WriteName(out_names.subspan(0, 0x100));
            *out_count = 1;
        }
        R_SUCCEED();
    }

    Result GetDefaultVirtualDeviceName(OutBuffer<BufferAttr_HipcMapAlias> out_name,
                                       u32 functionality) {
        LOG_DEBUG(Service_Audio, "called, functionality={}", functionality);
        WriteName(out_name);
        R_SUCCEED();
    }

    Result IsVirtualDeviceAvailable(InBuffer<BufferAttr_HipcMapAlias> name, u32 functionality) {
        LOG_DEBUG(Service_Audio, "called, functionality={}", functionality);
        R_SUCCEED();
    }

    Result IsVirtualDeviceStreamFormatSupported(InBuffer<BufferAttr_HipcMapAlias> name,
                                                StreamFormat format, u32 functionality) {
        LOG_DEBUG(Service_Audio, "called, format={}/{}/{}, functionality={}",
                  format.sample_format, format.channel_count, format.sample_rate, functionality);
        R_SUCCEED();
    }

    // The profile is 0x228 bytes, starting with the name. OutputStreamImpl::Open reads a u64 at
    // 0x218 as the device latency in nanoseconds (the stream needs at least latency * rate
    // frames per buffer) and GetVirtualOutputDeviceProperties copies it with the u32 at 0x108;
    // the u32 at 0x220 is the mixer input sysmix::CreateStream gets, which must not be 0
    Result GetVirtualDeviceProfile(InBuffer<BufferAttr_HipcMapAlias> name,
                                   OutBuffer<BufferAttr_HipcMapAlias> out_profile,
                                   u32 functionality) {
        LOG_DEBUG(Service_Audio, "called, functionality={}, size={:#x}", functionality,
                  out_profile.size());
        std::ranges::fill(out_profile, u8{0});
        if (out_profile.size() >= 0x220) {
            std::memcpy(out_profile.data(), name.data(), std::min<size_t>(name.size(), 0x100));
            std::memcpy(out_profile.data() + 0x218, &OutputLatencyNs, sizeof(u64));
            std::memcpy(out_profile.data() + 0x220, &OutputMixInput, sizeof(u32));
        }
        R_SUCCEED();
    }

    // A sysmix stream id and the functionality in, with the device name; the stream gets
    // routed to the device (output streams are mixed into the audio output here regardless)
    Result AttachStream(InBuffer<BufferAttr_HipcMapAlias> name, u32 stream_id, u32 functionality) {
        LOG_DEBUG(Service_Audio, "called, stream_id={}, functionality={}", stream_id, functionality);
        R_SUCCEED();
    }

    Result DetachStream(u32 stream_id) {
        LOG_DEBUG(Service_Audio, "called, stream_id={}", stream_id);
        R_SUCCEED();
    }

    Result GetDeviceStreamFormat(Out<StreamFormat> out_format,
                                 InBuffer<BufferAttr_HipcMapAlias> name, u32 functionality) {
        LOG_DEBUG(Service_Audio, "called, functionality={}", functionality);
        *out_format = OutputFormat;
        R_SUCCEED();
    }
};

// auddmg:u, nn::audio::devmgr::IUserSessionManager. devmgr::detail::InitializeImpl and
// CreateProxyContextImpl send command 0 with the process id, a u32 (2), the applet resource
// user id and a u64 (0) and keep the returned session; without the service they fail with 0xa99.
class IAudioDeviceManagerUserSessionManager final
    : public ServiceFramework<IAudioDeviceManagerUserSessionManager> {
public:
    explicit IAudioDeviceManagerUserSessionManager(Core::System& system_)
        : ServiceFramework{system_, "auddmg:u"} {
        static const FunctionInfo functions[] = {
            {0, D<&IAudioDeviceManagerUserSessionManager::OpenSession>, "OpenSession"},
        };
        RegisterHandlers(functions);
    }

private:
    Result OpenSession(Out<SharedPointer<IAudioDeviceManagerUserSession>> out_session, u32 kind,
                       ClientAppletResourceUserId aruid, u64 parameter) {
        LOG_WARNING(Service_Audio, "called, kind={}, aruid={:#x}, parameter={:#x}", kind,
                    aruid.pid, parameter);
        *out_session = std::make_shared<IAudioDeviceManagerUserSession>(system);
        R_SUCCEED();
    }
};

// audsmx:u, nn::audio::sysmix (Switch 2 system mixer streams), used by nn::audio::OpenOutputStream
// and the other stream types. The manager's command 0 opens an IUserSession (the client sends its
// process handle); its command 1 creates an IExternalStreamSession over client memory, 2 and 3
// engage and disengage streams. All three are domain objects.
//
// A stream's memory (sysmix::detail ring setup in the sdk): a 0x40-byte header at +0, the server's
// block at +0x40 (u32 read index at +0x44), the client's block at +0x1000 (u32 write index at
// +0x1004), then the frames at +0x1040, (buffer frames + 1) slots of channels * sample bytes. The
// server here consumes the frames at the stream's rate, plays PCM16 streams through an output
// sink stream of the audio core and signals the data event.
class ISysmixStreamSession final : public ServiceFramework<ISysmixStreamSession> {
public:
    ISysmixStreamSession(Core::System& system_, u64 memory_address_, u64 memory_size_)
        : ServiceFramework{system_, "IExternalStreamSession"},
          service_context{system_, "IExternalStreamSession"}, memory_address{memory_address_},
          memory_size{memory_size_} {
        static const FunctionInfo functions[] = {
            {0, &ISysmixStreamSession::Open, "Open"},
            {1, &ISysmixStreamSession::Close, "Close"},
            {2, &ISysmixStreamSession::ReturnZero, "Unknown2"},
            {3, &ISysmixStreamSession::ReturnZero, "Unknown3"},
            {4, &ISysmixStreamSession::Succeed, "SetGain"},
            {5, &ISysmixStreamSession::AcquireStateChangeEvent, "AcquireStateChangeEvent"},
            {6, &ISysmixStreamSession::Succeed, "ReleaseStateChangeEvent"},
            {7, &ISysmixStreamSession::AcquireDataEvent, "AcquireDataEvent"},
            {8, &ISysmixStreamSession::Succeed, "ReleaseDataEvent"},
            {9, &ISysmixStreamSession::Succeed, "Unknown9"},
            {10, &ISysmixStreamSession::ReturnZero, "Unknown10"},
            {11, &ISysmixStreamSession::ReturnZero, "Unknown11"},
            {12, &ISysmixStreamSession::Succeed, "Rename"},
        };
        RegisterHandlers(functions);
        state_event = service_context.CreateEvent("IExternalStreamSession::StateChangeEvent");
        data_event = service_context.CreateEvent("IExternalStreamSession::DataEvent");
        tick_event = Core::Timing::CreateEvent(
            "IExternalStreamSession::Tick",
            [this](s64, std::chrono::nanoseconds) -> std::optional<std::chrono::nanoseconds> {
                Tick();
                return std::nullopt;
            });
    }

    ~ISysmixStreamSession() override {
        system.CoreTiming().UnscheduleEvent(tick_event);
        CloseSink();
        service_context.CloseEvent(state_event);
        service_context.CloseEvent(data_event);
    }

private:
    static constexpr u64 ReadIndexOffset = 0x44;
    static constexpr u64 WriteIndexOffset = 0x1004;
    static constexpr u64 FramesOffset = 0x1040;
    static constexpr u32 SampleFormatPcm16 = 2;
    /// Sink buffers kept queued ahead of playback (one tick each)
    static constexpr u32 MaxQueuedBuffers = 8;
    static constexpr std::chrono::milliseconds TickPeriod{5};

    // StreamFormat (sample format, channels, rate), the buffer frame count and two option words
    void Open(HLERequestContext& ctx) {
        IPC::RequestParser rp{ctx};
        const u32 sample_format = rp.Pop<u32>();
        const s32 channels = rp.Pop<s32>();
        const s32 sample_rate = rp.Pop<s32>();
        const u32 buffer_frames = rp.Pop<u32>();
        LOG_INFO(Service_Audio, "sysmix stream open, format={}, channels={}, rate={}, frames={}",
                 sample_format, channels, sample_rate, buffer_frames);
        {
            std::scoped_lock lock{mutex};
            capacity = buffer_frames + 1;
            frames_per_tick = std::max<u32>(1, static_cast<u32>(sample_rate) / 200);
            frame_channels = static_cast<u32>(std::clamp(channels, 1, 6));
            pcm16 = sample_format == SampleFormatPcm16;
            open = true;
            CloseSinkLocked();
            // The audio core's sinks run at 48 kHz: other rates are consumed but not played
            if (pcm16 && sample_rate == 48'000) {
                sink = &system.AudioCore().GetOutputSink();
                sink_stream = sink->AcquireSinkStream(system, frame_channels, "SysmixStream",
                                                      ::AudioCore::Sink::StreamType::Out);
                if (sink_stream) {
                    sink_stream->SetRingSize(MaxQueuedBuffers);
                    sink_stream->Start();
                }
            }
        }
        system.CoreTiming().UnscheduleEvent(tick_event);
        system.CoreTiming().ScheduleLoopingEvent(TickPeriod, TickPeriod, tick_event);
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(ResultSuccess);
    }

    void Close(HLERequestContext& ctx) {
        system.CoreTiming().UnscheduleEvent(tick_event);
        {
            std::scoped_lock lock{mutex};
            open = false;
            CloseSinkLocked();
        }
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(ResultSuccess);
    }

    // The notification mode and a frame threshold
    void AcquireDataEvent(HLERequestContext& ctx) {
        IPC::RequestParser rp{ctx};
        const u32 mode = rp.Pop<u32>();
        const u32 threshold = rp.Pop<u32>();
        LOG_DEBUG(Service_Audio, "called, mode={}, threshold={}", mode, threshold);
        {
            std::scoped_lock lock{mutex};
            data_threshold = threshold;
        }
        IPC::ResponseBuilder rb{ctx, 2, 1};
        rb.Push(ResultSuccess);
        rb.PushCopyObjects(ctx, data_event->GetReadableEvent());
    }

    void AcquireStateChangeEvent(HLERequestContext& ctx) {
        IPC::ResponseBuilder rb{ctx, 2, 1};
        rb.Push(ResultSuccess);
        rb.PushCopyObjects(ctx, state_event->GetReadableEvent());
    }

    void ReturnZero(HLERequestContext& ctx) {
        LOG_DEBUG(Service_Audio, "(STUBBED) called, command={}", ctx.GetCommand());
        IPC::ResponseBuilder rb{ctx, 3};
        rb.Push(ResultSuccess);
        rb.Push<u32>(0);
    }

    void Succeed(HLERequestContext& ctx) {
        LOG_DEBUG(Service_Audio, "(STUBBED) called, command={}", ctx.GetCommand());
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(ResultSuccess);
    }

    // Consumes up to one tick of frames between the read and write indices
    void Tick() {
        std::scoped_lock lock{mutex};
        if (!open || capacity == 0 || memory_size < WriteIndexOffset + 4) {
            return;
        }
        auto& memory = system.ApplicationMemory();
        const u32 read = memory.Read32(memory_address + ReadIndexOffset);
        const u32 write = memory.Read32(memory_address + WriteIndexOffset);
        if (read >= capacity || write >= capacity) {
            return;
        }
        const u32 filled = write >= read ? write - read : write + capacity - read;
        u32 consumed = std::min(filled, frames_per_tick);
        // With a sink, consume only what it can queue: playback paces the stream
        if (sink_stream && sink_stream->GetQueueSize() >= MaxQueuedBuffers) {
            consumed = 0;
        }
        if (consumed != 0) {
            if (sink_stream && FramesOffset + u64{capacity} * frame_channels * 2 <= memory_size) {
                PlayFrames(read, consumed);
            }
            memory.Write32(memory_address + ReadIndexOffset, (read + consumed) % capacity);
        }
        const u32 free_frames = capacity - 1 - (filled - consumed);
        if (consumed != 0 || free_frames >= data_threshold) {
            data_event->Signal(system.Kernel());
        }
    }

    // Copies frames [read, read + count) of the ring to the sink stream
    void PlayFrames(u32 read, u32 count) {
        auto& memory = system.ApplicationMemory();
        samples.resize(size_t{count} * frame_channels);
        const u32 first = std::min(count, capacity - read);
        memory.ReadBlock(memory_address + FramesOffset + u64{read} * frame_channels * 2,
                         samples.data(), size_t{first} * frame_channels * 2);
        if (first < count) {
            memory.ReadBlock(memory_address + FramesOffset, samples.data() + size_t{first} * frame_channels,
                             size_t{count - first} * frame_channels * 2);
        }
        ::AudioCore::Sink::SinkBuffer buffer{
            .frames = count,
            .frames_played = 0,
            .tag = ++buffer_tag,
            .consumed = false,
        };
        if (const char* path = std::getenv("EDEN_SYSMIX_DUMP")) {
            // Debugging: the raw PCM16 frames each stream played, appended to <path>
            if (std::FILE* file = std::fopen(path, "ab")) {
                std::fwrite(samples.data(), sizeof(s16), samples.size(), file);
                std::fclose(file);
            }
        }
        sink_stream->AppendBuffer(buffer, samples);
    }

    void CloseSink() {
        std::scoped_lock lock{mutex};
        CloseSinkLocked();
    }

    void CloseSinkLocked() {
        if (sink_stream) {
            sink_stream->Stop();
            sink->CloseStream(sink_stream);
            sink_stream = nullptr;
        }
    }

    KernelHelpers::ServiceContext service_context;
    Kernel::KEvent* state_event{};
    Kernel::KEvent* data_event{};
    std::shared_ptr<Core::Timing::EventType> tick_event;
    std::mutex mutex;
    u64 memory_address;
    u64 memory_size;
    u32 capacity{};
    u32 frames_per_tick{};
    u32 frame_channels{2};
    u32 data_threshold{};
    bool pcm16{};
    bool open{};
    ::AudioCore::Sink::Sink* sink{};
    ::AudioCore::Sink::SinkStream* sink_stream{};
    std::vector<s16> samples;
    u64 buffer_tag{};
};

class ISysmixUserSession final : public ServiceFramework<ISysmixUserSession> {
public:
    explicit ISysmixUserSession(Core::System& system_)
        : ServiceFramework{system_, "ISysmixUserSession"} {
        static const FunctionInfo functions[] = {
            {1, &ISysmixUserSession::CreateStream, "CreateStream"},
            {2, &ISysmixUserSession::Succeed, "EngageStreams"},
            {3, &ISysmixUserSession::Succeed, "DisengageStreams"},
        };
        RegisterHandlers(functions);
    }

private:
    // The process id, the stream name (pointer buffer), a u32 kind, the option flags, the applet
    // resource user id and the stream memory (address and size, in the client process)
    void CreateStream(HLERequestContext& ctx) {
        IPC::RequestParser rp{ctx};
        const u32 kind = rp.Pop<u32>();
        const u32 flags = rp.Pop<u32>();
        const u64 aruid = rp.Pop<u64>();
        const u64 address = rp.Pop<u64>();
        const u64 size = rp.Pop<u64>();
        LOG_INFO(Service_Audio,
                 "sysmix create stream, kind={}, flags={:#x}, aruid={:#x}, memory={:#x}+{:#x}",
                 kind, flags, aruid, address, size);
        IPC::ResponseBuilder rb{ctx, 4, 0, 1};
        rb.Push(ResultSuccess);
        rb.Push<u32>(next_stream_id++);
        rb.Push<u32>(0);
        rb.PushIpcInterface(ctx, std::make_shared<ISysmixStreamSession>(system, address, size));
    }

    void Succeed(HLERequestContext& ctx) {
        LOG_DEBUG(Service_Audio, "(STUBBED) called, command={}", ctx.GetCommand());
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(ResultSuccess);
    }

    u32 next_stream_id{1};
};

class ISysmixUserSessionManager final : public ServiceFramework<ISysmixUserSessionManager> {
public:
    explicit ISysmixUserSessionManager(Core::System& system_)
        : ServiceFramework{system_, "audsmx:u"} {
        static const FunctionInfo functions[] = {
            {0, &ISysmixUserSessionManager::OpenSession, "OpenSession"},
        };
        RegisterHandlers(functions);
    }

private:
    // The process id and the client's process handle
    void OpenSession(HLERequestContext& ctx) {
        LOG_DEBUG(Service_Audio, "called");
        IPC::ResponseBuilder rb{ctx, 2, 0, 1};
        rb.Push(ResultSuccess);
        rb.PushIpcInterface(ctx, std::make_shared<ISysmixUserSession>(system));
    }
};

class IAudioOutManagerForApplet final : public ServiceFramework<IAudioOutManagerForApplet> {
public:
    explicit IAudioOutManagerForApplet(Core::System& system_)
        : ServiceFramework{system_, "audout:a"} {
        // clang-format off
        static const FunctionInfo functions[] = {
            {0, nullptr, "RequestSuspend"},
            {1, nullptr, "RequestResume"},
            {2, nullptr, "GetProcessMasterVolume"},
            {3, nullptr, "SetProcessMasterVolume"},
            {4, nullptr, "GetProcessRecordVolume"},
            {5, nullptr, "SetProcessRecordVolume"},
        };
        // clang-format on
        RegisterHandlers(functions);
    }
};

class IAudioSnoopManager final : public ServiceFramework<IAudioSnoopManager> {
public:
    explicit IAudioSnoopManager(Core::System& system_)
        : ServiceFramework{system_, "auddev"} {
        // clang-format off
        static const FunctionInfo functions[] = {
            {0, nullptr, "GetDspStatistics"},
            {1, nullptr, "GetAppletStateSummaries"},
            {2, nullptr, "SetDspStatisticsParameter"},
            {3, nullptr, "GetDspStatisticsParameter"},
            {6, nullptr, "GetDspUsage"},
        };
        // clang-format on
        RegisterHandlers(functions);
    }
};

class IAudioInManagerForApplet final : public ServiceFramework<IAudioInManagerForApplet> {
public:
    explicit IAudioInManagerForApplet(Core::System& system_)
        : ServiceFramework{system_, "audin:a"} {
        // clang-format off
        static const FunctionInfo functions[] = {
            {0, nullptr, "RequestSuspend"},
            {1, nullptr, "RequestResume"},
            {2, nullptr, "GetProcessMasterVolume"},
            {3, nullptr, "SetProcessMasterVolume"},
        };
        // clang-format on
        RegisterHandlers(functions);
    }
};

class IAudioRendererManagerForApplet final : public ServiceFramework<IAudioRendererManagerForApplet> {
public:
    explicit IAudioRendererManagerForApplet(Core::System& system_)
        : ServiceFramework{system_, "audren:a"} {
        // clang-format off
        static const FunctionInfo functions[] = {
            {0, nullptr, "RequestSuspend"},
            {1, nullptr, "RequestResume"},
            {2, nullptr, "GetProcessMasterVolume"},
            {3, nullptr, "SetProcessMasterVolume"},
            {4, nullptr, "RegisterAppletResourceUserId"},
            {5, nullptr, "UnregisterAppletResourceUserId"},
            {6, nullptr, "GetProcessRecordVolume"},
            {7, nullptr, "SetProcessRecordVolume"},
        };
        // clang-format on
        RegisterHandlers(functions);
    }
};

class IAudioOutManagerForDebugger final : public ServiceFramework<IAudioOutManagerForDebugger> {
public:
    explicit IAudioOutManagerForDebugger(Core::System& system_)
        : ServiceFramework{system_, "audout:d"} {
        // clang-format off
        static const FunctionInfo functions[] = {
            {0, nullptr, "RequestSuspend"},
            {1, nullptr, "RequestResume"},
        };
        // clang-format on
        RegisterHandlers(functions);
    }
};

class IAudioInManagerForDebugger final : public ServiceFramework<IAudioInManagerForDebugger> {
public:
    explicit IAudioInManagerForDebugger(Core::System& system_)
        : ServiceFramework{system_, "audin:d"} {
        // clang-format off
        static const FunctionInfo functions[] = {
            {0, nullptr, "RequestSuspend"},
            {1, nullptr, "RequestResume"},
        };
        // clang-format on
        RegisterHandlers(functions);
    }
};

class IFinalOutputRecorderManagerForDebugger final : public ServiceFramework<IFinalOutputRecorderManagerForDebugger> {
public:
    explicit IFinalOutputRecorderManagerForDebugger(Core::System& system_)
        : ServiceFramework{system_, "audrec:d"} {
        // clang-format off
        static const FunctionInfo functions[] = {
            {0, nullptr, "RequestSuspend"},
            {1, nullptr, "RequestResume"},
        };
        // clang-format on
        RegisterHandlers(functions);
    }
};

class IAudioRendererManagerForDebugger final : public ServiceFramework<IAudioRendererManagerForDebugger> {
public:
    explicit IAudioRendererManagerForDebugger(Core::System& system_)
        : ServiceFramework{system_, "audren:d"} {
        // clang-format off
        static const FunctionInfo functions[] = {
            {0, nullptr, "RequestSuspend"},
            {1, nullptr, "RequestResume"},
        };
        // clang-format on
        RegisterHandlers(functions);
    }
};

class IAudioSystemManagerForApplet final : public ServiceFramework<IAudioSystemManagerForApplet> {
public:
    explicit IAudioSystemManagerForApplet(Core::System& system_)
        : ServiceFramework{system_, "aud:a"} {
        // clang-format off
        static const FunctionInfo functions[] = {
            {0, nullptr, "RegisterAppletResourceUserId"},
            {1, nullptr, "UnregisterAppletResourceUserId"},
            {2, nullptr, "RequestSuspendAudio"},
            {3, nullptr, "RequestResumeAudio"},
            {4, nullptr, "GetAudioOutputProcessMasterVolume"},
            {5, nullptr, "SetAudioOutputProcessMasterVolume"},
            {6, nullptr, "GetAudioInputProcessMasterVolume"},
            {7, nullptr, "SetAudioInputProcessMasterVolume"},
            {8, nullptr, "GetAudioOutputProcessRecordVolume"},
            {9, nullptr, "SetAudioOutputProcessRecordVolume"},
            {10, nullptr, "GetAppletStateSummaries"},
        };
        // clang-format on
        RegisterHandlers(functions);
    }
};

class IAudioSystemManagerForDebugger final : public ServiceFramework<IAudioSystemManagerForDebugger> {
public:
    explicit IAudioSystemManagerForDebugger(Core::System& system_)
        : ServiceFramework{system_, "aud:d"} {
        // clang-format off
        static const FunctionInfo functions[] = {
            {0, nullptr, "RequestSuspendAudioForDebug"},
            {1, nullptr, "RequestResumeAudioForDebug"},
        };
        // clang-format on
        RegisterHandlers(functions);
    }
};

void LoopProcess(Core::System& system) {
    auto server_manager = std::make_unique<ServerManager>(system);

    server_manager->RegisterNamedService("aud:a", std::make_shared<IAudioSystemManagerForApplet>(system), 30);
    server_manager->RegisterNamedService("aud:d", std::make_shared<IAudioSystemManagerForDebugger>(system), 30);

    server_manager->RegisterNamedService("audout:d", std::make_shared<IAudioOutManagerForDebugger>(system), 30);
    server_manager->RegisterNamedService("audin:d", std::make_shared<IAudioInManagerForDebugger>(system), 30);
    server_manager->RegisterNamedService("audrec:d", std::make_shared<IFinalOutputRecorderManagerForDebugger>(system), 30);
    server_manager->RegisterNamedService("audren:d", std::make_shared<IAudioRendererManagerForDebugger>(system), 30);

    auto audio_input = std::make_shared<IAudioInManager>(system);
    server_manager->RegisterNamedService("audin:u", audio_input, 30);
    server_manager->RegisterNamedService("audin:a", std::make_shared<IAudioInManagerForApplet>(system), 30);
    auto audio_output = std::make_shared<IAudioOutManager>(system);
    server_manager->RegisterNamedService("audout:u", audio_output, 30);
    server_manager->RegisterNamedService("audout:a", std::make_shared<IAudioOutManagerForApplet>(system), 30);
    server_manager->RegisterNamedService("auddev", std::make_shared<IAudioSnoopManager>(system), 30);
    // Depends on audout:u and audin:u on ctor!
    server_manager->RegisterNamedService("audctl", std::make_shared<IAudioController>(system), 30);
    server_manager->RegisterNamedService("audrec:a", std::make_shared<IFinalOutputRecorderManagerForApplet>(system), 30);
    server_manager->RegisterNamedService("audrec:u", std::make_shared<IFinalOutputRecorderManager>(system), 30);
    auto audio_renderer = std::make_shared<IAudioRendererManager>(system);
    server_manager->RegisterNamedService("audren:u", audio_renderer, 30);
    server_manager->RegisterNamedService("audren:a", std::make_shared<IAudioRendererManagerForApplet>(system), 30);
    server_manager->RegisterNamedService("auddmg:u", std::make_shared<IAudioDeviceManagerUserSessionManager>(system), 30);
    server_manager->RegisterNamedService("audsmx:u", std::make_shared<ISysmixUserSessionManager>(system), 30);
    server_manager->RegisterNamedService("hwopus", std::make_shared<IHardwareOpusDecoderManager>(system), 25);
    server_manager->RegisterNamedService("aud:u",
        std::make_shared<INativeAudioManager>(system, audio_input, audio_output, audio_renderer), 30);
    ServerManager::RunServer(std::move(server_manager));
}

} // namespace Service::Audio
