// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "common/host_sampler.h"

#if defined(_WIN32) && defined(_M_X64)

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <windows.h>

#include <dbghelp.h>

#include <timeapi.h>

#pragma comment(lib, "dbghelp.lib")
#pragma comment(lib, "winmm.lib")

namespace Common::HostSampler {
namespace {

// Each registered thread is suspended about every millisecond, its instruction pointer and a few
// callers (from the x64 unwind tables) are recorded, and every 5 s host-samples-<name>.txt in the
// working directory is rewritten with the functions by exclusive (where the thread was) and
// inclusive (anywhere on the recorded stack) sample counts.
constexpr int Depth = 12;

struct Target {
    std::string name;
    HANDLE handle{};
    std::map<std::array<DWORD64, Depth>, unsigned long long> stacks;
    unsigned long long samples{};
};

std::mutex targets_mutex;
std::vector<Target*> targets;
std::once_flag start_flag;

std::array<DWORD64, Depth> Unwind(CONTEXT context) {
    std::array<DWORD64, Depth> frames{};
    for (int i = 0; i < Depth && context.Rip != 0; ++i) {
        frames[i] = context.Rip;
        DWORD64 image_base{};
        PRUNTIME_FUNCTION function = RtlLookupFunctionEntry(context.Rip, &image_base, nullptr);
        if (!function) {
            // A leaf function: the return address is at the top of the stack
            context.Rip = *reinterpret_cast<DWORD64*>(context.Rsp);
            context.Rsp += 8;
            continue;
        }
        PVOID handler_data{};
        DWORD64 establisher{};
        RtlVirtualUnwind(UNW_FLAG_NHANDLER, image_base, context.Rip, function, &context,
                         &handler_data, &establisher, nullptr);
    }
    return frames;
}

std::string Symbol(DWORD64 address, std::map<DWORD64, std::string>& cache) {
    if (const auto it = cache.find(address); it != cache.end()) {
        return it->second;
    }
    alignas(SYMBOL_INFO) char buffer[sizeof(SYMBOL_INFO) + 512]{};
    auto* const info = reinterpret_cast<SYMBOL_INFO*>(buffer);
    info->SizeOfStruct = sizeof(SYMBOL_INFO);
    info->MaxNameLen = 511;
    DWORD64 displacement{};
    std::string name = SymFromAddr(GetCurrentProcess(), address, &displacement, info)
                           ? std::string{info->Name}
                           : "?";
    cache.emplace(address, name);
    return name;
}

void Report() {
    static bool symbols = false;
    if (!symbols) {
        SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS);
        symbols = SymInitialize(GetCurrentProcess(), nullptr, TRUE);
    }
    static std::map<DWORD64, std::string> cache;
    std::scoped_lock lock{targets_mutex};
    for (Target* const target : targets) {
        std::map<std::string, unsigned long long> exclusive;
        std::map<std::string, unsigned long long> inclusive;
        // EDEN_HOST_SAMPLER_FOCUS=<part of a function name>: also count the callers above it
        static const char* const focus = std::getenv("EDEN_HOST_SAMPLER_FOCUS");
        std::map<std::string, unsigned long long> callers;
        for (const auto& [frames, count] : target->stacks) {
            std::vector<std::string> seen;
            for (int i = 0; focus && i < Depth && frames[i] != 0; ++i) {
                if (Symbol(frames[i], cache).find(focus) == std::string::npos) {
                    continue;
                }
                std::string chain;
                for (int j = i + 1; j < Depth && j <= i + 4 && frames[j] != 0; ++j) {
                    chain += (j == i + 1 ? "" : " <- ") + Symbol(frames[j], cache);
                }
                callers[chain] += count;
                break;
            }
            for (int i = 0; i < Depth && frames[i] != 0; ++i) {
                std::string name = Symbol(frames[i], cache);
                if (i == 0) {
                    exclusive[name] += count;
                }
                if (std::find(seen.begin(), seen.end(), name) == seen.end()) {
                    inclusive[name] += count;
                    seen.push_back(std::move(name));
                }
            }
        }
        const auto sorted = [](const std::map<std::string, unsigned long long>& map) {
            std::vector<std::pair<unsigned long long, std::string>> list;
            for (const auto& [name, count] : map) {
                list.emplace_back(count, name);
            }
            std::sort(list.rbegin(), list.rend());
            return list;
        };
        const std::string path = "host-samples-" + target->name + ".txt";
        if (std::FILE* file = std::fopen(path.c_str(), "w")) {
            const double total = static_cast<double>((std::max)(target->samples, 1ULL));
            std::fprintf(file, "%llu samples\n\nexclusive\n", target->samples);
            int lines = 0;
            for (const auto& [count, name] : sorted(exclusive)) {
                if (lines++ == 80) break;
                std::fprintf(file, "%6.2f%% %s\n", 100.0 * count / total, name.c_str());
            }
            if (focus) {
                std::fprintf(file, "\ncallers of %s\n", focus);
                int callers_lines = 0;
                for (const auto& [count, name] : sorted(callers)) {
                    if (callers_lines++ == 25) break;
                    std::fprintf(file, "%6.2f%% %s\n", 100.0 * count / total, name.c_str());
                }
            }
            std::fprintf(file, "\ninclusive\n");
            lines = 0;
            for (const auto& [count, name] : sorted(inclusive)) {
                if (lines++ == 150) break;
                std::fprintf(file, "%6.2f%% %s\n", 100.0 * count / total, name.c_str());
            }
            std::fclose(file);
        }
    }
}

void Run() {
    timeBeginPeriod(1);
    auto next_report = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    for (;;) {
        {
            std::scoped_lock lock{targets_mutex};
            for (Target* const target : targets) {
                if (SuspendThread(target->handle) == static_cast<DWORD>(-1)) {
                    continue;
                }
                CONTEXT context{};
                context.ContextFlags = CONTEXT_FULL;
                if (GetThreadContext(target->handle, &context)) {
                    ++target->stacks[Unwind(context)];
                    ++target->samples;
                }
                ResumeThread(target->handle);
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        if (std::chrono::steady_clock::now() >= next_report) {
            Report();
            next_report += std::chrono::seconds(5);
        }
    }
}

} // Anonymous namespace

void RegisterCurrentThread(const char* name) {
    const char* const list = std::getenv("EDEN_HOST_SAMPLER");
    if (!list) {
        return;
    }
    const std::string names = std::string{","} + list + ",";
    if (names.find(std::string{","} + name + ",") == std::string::npos) {
        return;
    }
    HANDLE handle{};
    DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &handle,
                    THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE,
                    0);
    {
        std::scoped_lock lock{targets_mutex};
        targets.push_back(new Target{.name = name, .handle = handle});
    }
    std::call_once(start_flag, [] { std::thread{Run}.detach(); });
}

} // namespace Common::HostSampler

#else

namespace Common::HostSampler {
void RegisterCurrentThread(const char*) {}
} // namespace Common::HostSampler

#endif
