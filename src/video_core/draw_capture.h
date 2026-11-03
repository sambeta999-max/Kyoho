// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// Per-draw capture for debugging what a frame draws:
// - EDEN_DRAW_CAPTURE=<seconds>[,<frames>] or the EDEN_SCRIPT action c<seconds> starts a capture
//   at the next frame; it lasts <frames> frames (EDEN_DRAW_FRAMES for the script action, default 1)
// - every draw, clear and dispatch of those frames goes to <out>/frame<n>.txt as one event line:
//   shader hashes, render targets, textures, storage buffers, vertex streams, indirect arguments
// - EDEN_DRAW_CAPTURE_DUMP=<event>[-<last>][,...] also writes the color render targets after those
//   events as <out>/f<n>_e<event>_rt<i>.bin (raw texels, described by the event's "dump" line);
//   an item h<hash> picks the draws with that shader hash, dumped before (_pre) and after, and
//   writes their programs as <out>/capture_<hash>_<stage>.sph/.ash (the header, and the code for
//   nvdisasm and the [corpus] test)
// - EDEN_DRAW_PROBE=<x>,<y>[;<x>,<y>...] reads those texels of color render target 0 after every
//   draw of the captured frames into a 2560x1440 target and logs the draws that change them
//   ("probe" lines; each read waits for the GPU)
// - EDEN_DRAW_SKIP=<hash>[,<hash>...] skips every draw or dispatch with one of these shader hashes,
//   captured or not
// - EDEN_DRAW_CAPTURE_OUT picks <out> (default %TEMP%/drawcap)

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "common/common_types.h"

namespace VideoCommon::DrawCapture {

class State {
public:
    State() {
        if (const char* value = std::getenv("EDEN_DRAW_CAPTURE")) {
            const std::string text{value};
            start_seconds = std::atof(text.c_str());
            if (const size_t comma = text.find(','); comma != std::string::npos) {
                frames = static_cast<u32>(std::max(1, std::atoi(text.c_str() + comma + 1)));
            }
            timed = true;
        }
        if (const char* value = std::getenv("EDEN_DRAW_CAPTURE_DUMP")) {
            const std::string text{value};
            size_t pos = 0;
            while (pos < text.size()) {
                const size_t end = std::min(text.find(',', pos), text.size());
                const std::string item{text.substr(pos, end - pos)};
                if (!item.empty() && item[0] == 'h') {
                    dump_hashes.push_back(std::strtoull(item.c_str() + 1, nullptr, 16));
                    pos = end + 1;
                    continue;
                }
                const u32 first = static_cast<u32>(std::atoi(item.c_str()));
                const size_t dash = item.find('-');
                const u32 last = dash == std::string::npos
                                     ? first
                                     : static_cast<u32>(std::atoi(item.c_str() + dash + 1));
                dump_ranges.emplace_back(first, last);
                pos = end + 1;
            }
        }
        if (const char* value = std::getenv("EDEN_DRAW_PROBE")) {
            std::string text{value};
            size_t pos = 0;
            while (pos < text.size()) {
                const size_t end = std::min(text.find(';', pos), text.size());
                const std::string item{text.substr(pos, end - pos)};
                const size_t comma = item.find(',');
                if (comma != std::string::npos) {
                    probes.push_back({static_cast<u32>(std::atoi(item.c_str())),
                                      static_cast<u32>(std::atoi(item.c_str() + comma + 1))});
                }
                pos = end + 1;
            }
        }
        if (const char* value = std::getenv("EDEN_DRAW_SKIP")) {
            std::string text{value};
            size_t pos = 0;
            while (pos < text.size()) {
                const size_t end = std::min(text.find(',', pos), text.size());
                skip_hashes.push_back(std::strtoull(text.substr(pos, end - pos).c_str(), nullptr, 16));
                pos = end + 1;
            }
        }
        if (const char* value = std::getenv("EDEN_DRAW_CAPTURE_OUT")) {
            out_dir = value;
        } else if (const char* temp = std::getenv("TEMP")) {
            out_dir = std::string{temp} + "/drawcap";
        } else {
            out_dir = "drawcap";
        }
    }

    /// Asks for a capture from the next frame on (the EDEN_SCRIPT c action)
    void Request(u32 num_frames = 1) {
        requested_frames.store(num_frames);
    }

    /// Called once per frame by the rasterizer
    void TickFrame() {
        std::scoped_lock lock{mutex};
        if (active) {
            file << fmt::format("end frame {}\n", frame);
            file.flush();
            ++frame;
            if (--remaining == 0) {
                active = false;
                file.close();
                return;
            }
            OpenFrame();
            return;
        }
        u32 wanted = requested_frames.exchange(0);
        if (wanted == 0 && timed && Elapsed() >= start_seconds) {
            timed = false;
            wanted = frames;
        }
        if (wanted == 0) {
            return;
        }
        std::error_code ec;
        std::filesystem::create_directories(out_dir, ec);
        remaining = wanted;
        active = true;
        OpenFrame();
    }

    [[nodiscard]] bool Active() const noexcept {
        return active;
    }

    /// Index of the next event of the frame
    [[nodiscard]] u32 NextEvent() noexcept {
        return event++;
    }

    [[nodiscard]] bool ShouldDump(u32 index) const noexcept {
        if (!active) {
            return false;
        }
        for (const auto& [first, last] : dump_ranges) {
            if (index >= first && index <= last) {
                return true;
            }
        }
        return false;
    }

    [[nodiscard]] bool DumpHash(u64 hash) const noexcept {
        if (!active) {
            return false;
        }
        for (const u64 dump_hash : dump_hashes) {
            if (dump_hash == hash) {
                return true;
            }
        }
        return false;
    }

    /// Whether a program was written already
    [[nodiscard]] bool FirstProgramDump(u64 hash) {
        std::scoped_lock lock{mutex};
        if (std::find(dumped_programs.begin(), dumped_programs.end(), hash) != dumped_programs.end()) {
            return false;
        }
        dumped_programs.push_back(hash);
        return true;
    }

    /// Path of a program dump without its extension
    [[nodiscard]] std::string ProgramPath(u64 hash, size_t stage) const {
        static constexpr const char* STAGES[]{"va", "vs", "tc", "te", "gs", "fs"};
        return fmt::format("{}/capture_{:016x}_{}", out_dir, hash, STAGES[stage]);
    }

    [[nodiscard]] std::string DumpPath(u32 index, u32 rt, const char* suffix) const {
        return fmt::format("{}/f{}_e{}{}_rt{}.bin", out_dir, frame, index, suffix, rt);
    }

    [[nodiscard]] bool Skip(u64 hash) const noexcept {
        for (const u64 skip : skip_hashes) {
            if (skip == hash) {
                return true;
            }
        }
        return false;
    }

    [[nodiscard]] bool HasSkips() const noexcept {
        return !skip_hashes.empty();
    }

    void Line(const std::string& text) {
        std::scoped_lock lock{mutex};
        if (active) {
            file << text << '\n';
        }
    }

    /// Bindings a pipeline's configuration collected for the current event
    std::vector<std::string> bindings;

    struct Probe {
        u32 x;
        u32 y;
        std::vector<std::pair<u64, u64>> last; ///< image, value last seen there
    };
    /// Texels watched by EDEN_DRAW_PROBE, with the image and value last seen
    std::vector<Probe> probes;

private:
    [[nodiscard]] double Elapsed() const {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
    }

    void OpenFrame() {
        event = 0;
        if (file.is_open()) {
            file.close();
        }
        file.open(fmt::format("{}/frame{}.txt", out_dir, frame));
        file << fmt::format("frame {} at {:.3f} s\n", frame, Elapsed());
    }

    std::mutex mutex;
    std::ofstream file;
    std::string out_dir;
    std::vector<u64> skip_hashes;
    std::chrono::steady_clock::time_point start_time{std::chrono::steady_clock::now()};
    std::atomic<u32> requested_frames{0};
    double start_seconds{};
    u32 frames{1};
    u32 remaining{};
    u32 frame{};
    u32 event{};
    std::vector<std::pair<u32, u32>> dump_ranges;
    std::vector<u64> dump_hashes;
    std::vector<u64> dumped_programs;
    bool timed{};
    bool active{};
};

inline State& Get() {
    static State state;
    return state;
}

} // namespace VideoCommon::DrawCapture
