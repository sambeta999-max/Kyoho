// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Offline coverage check of the SM86 frontend: translates every program dumped with
// dump_guest_shaders (<program>_<hash>_<stage>.ash, plus .sph headers for graphics stages) and
// counts what is not implemented. Hidden test; run with
//   EDEN_SHADER_CORPUS=<dump dir>[;<dump dir>...] tests.exe "[corpus]"
// EDEN_CORPUS_OUT=<file> also writes the summary there.

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <regex>
#include <set>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include "common/common_types.h"
#include "shader_recompiler/environment.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ampere/translate/translate.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/object_pool.h"

namespace {

using namespace Shader;

constexpr u32 AMPERE_HEADER_SIZE{0x80};

class CorpusEnvironment final : public Environment {
public:
    CorpusEnvironment(std::vector<u64> words, Stage stage_, const std::vector<u8>& header)
        : code{std::move(words)} {
        stage = stage_;
        is_ampere = true;
        std::memcpy(&sph, header.data(), std::min(header.size(), sizeof(sph)));
    }

    u64 ReadInstruction(u32 address) override {
        const size_t index{address / 8};
        return index < code.size() ? code[index] : 0;
    }
    u32 ReadCbufValue(u32, u32) override {
        return 0;
    }
    TextureType ReadTextureType(u32) override {
        return TextureType::Color2D;
    }
    TexturePixelFormat ReadTexturePixelFormat(u32) override {
        return {};
    }
    bool IsTexturePixelFormatInteger(u32) override {
        return false;
    }
    u32 ReadViewportTransformState() override {
        return 0;
    }
    u32 TextureBoundBuffer() const override {
        return 0;
    }
    u32 LocalMemorySize() const override {
        return 0x1000;
    }
    u32 SharedMemorySize() const override {
        return 0xc000;
    }
    std::array<u32, 3> WorkgroupSize() const override {
        return {32, 1, 1};
    }
    bool HasHLEMacroState() const override {
        return false;
    }
    std::optional<ReplaceConstant> GetReplaceConstBuffer(u32, u32) override {
        return std::nullopt;
    }
    void Dump(u64, u64) override {}

private:
    std::vector<u64> code;
};

std::optional<Stage> StageFromSuffix(const std::string& suffix) {
    static const std::map<std::string, Stage> stages{
        {"vs", Stage::VertexB},  {"va", Stage::VertexA},
        {"tc", Stage::TessellationControl}, {"te", Stage::TessellationEval},
        {"gs", Stage::Geometry}, {"fs", Stage::Fragment},
        {"cs", Stage::Compute},
    };
    const auto it{stages.find(suffix)};
    return it == stages.end() ? std::nullopt : std::optional{it->second};
}

std::vector<u8> ReadFile(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

// Addresses and offsets in messages would split one gap into many entries
std::string Normalize(std::string message) {
    static const std::regex address{R"( at 0x[0-9a-fA-F]+)"};
    static const std::regex words{R"(Invalid instruction [0-9a-f]+ [0-9a-f]+)"};
    message = std::regex_replace(message, address, "");
    return std::regex_replace(message, words, "Invalid instruction");
}

struct Gap {
    size_t count{};
    std::set<std::string> programs;
};

std::map<std::string, Gap>* g_gaps{};
const std::string* g_program{};

void Observe(const char* message) {
    Gap& gap{(*g_gaps)[Normalize(message)]};
    ++gap.count;
    gap.programs.insert(*g_program);
}

} // Anonymous namespace

TEST_CASE("Ampere corpus coverage", "[.][corpus]") {
    const char* const dirs{std::getenv("EDEN_SHADER_CORPUS")};
    REQUIRE(dirs != nullptr);
    std::vector<std::filesystem::path> files;
    std::string list{dirs};
    for (size_t begin = 0; begin <= list.size();) {
        const size_t end{std::min(list.find(';', begin), list.size())};
        const std::filesystem::path dir{list.substr(begin, end - begin)};
        if (std::filesystem::is_directory(dir)) {
            for (const auto& entry : std::filesystem::directory_iterator(dir)) {
                if (entry.path().extension() == ".ash") {
                    files.push_back(entry.path());
                }
            }
        }
        begin = end + 1;
    }
    std::ranges::sort(files);

    std::map<std::string, Gap> gaps;
    size_t clean{};
    size_t rejected{};
    size_t partial{};
    size_t maxwell{};
    g_gaps = &gaps;
    Ampere::unimplemented_observer = Observe;
    for (const auto& path : files) {
        const std::string name{path.stem().string()};
        const auto stage{StageFromSuffix(name.substr(name.rfind('_') + 1))};
        if (!stage) {
            continue;
        }
        const bool graphics{*stage != Stage::Compute};
        std::vector<u8> header;
        if (graphics) {
            auto sph_path{path};
            header = ReadFile(sph_path.replace_extension(".sph"));
            if (!header.empty() && header.size() < AMPERE_HEADER_SIZE) {
                // A 0x50-byte header is a Maxwell program, which the Maxwell frontend handles
                ++maxwell;
                continue;
            }
            header.resize(AMPERE_HEADER_SIZE);
        }
        std::vector<u8> bytes{header};
        const std::vector<u8> body{ReadFile(path)};
        bytes.insert(bytes.end(), body.begin(), body.end());
        bytes.resize((bytes.size() + 15) / 16 * 16);
        std::vector<u64> words(bytes.size() / 8);
        std::memcpy(words.data(), bytes.data(), bytes.size());

        g_program = &name;
        try {
            CorpusEnvironment env{std::move(words), *stage, header};
            ObjectPool<IR::Inst> inst_pool;
            ObjectPool<IR::Block> block_pool;
            ObjectPool<Maxwell::Flow::Block> flow_pool;
            Maxwell::Flow::CFG cfg{env, flow_pool, graphics ? AMPERE_HEADER_SIZE : 0u,
                                   *stage == Stage::VertexA};
            HostTranslateInfo host_info{};
            host_info.support_float64 = true;
            host_info.support_float16 = true;
            host_info.support_int64 = true;
            host_info.support_conditional_barrier = true;
            [[maybe_unused]] const IR::Program program{
                Maxwell::TranslateProgram(inst_pool, block_pool, env, cfg, host_info)};
            // EDEN_CORPUS_IR=<dir> writes each program's IR there as <program>.ir
            if (const char* ir_dir{std::getenv("EDEN_CORPUS_IR")}) {
                std::ofstream{std::filesystem::path{ir_dir} / (name + ".ir")}
                    << IR::DumpProgram(program);
            }
            const bool skipped{std::ranges::any_of(gaps, [&](const auto& item) {
                return item.second.programs.contains(name);
            })};
            (skipped ? partial : clean) += 1;
        } catch (const std::exception& exception) {
            Observe((std::string{"[rejected] "} + exception.what()).c_str());
            if (std::getenv("EDEN_CORPUS_VERBOSE")) {
                fmt::print("{}: {}\n", name, exception.what());
            }
            ++rejected;
        }
    }
    Ampere::unimplemented_observer = nullptr;

    std::vector<std::pair<std::string, const Gap*>> sorted;
    for (const auto& [message, gap] : gaps) {
        sorted.emplace_back(message, &gap);
    }
    std::ranges::sort(sorted, [](const auto& lhs, const auto& rhs) {
        return lhs.second->programs.size() != rhs.second->programs.size()
                   ? lhs.second->programs.size() > rhs.second->programs.size()
                   : lhs.second->count > rhs.second->count;
    });
    std::string report{fmt::format(
        "{} SM86 programs: {} clean, {} with skipped instructions, {} rejected ({} Maxwell "
        "programs not checked)\n"
        "programs  occurrences  gap\n",
        clean + partial + rejected, clean, partial, rejected, maxwell)};
    for (const auto& [message, gap] : sorted) {
        std::string examples;
        for (const std::string& program : gap->programs) {
            if (examples.size() > 120) {
                break;
            }
            examples += ' ' + program;
        }
        report += fmt::format("{:8}  {:11}  {}\n          e.g.{}\n", gap->programs.size(),
                              gap->count, message, examples);
    }
    fmt::print("{}", report);
    if (const char* const out{std::getenv("EDEN_CORPUS_OUT")}) {
        std::ofstream{out} << report;
    }
}
