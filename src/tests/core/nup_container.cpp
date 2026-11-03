// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <openssl/evp.h>
#include <zstd.h>

#include "core/file_sys/nup_container.h"
#include "core/file_sys/vfs/vfs_vector.h"

namespace {

using FileSys::NupContainer;

template <typename T>
void Append(std::vector<u8>& bytes, T value) {
    const auto* raw = reinterpret_cast<const u8*>(&value);
    bytes.insert(bytes.end(), raw, raw + sizeof(value));
}

std::array<u8, 0x20> Sha256(const std::vector<u8>& data) {
    std::array<u8, 0x20> out{};
    unsigned int length = 0;
    EVP_Digest(data.data(), data.size(), out.data(), &length, EVP_sha256(), nullptr);
    return out;
}

std::vector<u8> Bytes(std::string_view text) {
    return {text.begin(), text.end()};
}

/// A package tree for the builder: paths below the package root ("romfs/a/b.bin")
struct TestPackage {
    u64 title_id;
    u32 version;
    u8 kind;
    std::map<std::string, std::pair<std::vector<u8>, bool>> files; // path -> (data, zstd)
};

/// Writes a NUP v1 container as SPEC.md lays it out
std::vector<u8> BuildNup(const std::vector<TestPackage>& packages) {
    struct Dir {
        std::string name;
        std::map<std::string, u32> dirs;
        std::map<std::string, u32> files;
    };
    struct File {
        std::string name;
        std::vector<u8> data;
        bool zstd;
    };
    std::vector<Dir> dirs;
    std::vector<File> files;
    struct Root {
        u32 root;
        u32 exefs;
        u32 romfs;
    };
    std::vector<Root> roots;
    for (const auto& package : packages) {
        const u32 root = static_cast<u32>(dirs.size());
        dirs.push_back({""});
        for (const auto& [path, content] : package.files) {
            u32 dir = root;
            std::size_t start = 0;
            for (std::size_t slash; (slash = path.find('/', start)) != std::string::npos;
                 start = slash + 1) {
                const std::string part = path.substr(start, slash - start);
                if (!dirs[dir].dirs.contains(part)) {
                    dirs[dir].dirs[part] = static_cast<u32>(dirs.size());
                    dirs.push_back({part});
                }
                dir = dirs[dir].dirs[part];
            }
            const std::string name = path.substr(start);
            dirs[dir].files[name] = static_cast<u32>(files.size());
            files.push_back({name, content.first, content.second});
        }
        const auto find = [&](const char* name) {
            const auto it = dirs[root].dirs.find(name);
            return it == dirs[root].dirs.end() ? 0xFFFFFFFFU : it->second;
        };
        roots.push_back({root, find("exefs"), find("romfs")});
    }

    std::vector<u8> pool;
    const auto intern = [&](const std::string& s) {
        const u32 offset = static_cast<u32>(pool.size());
        pool.insert(pool.end(), s.begin(), s.end());
        return std::pair<u32, u32>{offset, static_cast<u32>(s.size())};
    };
    std::vector<u32> children;
    std::vector<u32> first_child;
    for (const auto& dir : dirs) {
        first_child.push_back(static_cast<u32>(children.size()));
        // Children sorted by name, directories and files together
        std::map<std::string, u32> sorted;
        for (const auto& [name, index] : dir.dirs) {
            sorted[name] = index;
        }
        for (const auto& [name, index] : dir.files) {
            sorted[name] = static_cast<u32>(dirs.size()) + index;
        }
        for (const auto& [name, node] : sorted) {
            children.push_back(node);
        }
    }

    const u64 toc_head = 0x30 + 0x40 * packages.size() + 0x18 * dirs.size() + 0x50 * files.size();
    const u64 pool_offset = toc_head + 4 * children.size();
    std::vector<std::pair<u32, u32>> dir_names;
    for (const auto& dir : dirs) {
        dir_names.push_back(intern(dir.name));
    }
    std::vector<std::pair<u32, u32>> file_names;
    for (const auto& file : files) {
        file_names.push_back(intern(file.name));
    }
    const u64 toc_size = (pool_offset + pool.size() + 15) & ~u64{15};
    const u64 blob_offset = (0x80 + toc_size + 15) & ~u64{15};

    std::vector<u8> blob;
    struct Extent {
        u64 offset, size;
        u8 codec;
    };
    std::vector<Extent> extents;
    for (const auto& file : files) {
        if (file.data.empty()) {
            extents.push_back({0, 0, 0});
            continue;
        }
        std::vector<u8> stored = file.data;
        u8 codec = 0;
        if (file.zstd) {
            stored.resize(ZSTD_compressBound(file.data.size()));
            stored.resize(ZSTD_compress(stored.data(), stored.size(), file.data.data(),
                                        file.data.size(), 3));
            codec = 1;
        }
        blob.resize((blob.size() + 15) & ~std::size_t{15});
        extents.push_back({blob_offset + blob.size(), stored.size(), codec});
        blob.insert(blob.end(), stored.begin(), stored.end());
    }
    blob.resize((blob.size() + 15) & ~std::size_t{15});

    std::vector<u8> toc;
    Append<u32>(toc, 0x01434F54);
    Append<u32>(toc, static_cast<u32>(packages.size()));
    Append<u64>(toc, dirs.size());
    Append<u64>(toc, files.size());
    Append<u64>(toc, pool_offset);
    Append<u64>(toc, pool.size());
    Append<u64>(toc, 0);
    for (std::size_t i = 0; i < packages.size(); ++i) {
        Append<u64>(toc, packages[i].title_id);
        Append<u32>(toc, roots[i].root);
        Append<u32>(toc, roots[i].exefs);
        Append<u32>(toc, roots[i].romfs);
        Append<u32>(toc, packages[i].version);
        Append<u8>(toc, packages[i].kind);
        toc.resize(toc.size() + 0x40 - 0x19, 0);
    }
    for (std::size_t i = 0; i < dirs.size(); ++i) {
        Append<u32>(toc, dir_names[i].first);
        Append<u32>(toc, dir_names[i].second);
        Append<u32>(toc, first_child[i]);
        Append<u32>(toc, static_cast<u32>(dirs[i].dirs.size() + dirs[i].files.size()));
        Append<u64>(toc, 0);
    }
    for (std::size_t i = 0; i < files.size(); ++i) {
        Append<u32>(toc, file_names[i].first);
        Append<u32>(toc, file_names[i].second);
        Append<u64>(toc, files[i].data.size());
        Append<u64>(toc, extents[i].offset);
        Append<u64>(toc, extents[i].size);
        const auto hash = Sha256(files[i].data);
        toc.insert(toc.end(), hash.begin(), hash.end());
        Append<u8>(toc, extents[i].codec);
        toc.resize(toc.size() + 0x0F, 0);
    }
    for (const u32 child : children) {
        Append<u32>(toc, child);
    }
    REQUIRE(toc.size() == pool_offset);
    toc.insert(toc.end(), pool.begin(), pool.end());
    toc.resize(toc_size, 0);

    std::vector<u8> out;
    const char magic[4] = {'N', 'U', 'P', 1};
    out.insert(out.end(), magic, magic + 4);
    Append<u16>(out, 0x80);
    Append<u16>(out, 1);
    Append<u64>(out, 0x80);
    Append<u64>(out, toc_size);
    const auto toc_hash = Sha256(toc);
    out.insert(out.end(), toc_hash.begin(), toc_hash.end());
    Append<u64>(out, blob_offset);
    Append<u64>(out, blob_offset + blob.size());
    Append<u32>(out, static_cast<u32>(packages.size()));
    Append<u32>(out, 0);
    out.resize(0x80, 0);
    out.insert(out.end(), toc.begin(), toc.end());
    out.resize(blob_offset, 0);
    out.insert(out.end(), blob.begin(), blob.end());
    return out;
}

FileSys::VirtualFile AsFile(std::vector<u8> bytes) {
    return std::make_shared<FileSys::VectorVfsFile>(std::move(bytes), "test.nup");
}

std::string ReadText(const FileSys::VirtualFile& file) {
    REQUIRE(file != nullptr);
    const auto bytes = file->ReadAllBytes();
    return {bytes.begin(), bytes.end()};
}

std::vector<TestPackage> BaseAndUpdate() {
    const std::string big(100000, 'z');
    return {
        {0x0400000000001000, 0, 0,
         {{"exefs/main", {Bytes("main-base"), false}},
          {"romfs/a.bin", {Bytes("base a " + big), true}},
          {"romfs/sub/b.bin", {Bytes("base b"), false}},
          {"romfs/empty", {{}, false}}}},
        {0x0400000000001800, 0x10000, 1,
         {{"exefs/main", {Bytes("main-update"), true}},
          {"romfs/a.bin", {Bytes("update a"), true}},
          {"romfs/c.bin", {Bytes("update c"), false}}}},
    };
}

} // Anonymous namespace

TEST_CASE("NUP: packages, raw and zstd members", "[file_sys]") {
    NupContainer nup{AsFile(BuildNup(BaseAndUpdate()))};
    REQUIRE(nup.GetStatus() == Loader::ResultStatus::Success);
    REQUIRE(nup.FindBasePackage() == std::size_t{0});
    REQUIRE(nup.FindBestUpdatePackage() == std::size_t{1});

    REQUIRE(ReadText(nup.GetExeFSDir(0)->GetFile("main")) == "main-base");
    REQUIRE(ReadText(nup.GetExeFSDir(1)->GetFile("main")) == "main-update");

    const auto base_romfs = nup.GetRomFSDir(0);
    const auto a = base_romfs->GetFile("a.bin");
    REQUIRE(a->GetSize() == std::size_t{7 + 100000});
    // A read in the middle of a zstd member
    std::array<u8, 4> middle{};
    REQUIRE(a->Read(middle.data(), middle.size(), 50000) == middle.size());
    REQUIRE(middle == std::array<u8, 4>{'z', 'z', 'z', 'z'});
    REQUIRE(ReadText(base_romfs->GetSubdirectory("sub")->GetFile("b.bin")) == "base b");
    REQUIRE(base_romfs->GetFile("empty")->GetSize() == 0);

    REQUIRE(nup.VerifyIntegrity({}) == Loader::ResultStatus::Success);
}

TEST_CASE("NUP: the update shadows the base", "[file_sys]") {
    NupContainer nup{AsFile(BuildNup(BaseAndUpdate()))};
    REQUIRE(nup.GetStatus() == Loader::ResultStatus::Success);
    const auto merged = NupContainer::MergeShadowDirs(nup.GetRomFSDir(0), nup.GetRomFSDir(1));
    REQUIRE(ReadText(merged->GetFile("a.bin")) == "update a");
    REQUIRE(ReadText(merged->GetFile("c.bin")) == "update c");
    REQUIRE(ReadText(merged->GetSubdirectory("sub")->GetFile("b.bin")) == "base b");
}

TEST_CASE("NUP: a modified table of contents is rejected", "[file_sys]") {
    auto bytes = BuildNup(BaseAndUpdate());
    bytes[0x80 + 0x30] ^= 1; // First byte of the first package's title ID
    NupContainer nup{AsFile(std::move(bytes))};
    REQUIRE(nup.GetStatus() == Loader::ResultStatus::ErrorNUPBadFormat);
}

TEST_CASE("NUP: a modified member fails the integrity check", "[file_sys]") {
    auto bytes = BuildNup(BaseAndUpdate());
    // "base b" is stored raw: flip a byte of it in the blob
    const std::string needle = "base b";
    const auto it = std::search(bytes.begin(), bytes.end(), needle.begin(), needle.end());
    REQUIRE(it != bytes.end());
    *it ^= 1;
    NupContainer nup{AsFile(std::move(bytes))};
    REQUIRE(nup.GetStatus() == Loader::ResultStatus::Success);
    REQUIRE(nup.VerifyIntegrity({}) == Loader::ResultStatus::ErrorIntegrityVerificationFailed);
}
