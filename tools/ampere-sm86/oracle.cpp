// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Runs a kernel "name(const unsigned* in, unsigned* out)" from a cubin on the local GPU.
// Usage: oracle <cubin> <kernel> <threads> <words_per_thread> <hex inputs...>
// Each thread reads words_per_thread inputs and writes words_per_thread outputs.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda.h>

#define CHECK(x)                                                                                   \
    do {                                                                                           \
        CUresult r_ = (x);                                                                         \
        if (r_ != CUDA_SUCCESS) {                                                                  \
            const char* s_;                                                                        \
            cuGetErrorString(r_, &s_);                                                             \
            std::fprintf(stderr, "%s failed: %s\n", #x, s_);                                       \
            return 1;                                                                              \
        }                                                                                          \
    } while (0)

int main(int argc, char** argv) {
    if (argc < 5) {
        return 2;
    }
    const unsigned threads = std::strtoul(argv[3], nullptr, 0);
    const unsigned words = std::strtoul(argv[4], nullptr, 0);
    std::vector<unsigned> in(threads * words), out(threads * words, 0xcdcdcdcd);
    for (int i = 5; i < argc && i - 5 < static_cast<int>(in.size()); ++i) {
        in[i - 5] = std::strtoul(argv[i], nullptr, 16);
    }
    CUdevice dev;
    CUcontext ctx;
    CUmodule mod;
    CUfunction fn;
    CUdeviceptr din, dout;
    CHECK(cuInit(0));
    CHECK(cuDeviceGet(&dev, 0));
    CHECK(cuCtxCreate(&ctx, 0, dev));
    CHECK(cuModuleLoad(&mod, argv[1]));
    CHECK(cuModuleGetFunction(&fn, mod, argv[2]));
    CHECK(cuMemAlloc(&din, in.size() * 4));
    CHECK(cuMemAlloc(&dout, out.size() * 4));
    CHECK(cuMemcpyHtoD(din, in.data(), in.size() * 4));
    CHECK(cuMemcpyHtoD(dout, out.data(), out.size() * 4));
    void* params[] = {&din, &dout};
    CHECK(cuLaunchKernel(fn, 1, 1, 1, threads, 1, 1, 0, nullptr, params, nullptr));
    CHECK(cuCtxSynchronize());
    CHECK(cuMemcpyDtoH(out.data(), dout, out.size() * 4));
    for (unsigned t = 0; t < threads; ++t) {
        for (unsigned w = 0; w < words; ++w) {
            std::printf("%08x%c", out[t * words + w], w + 1 == words ? '\n' : ' ');
        }
    }
    return 0;
}
