// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#define K(name, body) extern "C" __global__ void name(const unsigned* in, unsigned* out) { \
  unsigned t = threadIdx.x * 4; unsigned a = in[t], b = in[t + 1], c = in[t + 2]; unsigned r; body; out[t] = r; }
K(i2f, { float f; asm volatile("cvt.rn.f32.s32 %0, %1;" : "=f"(f) : "r"(a)); r = __float_as_uint(f); })
K(f2h, { unsigned short h; asm volatile("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(__uint_as_float(a))); r = h; })
K(prmt_r, { asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(c), "r"(b)); })
K(prmt_i, { asm volatile("prmt.b32 %0, %1, %2, 0x9a31;" : "=r"(r) : "r"(a), "r"(c)); })
K(lea, { r = (a << 5) + b; })
K(leahi, { unsigned long long p = ((unsigned long long)c << 32) | b; p += (unsigned long long)a << 3; r = (unsigned)(p >> 32); })
K(i2f_rm, { float f; asm volatile("cvt.rm.f32.u32 %0, %1;" : "=f"(f) : "r"(a)); r = __float_as_uint(f); })
K(i2f_s16, { float f; short s = (short)(a >> 16); asm volatile("cvt.rn.f32.s16 %0, %1;" : "=f"(f) : "h"(s)); r = __float_as_uint(f); })
K(i2f_s8, { float f; asm volatile("{ .reg .s8 b; cvt.s8.u32 b, %1; cvt.rn.f32.s8 %0, b; }" : "=f"(f) : "r"(a >> 16)); r = __float_as_uint(f); })
