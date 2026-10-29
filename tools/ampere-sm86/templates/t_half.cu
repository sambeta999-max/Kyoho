// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cuda_fp16.h>
#define K(name, body) extern "C" __global__ void name(const unsigned* in, unsigned* out) { \
  unsigned t = threadIdx.x * 4; unsigned a = in[t], b = in[t + 1], c = in[t + 2]; unsigned r; body; out[t] = r; }
#define H2(x) (*reinterpret_cast<const __half2*>(&(x)))
#define U(x) (*reinterpret_cast<const unsigned*>(&(x)))
K(hadd2, { __half2 v = __hadd2(H2(a), H2(b)); r = U(v); })
K(hmul2, { __half2 v = __hmul2(H2(a), H2(b)); r = U(v); })
K(hfma2, { __half2 v = __hfma2(H2(a), H2(b), H2(c)); r = U(v); })
K(hset2, { __half2 v = __hgt2(H2(a), H2(b)); r = U(v); })
K(hmax2, { __half2 v = __hmax2(H2(a), H2(b)); r = U(v); })
K(f2fp, { __half2 v = __floats2half2_rn(__uint_as_float(a), __uint_as_float(b)); r = U(v); })
