// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

extern "C" __global__ void k(const unsigned* in, unsigned* out) {
  unsigned i = threadIdx.x; float x = __uint_as_float(in[i]); int a;
  asm volatile("cvt.rni.s32.f32 %0, %1;" : "=r"(a) : "f"(x));
  out[i] = (unsigned)a;
}
