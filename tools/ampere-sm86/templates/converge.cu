// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Divergent nested loops: BREAK with an explicit BRA, and BRA.CONV fast paths for warp operations
extern "C" __global__ void brk2(const int* in, int* out, int n) {
  int i = threadIdx.x, acc = 0;
  for (int j = 0; j < n; ++j) {
    for (int k = 0; k < n; ++k) {
      int v = in[(i * 7 + j * 3 + k) & 255];
      if (v == 42) goto done;
      acc += __shfl_xor_sync(0xffffffff, v, 1);
    }
    acc = __reduce_add_sync(0xffffffff, acc);
  }
done:
  out[i] = acc + __shfl_sync(0xffffffff, acc, 3);
}
