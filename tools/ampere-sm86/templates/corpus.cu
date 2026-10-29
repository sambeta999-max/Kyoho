// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Kernels compiled for sm_86 to collect real compiler output for the decoder tests
#include <cuda_fp16.h>
#include <cuda_bf16.h>
extern "C" __global__ void arith(float* o, const float* a, const double* d, double* od, int* io, const int* ia, unsigned long long* u64, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = a[i], y = a[i ^ 1];
  o[i] = __fmaf_rn(x, y, 1.5f) + sinf(x) * cosf(y) + __expf(x) + rsqrtf(fabsf(y)) + fminf(x, y) + floorf(x) + truncf(y) + __saturatef(x);
  od[i] = d[i] * d[i ^ 3] + sqrt(d[i]) + floor(d[i]) + (double)ia[i] + (double)(long long)u64[i];
  int v = ia[i];
  io[i] = __popc(v) + __clz(v) + __brev(v) + (v >> 3) + ((unsigned)v >> 5) + __funnelshift_r(v, ia[i+1], i) + __byte_perm(v, i, 0x5410) + max(v, i) + __mulhi(v, i) + (int)x + (int)(unsigned)y + __vabsdiffu4(v, i);
  u64[i] = u64[i] * 3 + __umul64hi(u64[i], 7ull) + (u64[i] << (i & 63));
  io[n + i] = (v & i) | (~v ^ (i >> 2)) | ((v | i) & 0xf0f0);
}
extern "C" __global__ void halfk(__half2* h, __nv_bfloat162* b, int n) {
  int i = threadIdx.x;
  __half2 x = h[i], y = h[i ^ 1];
  h[i] = __hfma2(x, y, __hmax2(x, y)) + __hmul2(x, y);
  if (__hbgt2(x, y)) h[i + n] = __h2div(x, y);
  b[i] = __hfma2(b[i], b[i ^ 1], b[i]);
}
extern "C" __global__ void warp(int* o, const int* a, unsigned* cnt, unsigned long long* u) {
  __shared__ int s[256];
  int i = threadIdx.x;
  s[i] = a[i];
  __syncthreads();
  int v = s[255 - i];
  v += __shfl_sync(0xffffffff, v, (i + 1) & 31) + __shfl_xor_sync(0xffffffff, v, 4) + __shfl_down_sync(0xffffffff, v, 2);
  v += __ballot_sync(0xffffffff, v > 3) + __any_sync(0xffffffff, v == 0) + __reduce_add_sync(0xffffffff, v) + __match_any_sync(0xffffffff, v);
  atomicAdd(cnt, 1u); atomicMax(&s[i & 7], v); atomicCAS(cnt + 1, 0u, (unsigned)v); atomicExch(u, (unsigned long long)v);
  atomicAdd((float*)(cnt + 2), 1.0f); atomicAdd((double*)(u + 1), 2.0);
  __threadfence(); __syncwarp();
  o[i] = v + s[i ^ 1];
  if (v == 12345) asm volatile("trap;");
}
extern "C" __global__ void texk(cudaTextureObject_t t, cudaTextureObject_t t3, cudaSurfaceObject_t sf, float4* o, float u, float v) {
  int i = threadIdx.x;
  float4 a = tex2D<float4>(t, u + i, v);
  float4 b = tex2DLod<float4>(t, u, v, 1.5f);
  float4 c = tex2DGrad<float4>(t, u, v, make_float2(1, 0), make_float2(0, 1));
  float4 d = tex2Dgather<float4>(t, u, v, 1);
  float4 e = tex3D<float4>(t3, u, v, 0.5f);
  float4 f = tex1Dfetch<float4>(t, i);
  float4 g = texCubemap<float4>(t3, u, v, 1.0f);
  float4 h = tex2DLayered<float4>(t, u, v, 2);
  float4 r; surf2Dread(&r, sf, i * 16, 3);
  surf2Dwrite(make_float4(a.x + b.y + c.z + d.w + e.x + f.y + g.z + h.w + r.x, 0, 0, 0), sf, i * 16, 2);
  o[i] = a;
}
