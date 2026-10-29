#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later

"""Generate SM86 instruction test vectors by running the exact encodings on the local GPU.

Usage: python3 gen_hw_vectors.py <hw_dir> <hwrun.py> <out.inc>
Every case patches one instruction of a CUDA template whose register assignment is known, runs it
with chosen inputs and records the result written to R7.
"""
import os, random, struct, subprocess, sys

HW, HWRUN, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
ORACLE = next(f"{HW}/{n}" for n in ("oracle.exe", "oracle") if os.path.exists(f"{HW}/{n}"))

def f32(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]

rng = random.Random(86)
FLOATS = [0x7fc00000, 0xffc00000, 0x7f800000, 0xff800000, 0x00000000, 0x80000000, 0x00000001,
          0x807fffff, f32(-2.5), f32(2.5), f32(0.5), f32(-0.5), f32(1.5), f32(65504.0),
          f32(65520.0), f32(1e-5), f32(-1e-8), f32(3e10), f32(-3e10), f32(255.5), f32(-128.5),
          f32(32767.5), f32(-40000.0), f32(4294967296.0)]
FLOATS += [f32(rng.uniform(-70000, 70000)) for _ in range(8)]
INTS = [0, 1, 0xffffffff, 0x7fffffff, 0x80000000, 0x01000001, 0x00ffffff, 0xff7f8001,
        0x12345678, 0x80008000, 0x7fff7fff]
INTS += [rng.getrandbits(32) for _ in range(9)]
HALVES = [0x3c00, 0xbc00, 0x7c00, 0xfc00, 0x7e00, 0x0001, 0x8000, 0x7bff, 0x3555]
HALF_PAIRS = [(a << 16) | b for a, b in zip(HALVES, reversed(HALVES))]
HALF_PAIRS += [rng.getrandbits(32) for _ in range(6)]

F2I = ("t_f2i.cubin", "k", 1, 0x0000000200077305, 0x004e280000203100)
I2FP = ("t_ops.cubin", "i2f", 4, 0x0000000200077245, 0x004fca0000201400)
F2F = ("t_ops.cubin", "f2h", 4, 0x0000000200077304, 0x004e280000200800)
PRMT_R = ("t_ops.cubin", "prmt_r", 4, 0x0000000700077216, 0x004fca0000000006)
PRMT_I = ("t_ops.cubin", "prmt_i", 4, 0x00009a3100077816, 0x004fca0000000007)
LEA = ("t_ops.cubin", "lea", 4, 0x0000000700077211, 0x004fca00078e28ff)

def field(value, offset, count, x):
    mask = ((1 << count) - 1) << offset
    return (value & ~mask) | ((x << offset) & mask)

def f2i_hi(size, signed, ntz, rnd):
    hi = F2I[4]
    hi = field(hi, 11, 2, size)
    hi = field(hi, 8, 1, signed)
    hi = field(hi, 13, 1, ntz)
    return field(hi, 14, 2, rnd)

cases = []
for size, bits in ((0, 8), (1, 16), (2, 32)):
    for signed in (0, 1):
        for ntz in (0, 1):
            for rnd, rname in ((0, "rn"), (1, "floor"), (2, "ceil"), (3, "trunc")):
                if bits != 32 and rnd not in (0, 3):
                    continue
                name = f"F2I.{'S' if signed else 'U'}{bits}.{rname}{'.NTZ' if ntz else ''}"
                cases.append((name, F2I, F2I[3], f2i_hi(size, signed, ntz, rnd), [2], FLOATS))

# Variable latency I2F needs the scoreboard bits of a compiler-generated I2F
I2F = ("t_ops.cubin", "i2f_rm", 4, 0x0000000200077306, 0x004e280000205000)
i2f_s32 = field(field(I2F[4], 10, 1, 1), 14, 2, 0)
cases += [
    ("I2FP.F32.S32", I2FP, I2FP[3], I2FP[4], [2], INTS),
    ("I2FP.F32.U32", I2FP, I2FP[3], field(I2FP[4], 10, 1, 0), [2], INTS),
    ("I2F.F32.S32", I2F, I2F[3], i2f_s32, [2], INTS),
    ("I2F.F32.U32.RM", I2F, I2F[3], I2F[4], [2], INTS),
    ("I2F.F32.S8.B2", I2F, field(I2F[3], 60, 2, 2), field(i2f_s32, 20, 2, 0), [2], INTS),
    ("I2F.F32.U16.H1", I2F, field(I2F[3], 60, 2, 1), field(field(I2F[4], 20, 2, 1), 14, 2, 0),
     [2], INTS),
    ("F2F.F16.F32", F2F, F2F[3], F2F[4], [2], FLOATS),
    ("F2F.F16.F32.RZ", F2F, F2F[3], field(F2F[4], 14, 2, 3), [2], FLOATS),
    ("F2F.F32.F16", F2F, F2F[3], field(field(F2F[4], 11, 2, 2), 20, 2, 1), [2], HALF_PAIRS),
    ("F2F.F32.F16.H1", F2F, field(F2F[3], 60, 1, 1), field(field(F2F[4], 11, 2, 2), 20, 2, 1),
     [2], HALF_PAIRS),
    ("PRMT", PRMT_R, PRMT_R[3], PRMT_R[4], [0, 7, 6], None),
    ("PRMT.IMM", PRMT_I, PRMT_I[3], PRMT_I[4], [0, None, 7], None),
    ("LEA", LEA, LEA[3], LEA[4], [0, 7], None),
    ("LEA.31", LEA, LEA[3], field(LEA[4], 11, 5, 31), [0, 7], None),
    ("LEA.HI", LEA, LEA[3], field(field(LEA[4], 16, 1, 1), 0, 8, 0), [0, 7], None),
    ("LEA.HI.7", LEA, LEA[3], field(field(field(LEA[4], 16, 1, 1), 0, 8, 0), 11, 5, 7), [0, 7],
     None),
]

HALF = ("t_half.cubin", "hadd2", 4, 0x0000000700077230, 0x004fca0000000000)
HMUL2 = ("t_half.cubin", "hmul2", 4, 0x0000000700077232, 0x004fca0000000000)
HFMA2 = ("t_half.cubin", "hfma2", 4, 0x0000000700077231, 0x004fca0000000006)
HSET2 = ("t_half.cubin", "hset2", 4, 0x0000000700077233, 0x004fca0003804080)
HMNMX2 = ("t_half.cubin", "hmax2", 4, 0x0000000700077240, 0x004fca0007800000)
F2FP = ("t_half.cubin", "f2fp", 4, 0x000000000707723e, 0x004fca00000000ff)

SPECIAL_HALVES = [0x0000, 0x8000, 0x0001, 0x03ff, 0x0400, 0x3c00, 0xbc00, 0x3555, 0x7bff,
                  0xfbff, 0x7c00, 0xfc00, 0x7e00, 0x3800, 0xc500, 0x5640]


def half_inputs():
    rows = []
    for _ in range(24):
        row = []
        for _ in range(3):
            lo = rng.choice(SPECIAL_HALVES) if rng.random() < 0.4 else rng.getrandbits(15) & 0x7bff | (rng.getrandbits(1) << 15)
            hi = rng.choice(SPECIAL_HALVES) if rng.random() < 0.4 else rng.getrandbits(15) & 0x7bff | (rng.getrandbits(1) << 15)
            row.append((hi << 16) | lo)
        rows.append(row)
    return rows


cases += [
    ("HADD2", HALF, HALF[3], HALF[4], [0, 7, 6], "half"),
    ("HADD2.NEG_B", HALF, HALF[3] | (1 << 63), HALF[4], [0, 7, 6], "half"),
    ("HADD2.A_H1_H1", HALF, HALF[3], field(HALF[4], 10, 2, 3), [0, 7, 6], "half"),
    ("HADD2.SAT", HALF, HALF[3], field(HALF[4], 13, 1, 1), [0, 7, 6], "half"),
    ("HMUL2", HMUL2, HMUL2[3], HMUL2[4], [0, 7, 6], "half"),
    ("HMUL2.B_H0_H0", HMUL2, field(HMUL2[3], 60, 2, 2), HMUL2[4], [0, 7, 6], "half"),
    ("HFMA2", HFMA2, HFMA2[3], HFMA2[4], [0, 7, 6], "half"),
    ("HFMA2.NEG_C", HFMA2, HFMA2[3], field(HFMA2[4], 20, 1, 1), [0, 7, 6], "half"),
    ("HFMA2.SAT", HFMA2, HFMA2[3], field(HFMA2[4], 13, 1, 1), [0, 7, 6], "half"),
    ("HSET2.BF.GT", HSET2, HSET2[3], HSET2[4], [0, 7, 6], "half"),
    ("HSET2.GT", HSET2, HSET2[3], field(HSET2[4], 7, 1, 0), [0, 7, 6], "half"),
    ("HSET2.BF.LE", HSET2, HSET2[3], field(HSET2[4], 12, 4, 3), [0, 7, 6], "half"),
    ("HSET2.BF.NEU", HSET2, HSET2[3], field(HSET2[4], 12, 4, 0xd), [0, 7, 6], "half"),
    ("HMNMX2.MAX", HMNMX2, HMNMX2[3], HMNMX2[4], [0, 7, 6], "half"),
    ("HMNMX2.MIN", HMNMX2, HMNMX2[3], field(HMNMX2[4], 26, 1, 0), [0, 7, 6], "half"),
    ("F2FP.PACK_AB", F2FP, F2FP[3], F2FP[4], [0, 7], FLOATS),
    ("F2FP.PACK_AB.RZ", F2FP, F2FP[3], field(F2FP[4], 15, 2, 3), [0, 7], FLOATS),
]


def int_inputs(count):
    rows = [[rng.getrandbits(32) for _ in range(count)] for _ in range(12)]
    rows += [[0x80ff7f01, 0x0000d3a8, 0x12345678], [0xffffffff, 0xfedcba98, 0], [1, 0x7531, 2]]
    return rows

out = open(OUT, "w", newline="\n")
for name, (cubin, kernel, words, find_lo, find_hi), lo, hi, regs, values in cases:
    if values is None:
        rows = int_inputs(len(regs))
    elif values == "half":
        rows = half_inputs()
    elif len(regs) == 2:
        rows = [[a, b] for a, b in zip(values, values[7:] + values[:7])]
    else:
        rows = [[v] for v in values]
    threads = len(rows)
    flat = []
    for row in rows:
        padded = list(row) + [0] * (words - len(row))
        flat += [f"{v:08x}" for v in padded[:words]]
    r = subprocess.run([sys.executable, "-I", HWRUN, ORACLE, f"{HW}/{cubin}", f"{find_lo:016x}",
                        f"{find_hi:016x}", f"{lo:016x}", f"{hi:016x}", kernel, str(threads),
                        str(words), *flat], capture_output=True, text=True, check=True)
    results = [line.split()[0] for line in r.stdout.splitlines()]
    assert len(results) == threads, (name, r.stdout, r.stderr)
    for row, result in zip(rows, results):
        inputs = ", ".join(f"{{{reg}, 0x{v:08x}}}" for reg, v in zip(regs, row) if reg is not None)
        out.write(f'{{"{name}", 0x{lo:016x}ULL, 0x{hi:016x}ULL, {{{{{inputs}}}}}, 0x{result}}},\n')
    print(f"{name}: {threads} vectors")
