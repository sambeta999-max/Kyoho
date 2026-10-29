#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later

"""Sweep SM86 12-bit opcode space with nvdisasm and record mnemonics.

Usage: python3 sweep_opcodes.py <workdir> <out.json> [trials]
"""
import json, os, random, re, struct, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor

WORK, OUT = sys.argv[1], sys.argv[2]
TRIALS = int(sys.argv[3]) if len(sys.argv) > 3 else 8
CTRL = 0x000fc00000000000
LINE = re.compile(r"/\*0000\*/\s+(.*?)\s*;")

def disasm(lo, hi):
    fd, path = tempfile.mkstemp(suffix=".bin", dir=WORK)
    with os.fdopen(fd, "wb") as f:
        f.write(struct.pack("<QQ", lo, hi))
    try:
        r = subprocess.run(["nvdisasm", "-b", "SM86", path], capture_output=True, text=True)
    finally:
        os.remove(path)
    m = LINE.search(r.stdout)
    return m.group(1) if m else None

def mnemonic(text):
    tok = text.split()
    if tok and tok[0].startswith("@"):
        tok = tok[1:]
    return tok[0] if tok else ""

def probe(op):
    rng = random.Random(op)
    samples = [(0x7000 | op, CTRL)]
    for _ in range(TRIALS):
        lo = (rng.getrandbits(48) << 16) | 0x7000 | op
        hi = (CTRL & ~((1 << 41) - 1)) | rng.getrandbits(41)
        samples.append((lo, hi))
    seen = {}
    for lo, hi in samples:
        text = disasm(lo, hi)
        if text is not None:
            seen.setdefault(mnemonic(text).split(".")[0], []).append(
                {"lo": f"{lo:016x}", "hi": f"{hi:016x}", "text": text})
    return op, seen

with ThreadPoolExecutor(max_workers=os.cpu_count() or 8) as pool:
    results = dict(pool.map(probe, range(0x1000)))

table = {f"{op:03x}": seen for op, seen in results.items() if seen}
with open(OUT, "w") as f:
    json.dump(table, f, indent=1)
print(f"valid opcodes: {len(table)}")
print(f"multi-mnemonic: {[k for k, v in table.items() if len(v) > 1]}")
