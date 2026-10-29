#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later

"""Second pass: probe the missing forms of every base that has a valid form.

Usage: python3 sweep_forms.py <workdir> <sweep.json> [extra_bases_hex,...]
Updates sweep.json in place.
"""
import json, os, random, re, struct, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor

WORK, PATH = sys.argv[1], sys.argv[2]
EXTRA = [int(x, 16) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else []
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
    if tok[0].startswith("@"):
        tok = tok[1:]
    return tok[0].split(".")[0]

sweep = json.load(open(PATH))
valid = {int(k, 16) for k in sweep}
bases = {op & 0x1ff for op in valid} | {b & 0x1ff for b in EXTRA}
candidates = [(f << 9) | b for b in bases for f in range(8) if ((f << 9) | b) not in valid]

def probe(op):
    rng = random.Random(op * 7919)
    base = op & 0x1ff
    trials = []
    for f in range(8):
        sib = (f << 9) | base
        for s in sweep.get(f"{sib:03x}", {}).values():
            for smp in s:
                lo = (int(smp["lo"], 16) & ~0xfff) | op
                trials.append((lo, int(smp["hi"], 16)))
                trials.append((lo, int(smp["hi"], 16) ^ (1 << 27)))  # toggle bit 91
    for _ in range(16):
        lo = 0x7000 | op
        hi = CTRL
        for bit in range(16, 105):
            if rng.random() < 0.08:
                if bit < 64:
                    lo |= 1 << bit
                else:
                    hi |= 1 << (bit - 64)
        trials.append((lo, hi))
    for _ in range(8):
        trials.append(((rng.getrandbits(48) << 16) | 0x7000 | op,
                       (CTRL & ~((1 << 41) - 1)) | rng.getrandbits(41)))
    for lo, hi in trials:
        text = disasm(lo, hi)
        if text is not None:
            return op, {mnemonic(text): [{"lo": f"{lo:016x}", "hi": f"{hi:016x}", "text": text}]}
    return op, None

with ThreadPoolExecutor(max_workers=os.cpu_count() or 8) as pool:
    found = {op: seen for op, seen in pool.map(probe, candidates) if seen}

for op, seen in found.items():
    sweep[f"{op:03x}"] = seen
json.dump(sweep, open(PATH, "w"), indent=1)
print(f"candidates {len(candidates)}, newly valid {len(found)}: {sorted(hex(o) for o in found)}")
