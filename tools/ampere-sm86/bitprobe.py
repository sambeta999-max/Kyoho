#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later

"""Flip single bits of a base SM86 encoding and show how nvdisasm's text changes.

Usage: python3 bitprobe.py <workdir> <lo_hex> <hi_hex> [first_bit] [last_bit]
"""
import os, re, struct, subprocess, sys, tempfile

WORK = sys.argv[1]
LO, HI = int(sys.argv[2], 16), int(sys.argv[3], 16)
FIRST = int(sys.argv[4]) if len(sys.argv) > 4 else 16
LAST = int(sys.argv[5]) if len(sys.argv) > 5 else 104
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
    return m.group(1) if m else "<invalid>"

base = disasm(LO, HI)
print(f"base: {base}")
for bit in range(FIRST, LAST + 1):
    lo, hi = LO, HI
    if bit < 64:
        lo ^= 1 << bit
    else:
        hi ^= 1 << (bit - 64)
    text = disasm(lo, hi)
    if text != base:
        print(f"{bit:3d}: {text}")
