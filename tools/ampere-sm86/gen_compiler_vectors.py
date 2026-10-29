#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later

"""Emit decoder test vectors from real nvcc output: one encoding per (opcode, mnemonic) pair.

Usage: python3 gen_compiler_vectors.py <cubin>... > ampere_decode_compiler_vectors.inc
"""
import re, subprocess, sys

INSN = re.compile(r"/\*[0-9a-f]{4,}\*/\s+(.*?)\s*;\s*/\* (0x[0-9a-f]{16}) \*/")
HI = re.compile(r"^\s*/\* (0x[0-9a-f]{16}) \*/\s*$")

seen = {}
for cubin in sys.argv[1:]:
    lines = subprocess.run(["nvdisasm", "-hex", cubin], capture_output=True, text=True,
                           check=True).stdout.splitlines()
    for i, line in enumerate(lines):
        m = INSN.search(line)
        if m and i + 1 < len(lines) and (h := HI.match(lines[i + 1])):
            text = m.group(1)
            if text.startswith("@"):
                text = text.split(None, 1)[1]
            lo, hi = int(m.group(2), 16), int(h.group(1), 16)
            seen.setdefault((lo & 0xfff, text.split()[0]), (lo, hi, text))
for (op, mn), (lo, hi, text) in sorted(seen.items()):
    print(f'{{0x{lo:016x}ULL, 0x{hi:016x}ULL, "{mn}"}}, // {text}')
