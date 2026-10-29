#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later

"""Emit the instruction words of one kernel of a cubin as a C++ initializer list.

Usage: python3 gen_kernel_code.py <cubin> <kernel> > kernel.inc
"""
import re, subprocess, sys

text = subprocess.run(["nvdisasm", "-hex", sys.argv[1]], capture_output=True, text=True,
                      check=True).stdout
section = text.split("\n.text." + sys.argv[2] + ":", 1)[1].split("//----", 1)[0]
lines = section.splitlines()
INSN = re.compile(r"/\*([0-9a-f]{4})\*/\s+(.*?)\s*;\s*/\* (0x[0-9a-f]{16}) \*/")
for i, line in enumerate(lines):
    m = INSN.search(line)
    if m:
        hi = re.search(r"0x[0-9a-f]{16}", lines[i + 1]).group(0)
        print(f"{m.group(3)}ULL, {hi}ULL, // {m.group(1)}: {m.group(2)}")
