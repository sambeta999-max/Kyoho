#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later

"""Patch one SASS instruction in a cubin and run it on the local GPU.

Usage: python3 hwrun.py <oracle.exe> <template.cubin> <find_lo> <find_hi> <new_lo> <new_hi>
                          <kernel> <threads> <words> [hex inputs...]
The instruction to replace is located by its exact 128-bit encoding.
"""
import os, struct, subprocess, sys, tempfile

exe, cubin = sys.argv[1], sys.argv[2]
find = struct.pack("<QQ", int(sys.argv[3], 16), int(sys.argv[4], 16))
new = struct.pack("<QQ", int(sys.argv[5], 16), int(sys.argv[6], 16))
data = bytearray(open(cubin, "rb").read())
pos = data.find(find)
if pos < 0 or data.find(find, pos + 1) >= 0:
    sys.exit("template instruction not found exactly once")
data[pos:pos + 16] = new
fd, patched = tempfile.mkstemp(suffix=".cubin", dir=os.path.dirname(cubin))
with os.fdopen(fd, "wb") as f:
    f.write(data)
try:
    r = subprocess.run([exe, patched, *sys.argv[7:]], capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
finally:
    os.remove(patched)
