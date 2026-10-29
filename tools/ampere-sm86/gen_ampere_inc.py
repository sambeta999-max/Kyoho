#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later

"""Generate ampere.inc and decoder test vectors from the nvdisasm sweep.

Usage: python3 gen_ampere_inc.py <sweep.json> <ampere.inc> <vectors.inc>
"""
import json, re, sys
from collections import Counter, defaultdict

sweep = json.load(open(sys.argv[1]))
INC, VEC = sys.argv[2], sys.argv[3]

def full_mnemonic(text):
    toks = text.split()
    if toks[0].startswith("@"):
        toks = toks[1:]
    return toks[0]

# Primary mnemonic of each 12-bit opcode: the one nvdisasm prints most often
primary = {}
samples = {}
for k, seen in sweep.items():
    op = int(k, 16)
    mn = max(seen, key=lambda m: len(seen[m]))
    primary[op] = mn
    samples[op] = seen[mn]

# Group opcodes by (mnemonic, base9)
groups = defaultdict(set)
for op, mn in primary.items():
    groups[(mn, op & 0x1ff)].add(op >> 9)

# Name each group by its mnemonic; mnemonics spread over several bases are told apart by the
# overrides below, or by their base number
by_mn = defaultdict(list)
for (mn, base) in groups:
    by_mn[mn].append(base)
names = {}
for (mn, base) in groups:
    names[(mn, base)] = f"{mn} ({base:03x})" if len(by_mn[mn]) > 1 else mn

# Families spread across several bases, named from the sweep samples.
# Bases whose distinction is not yet understood keep their number.
OVERRIDES = {
    0x024: "IMAD", 0x025: "IMAD.WIDE", 0x027: "IMAD.HI",
    0x031: "HFMA2", 0x035: "HFMA2.MMA",
    0x104: "F2F", 0x110: "F2F.F64",
    0x105: "F2I", 0x111: "F2I.F64",
    0x106: "I2F", 0x112: "I2F.64",
    0x107: "FRND", 0x113: "FRND.F64",
    0x01c: "PLOP3", 0x01d: "PLOP3 (1 sign)", 0x01e: "PLOP3 (2 sign)", 0x01f: "PLOP3 (3 sign)",
    0x09c: "UPLOP3", 0x09d: "UPLOP3 (1 sign)", 0x09e: "UPLOP3 (2 sign)", 0x09f: "UPLOP3 (3 sign)",
    0x090: "UIADD3", 0x097: "UIADD3.64",
    0x0a4: "UIMAD", 0x0a5: "UIMAD.WIDE",
    0x0b8: "ULDC (b8)", 0x0b9: "ULDC", 0x0bb: "ULDC (indexed)",
    0x152: "RPCMOV (to Rpc)", 0x153: "RPCMOV (from Rpc)", 0x154: "RPCMOV.64",
    0x155: "BMOV (from B)", 0x156: "BMOV (to B)", 0x157: "BMOV.64",
    0x18a: "ATOM", 0x18b: "ATOM.CAS",
    0x18c: "ATOMS", 0x18d: "ATOMS.CAS",
    0x1a8: "ATOMG", 0x1a9: "ATOMG.CAS",
    0x193: "SUATOM", 0x194: "SUATOM (bindless)", 0x195: "SUATOM.CAS", 0x196: "SUATOM.CAS (bindless)",
    0x197: "SULD.P", 0x198: "SULD.P (bindless)", 0x199: "SULD.D", 0x19a: "SULD.D (bindless)",
    0x19b: "SUST.P", 0x19c: "SUST.P (bindless)", 0x19d: "SUST.D", 0x19e: "SUST.D (bindless)",
    0x19f: "SURED", 0x1a0: "SURED (bindless)",
    # nvdisasm always prints these modifiers
    0x012: "LOP3", 0x092: "ULOP3", 0x00a: "FSET", 0x1ae: "LDGSTS",
    0x143: "CALL.ABS", 0x144: "CALL.REL", 0x1b0: "ARRIVES.LDGSTSBAR",
}
for i, fam in enumerate(["TEX", "TLD4", "TLD", "TMML", "TXD", "TXQ"]):
    base = 0x160 + 3 * i
    OVERRIDES.update({base: fam, base + 1: f"{fam} (bindless)", base + 2: f"{fam} ({base + 2:03x})"})
OVERRIDES.update({0x172: "FOOTPRINT", 0x173: "FOOTPRINT (bindless)", 0x174: "FOOTPRINT (174)"})

for key in names:
    if key[1] in OVERRIDES and OVERRIDES[key[1]].split(".")[0].split(" ")[0] == key[0]:
        names[key] = OVERRIDES[key[1]]

def ident(cute):
    return re.sub(r"[^A-Za-z0-9]+", "_", cute).strip("_").upper()

used = Counter(ident(c) for c in names.values())
lines = []
primary_name = {}
for (mn, base), forms in sorted(groups.items(), key=lambda x: (names[x[0]], x[0][1])):
    cute = names[(mn, base)]
    name = ident(cute)
    if used[name] > 1:
        name = f"{name}_{base:03X}"
    # One entry per base; forms that belong to another mnemonic must not alias it
    others = {f for f in range(8) if primary.get((f << 9) | base) not in (None, mn)}
    base_bits = format(base, "09b")
    base_str = f"{base_bits[0]} {base_bits[1:5]} {base_bits[5:]}"
    if not others:
        encs = [f"--- {base_str}"]
    else:
        encs = [f"{format(f, '03b')} {base_str}" for f in sorted(forms)]
    for i, enc in enumerate(encs):
        # Duplicate entries share the enum through INST_ALIAS
        macro = "INST" if i == 0 else "INST_ALIAS"
        lines.append(f'{macro}({name + ",":16s} "{cute}",{" " * max(1, 22 - len(cute))}"{enc}")')
    for f in forms:
        primary_name[(f << 9) | base] = name

with open(INC, "w", newline="\n") as f:
    f.write("// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project\n")
    f.write("// SPDX-License-Identifier: GPL-3.0-or-later\n\n")
    f.write("// SM86 opcode table, bits [11..0]: form[11..9] base[8..0].\n")
    f.write("// Generated from an nvdisasm -b SM86 sweep of the 12-bit opcode space.\n\n")
    f.write("\n".join(lines) + "\n")

with open(VEC, "w", newline="\n") as f:
    for op in sorted(primary):
        s = samples[op][0]
        f.write(f'{{0x{s["lo"]}ULL, 0x{s["hi"]}ULL, Opcode::{primary_name[op]}}}, // {s["text"]}\n')

print(f"{len(groups)} groups, {len(lines)} entries, {len(primary)} opcodes")
