#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later

"""Compare the methods of two NVIDIA class headers from open-gpu-doc.

Usage: python3 class_diff.py <old.h> <new.h> [--eden <engine.h>] > report.md
Example: class_diff.py clb197.h clc797.h --eden src/video_core/engines/maxwell_3d.h

With --eden, every affected offset is annotated with the Eden register field covering it, taken
from the ASSERT_REG_POSITION list of the engine header.

Methods are keyed by name without the class prefix. Array methods are compared by base offset
and stride. The report lists, in order of risk for code written against the old class:
methods whose offset now belongs to a different method, methods that moved, removed and new
methods, and methods whose fields changed.
"""
import os, re, sys
from collections import defaultdict

DEFINE = re.compile(r"#define\s+NV[0-9A-F]{4}_(\w+?)(\((\w)\))?\s+\(?(0x[0-9a-fA-F]+)(?:\+\(\w\)\*(\d+))?\)?\s*$")
FIELD = re.compile(r"#define\s+NV[0-9A-F]{4}_(\w+)\s+(\d+):(\d+)\s*$")


def parse(path):
    """Hex defines named after a field of the current method are its values, others are methods"""
    methods = {}
    current = None
    for line in open(path):
        m = FIELD.match(line)
        if m and current and m.group(1).startswith(current + "_"):
            field = m.group(1)[len(current) + 1:]
            methods[current]["fields"][field] = {"bits": f"{m.group(2)}:{m.group(3)}", "values": {}}
            continue
        m = DEFINE.match(line)
        if not m:
            continue
        name, value, stride = m.group(1), int(m.group(4), 16), m.group(5)
        if current and name.startswith(current + "_") and not m.group(2):
            rest = name[len(current) + 1:]
            fields = methods[current]["fields"]
            owner = next((f for f in sorted(fields, key=len, reverse=True)
                          if rest.startswith(f + "_")), None)
            if owner is not None:
                fields[owner]["values"][rest[len(owner) + 1:]] = value
                continue
        current = name
        # Host classes also describe GPFIFO entries and pushbuffer headers; those are not methods.
        # Values of a method declared earlier are not methods either.
        is_value = any(name.startswith(f"{other}_{field}_")
                       for other, method in methods.items() for field in method["fields"])
        if value % 4 != 0 or name.startswith(("GP_ENTRY", "DMA_")) or is_value:
            current = None
            continue
        methods[name] = {"offset": value, "stride": int(stride) if stride else 0, "fields": {}}
    return methods


def location(method):
    if method["stride"]:
        return f"0x{method['offset']:04x}+i*{method['stride']}"
    return f"0x{method['offset']:04x}"


def parse_eden(path):
    """Eden fields as (byte offset, name), from ASSERT_REG_POSITION in byte or word units"""
    text = open(path).read()
    scale = 4 if "== position * 4" in text else 1
    fields = [(int(m.group(2), 16) * scale, m.group(1))
              for m in re.finditer(r"^ASSERT_REG_POSITION\((\w+),\s*(0x[0-9a-fA-F]+)\);", text, re.M)]
    return sorted(fields)


eden = parse_eden(sys.argv[sys.argv.index("--eden") + 1]) if "--eden" in sys.argv else None


def eden_field(offset):
    """Eden field whose range, up to the next asserted field, contains the offset"""
    if not eden:
        return ""
    owner = ""
    for start, name in eden:
        if start > offset:
            break
        owner = name
    return f"`{owner}`" if owner else "-"


old, new = parse(sys.argv[1]), parse(sys.argv[2])
old_at = defaultdict(list)
for name, method in old.items():
    old_at[method["offset"]].append(name)
new_at = defaultdict(list)
for name, method in new.items():
    new_at[method["offset"]].append(name)

repurposed, moved, removed, added, changed = [], [], [], [], []
for name, method in sorted(old.items(), key=lambda x: x[1]["offset"]):
    # The old method's offset now decodes as a different method
    others = [n for n in new_at.get(method["offset"], []) if n != name]
    if others and name not in new_at.get(method["offset"], []):
        repurposed.append((method["offset"], name, others))
    if name not in new:
        removed.append(name)
    elif new[name]["offset"] != method["offset"] or new[name]["stride"] != method["stride"]:
        moved.append(name)
    elif new[name]["fields"] != method["fields"]:
        changed.append(name)
for name, method in sorted(new.items(), key=lambda x: x[1]["offset"]):
    if name not in old:
        added.append(name)

print(f"# {os.path.basename(sys.argv[1])} -> {os.path.basename(sys.argv[2])}\n")
print(f"{len(old)} methods before, {len(new)} after: {len(repurposed)} offsets reused by another "
      f"method, {len(moved)} moved, {len(removed)} removed, {len(added)} added, "
      f"{len(changed)} with changed fields.\n")

eden_column = " Eden field |" if eden else ""
eden_rule = "---|" if eden else ""


def eden_cell(offset):
    return f" {eden_field(offset)} |" if eden else ""


print("## Offsets now used by a different method\n")
print("Writes decoded with the old layout land in the wrong state.\n")
print(f"| Offset | Old method | New method |{eden_column}\n|---|---|---|{eden_rule}")
for offset, name, others in repurposed:
    print(f"| {location(old[name])} | {name} | {', '.join(others)} |{eden_cell(offset)}")

print("\n## Moved\n")
print(f"| Method | Old | New |{eden_column}\n|---|---|---|{eden_rule}")
for name in moved:
    print(f"| {name} | {location(old[name])} | {location(new[name])} |"
          f"{eden_cell(old[name]['offset'])}")

print("\n## Removed\n")
print(f"| Offset | Method |{eden_column}\n|---|---|{eden_rule}")
for name in removed:
    print(f"| {location(old[name])} | {name} |{eden_cell(old[name]['offset'])}")

print("\n## Added\n")
print("| Offset | Method |\n|---|---|")
for name in added:
    print(f"| {location(new[name])} | {name} |")

print("\n## Changed fields\n")
for name in changed:
    field = f", Eden {eden_field(new[name]['offset'])}" if eden else ""
    print(f"### {name} ({location(new[name])}){field}\n")
    before, after = old[name]["fields"], new[name]["fields"]
    for field in sorted(set(before) | set(after)):
        a, b = before.get(field), after.get(field)
        if a == b:
            continue
        if a is None:
            print(f"- added `{field}` {b['bits']}")
        elif b is None:
            print(f"- removed `{field}` {a['bits']}")
        else:
            if a["bits"] != b["bits"]:
                print(f"- `{field}` bits {a['bits']} -> {b['bits']}")
            for value in sorted(set(a["values"]) | set(b["values"])):
                va, vb = a["values"].get(value), b["values"].get(value)
                if va != vb:
                    print(f"- `{field}_{value}`: {va if va is not None else '-'} -> "
                          f"{vb if vb is not None else '-'}")
    print()
