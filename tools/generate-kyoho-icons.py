#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Kyoho Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regenerates the raster icons in dist/ from the SVG artwork in dist/branding/.

Needs Pillow and resvg-py (pip install pillow resvg-py). Run it after editing the SVGs:

    python tools/generate-kyoho-icons.py

kyoho.svg is the full bunch; kyoho_small.svg has fewer, larger grapes and is used for the icon
sizes up to 32 px, where ten grapes turn into a smudge.
"""

import io
import struct
from pathlib import Path

import resvg_py
from PIL import Image

DIST = Path(__file__).resolve().parent.parent / "dist"
BRANDING = DIST / "branding"
SMALL_MAX = 32
ICO_SIZES = (16, 20, 24, 32, 40, 48, 64, 96, 128, 256)
ICNS_SIZES = (16, 32, 64, 128, 256, 512, 1024)


def render(size: int) -> Image.Image:
    name = "kyoho_small.svg" if size <= SMALL_MAX else "kyoho.svg"
    png = resvg_py.svg_to_bytes(svg_path=str(BRANDING / name), width=size, height=size)
    return Image.open(io.BytesIO(bytes(png))).convert("RGBA")


def bmp_with_alpha(image: Image.Image) -> bytes:
    """A 32-bit BMP with a V5 header and channel masks: Pillow's own writer drops the alpha."""
    width, height = image.size
    pixels = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM).tobytes("raw", "BGRA")
    header = struct.pack("<IiiHHIIiiII", 124, width, height, 1, 32, 3, len(pixels), 2835, 2835,
                         0, 0)
    header += struct.pack("<4I", 0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000)
    header += b"BGRs" + bytes(124 - len(header) - 4)
    offset = 14 + len(header)
    return b"BM" + struct.pack("<IHHI", offset + len(pixels), 0, 0, offset) + header + pixels


def main() -> None:
    icon = render(256)
    icon.save(DIST / "kyoho.png", optimize=True)
    # The SDL frontend loads its window icon from a BMP
    (DIST / "kyoho.bmp").write_bytes(bmp_with_alpha(icon))
    icon.save(DIST / "kyoho.ico", sizes=[(s, s) for s in ICO_SIZES],
              append_images=[render(s) for s in ICO_SIZES if s != 256])
    render(1024).save(DIST / "kyoho.icns", append_images=[render(s) for s in ICNS_SIZES[:-1]])
    for name in ("kyoho.png", "kyoho.bmp", "kyoho.ico", "kyoho.icns"):
        print(f"{name}: {(DIST / name).stat().st_size} bytes")


if __name__ == "__main__":
    main()
