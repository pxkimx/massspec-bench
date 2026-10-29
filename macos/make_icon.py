#!/usr/bin/env python3
"""Draw the app icon from the same geometry as the logo in web/index.html, and build AppIcon.icns.

    python macos/make_icon.py

The coordinates below are the SVG's 32-unit viewBox (a rounded panel, an L of axes and five spectrum
peaks), scaled onto Apple's 1024 canvas, so the icon and the in-app mark cannot drift apart.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
RES = HERE / "MassSpec Bench.app" / "Contents" / "Resources"
SS = 4
CANVAS, INSET, BOX = 1024, 100, 824
PANEL, AXES = "#0d1b22", "#6a7b84"
# (x, y_top) in SVG units; every peak stands on the axis at y = 25
PEAKS = [(10.0, 17.0, "#4ea3ff"), (13.5, 8.0, "#00e0cf"), (17.0, 19.0, "#b27bff"), (20.5, 12.0, "#ffb020"), (24.0, 21.0, "#ff5c8a")]


def main():
    N = CANVAS * SS
    im = Image.new("RGBA", (N, N), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    u = BOX * SS / 32.0
    o = INSET * SS
    P = lambda x, y: (o + x * u, o + y * u)  # noqa: E731
    d.rounded_rectangle([P(0.5, 0.5), P(31.5, 31.5)], radius=7.5 * u, fill=PANEL)
    w = int(1.05 * u)
    d.line([P(7, 4.6), P(7, 25.4)], fill=AXES, width=w)
    d.line([P(6.6, 25), P(27.4, 25)], fill=AXES, width=w)
    for x, top, c in PEAKS:
        d.line([P(x, 25), P(x, top)], fill=c, width=int(2 * u))
        d.ellipse([P(x - 1, top - 1), P(x + 1, top + 1)], fill=c)   # round caps, as stroke-linecap="round"
    im = im.resize((CANVAS, CANVAS), Image.LANCZOS)
    RES.mkdir(parents=True, exist_ok=True)
    im.save(RES / "AppIcon.png")
    if shutil.which("iconutil"):
        with tempfile.TemporaryDirectory() as td:
            iset = Path(td) / "AppIcon.iconset"
            iset.mkdir()
            for s in (16, 32, 128, 256, 512):
                im.resize((s, s), Image.LANCZOS).save(iset / f"icon_{s}x{s}.png")
                im.resize((2 * s, 2 * s), Image.LANCZOS).save(iset / f"icon_{s}x{s}@2x.png")
            subprocess.run(["iconutil", "-c", "icns", str(iset), "-o", str(RES / "AppIcon.icns")], check=True)
    print("wrote", RES / "AppIcon.png")


if __name__ == "__main__":
    main()
