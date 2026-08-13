# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Andrei Orehov / Andromediastudio - https://andromediastudio.com
# comfyui-vae-float32 - https://github.com/AndreiOrehov/comfyui-vae-float32
"""Per-frame sharpness across an EXR sequence, to find tiling seams offline.

The node does this inside a graph; this is the same measurement for a folder you
already rendered, and for regression-testing a change in settings.

    python tools/measure_seams.py <folder-with-exr> [more folders...]
"""
import glob
import os
import sys

import numpy as np
import OpenEXR


def sharp(path):
    with OpenEXR.File(path) as f:
        a = f.channels()["RGB"].pixels.astype(np.float32)
    y = a[..., 0] * 0.2126 + a[..., 1] * 0.7152 + a[..., 2] * 0.0722
    lap = (-4.0 * y[1:-1, 1:-1] + y[:-2, 1:-1] + y[2:, 1:-1]
           + y[1:-1, :-2] + y[1:-1, 2:])
    return float(np.abs(lap).mean())


for folder in sys.argv[1:]:
    files = sorted(glob.glob(os.path.join(folder, "*.exr")))
    if not files:
        print(f"{folder}: no EXR files")
        continue
    v = np.array([sharp(f) for f in files])
    dips = [i + 1 for i in range(1, len(v) - 1) if v[i] < v[i - 1] * 0.93]
    print(f"\n{os.path.basename(folder)}: {len(v)} frames, median sharpness {np.median(v):.5f}")
    print(f"  soft frames: {dips}")
    if len(dips) > 1:
        gaps = np.diff(dips)
        print(f"  gaps: {list(gaps)}"
              + ("   <-- REGULAR, that is a temporal tiling seam"
                 if len(set(gaps)) == 1 else ""))
