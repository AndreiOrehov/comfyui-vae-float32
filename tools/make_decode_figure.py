# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Andrei Orehov / Andromediastudio - https://andromediastudio.com
# comfyui-vae-float32 - https://github.com/AndreiOrehov/comfyui-vae-float32
"""The one figure: stock decode against float32, on one frame, six ways.

Every number on the plate is computed here from the EXR passed in, so the
picture cannot drift away from the data the way a hand-typed caption does.
"Stock" is simulated honestly - clamp to [0,1] AND drop to the bfloat16 grid,
which is what comfy/sd.py hands over. Clamping alone leaves float32 precision
behind and inflates the level count by orders of magnitude.

Usage:  python make_decode_figure.py <frame.exr> [-o out.png]
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches      # noqa: E402
import matplotlib.pyplot as plt           # noqa: E402
import numpy as np                        # noqa: E402
import OpenEXR                            # noqa: E402
import torch                              # noqa: E402

BG, FG, DIM, ACC, WARN = "#0d1117", "#e6edf3", "#8b949e", "#4da3ff", "#f0883e"
LO, HI = 0.2, 0.3

ap = argparse.ArgumentParser()
ap.add_argument("source")
ap.add_argument("-o", "--out", default=None)
ap.add_argument("--patch", type=int, default=190, help="side of the detail patch, in pixels")
args = ap.parse_args()

f32 = list(OpenEXR.File(args.source).channels().values())[0].pixels.astype(np.float32)
stock = torch.from_numpy(np.clip(f32, 0, 1)).to(torch.bfloat16).to(torch.float32).numpy()
H, W = f32.shape[:2]


def levels(a):
    w = a[(a > LO) & (a < HI)]
    return int(np.unique(w).size)


def show(ax, img, title=None, sub=None):
    ax.imshow(np.clip(img, 0, 1))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#30363d")
    if title:
        ax.set_title(title + ("\n" + sub if sub else ""), color=FG, fontsize=11, pad=8)


# --- pick the patch: smooth, mid-dark, and actually inside the measured window
lum = f32 @ np.array([0.2126, 0.7152, 0.0722], np.float32)
P = args.patch
best, px, py = None, 0, 0
for y in range(0, H - P, 24):
    for x in range(0, W - P, 24):
        block = lum[y:y + P, x:x + P]
        if not (LO < block.mean() < HI + 0.25):
            continue
        # smooth: little high-frequency energy, so the bf16 staircase is not hidden by detail
        e = float(np.abs(np.diff(block, axis=1)).mean())
        share = float(((block > LO) & (block < HI)).mean())
        score = e - share * 0.02
        if best is None or score < best:
            best, px, py = score, x, y

cs, cf = stock[py:py + P, px:px + P], f32[py:py + P, px:px + P]
d = np.abs(stock - f32)

fig = plt.figure(figsize=(17.6, 10.4), dpi=100, facecolor=BG)
fig.text(0.5, 0.962, "LTX-2.5 decode: stock ComfyUI vs float32 EXR  -  same latent, same frame",
         color=FG, fontsize=17, ha="center")

ax = fig.add_axes([0.035, 0.545, 0.28, 0.34]); show(ax, f32)
ax.set_title(f"{os.path.basename(args.source)}  (float32 EXR)", color=DIM, fontsize=11, pad=8)
ax.add_patch(patches.Rectangle((px, py), P, P, fill=False, ec="#f5d90a", lw=2))

ax = fig.add_axes([0.355, 0.545, 0.28, 0.34])
show(ax, cs, "stock decode  -  bf16 + clamp", f"{levels(cs):,} levels in [{LO},{HI}] here")
ax = fig.add_axes([0.675, 0.545, 0.28, 0.34])
show(ax, cf, "ours  -  fp32, no clamp", f"{levels(cf):,} levels in [{LO},{HI}] here")

# --- what the clamp deletes
vis = np.dstack([np.clip(lum, 0, 1) ** (1 / 2.2) * 0.30] * 3)
over, under = (f32 > 1.0).any(2), (f32 < 0.0).any(2)
vis[over] = (1.0, 0.25, 0.20); vis[under] = (0.25, 0.50, 1.00)
ax = fig.add_axes([0.035, 0.115, 0.28, 0.34])
show(ax, vis, "what the clamp deletes", "red  above 1.0        blue  below 0.0")

# --- how far apart the two decodes are
ax = fig.add_axes([0.375, 0.135, 0.26, 0.30], facecolor=BG)
nz = (stock - f32).ravel()
ax.hist(nz[np.abs(nz) > 1e-9], bins=160, color=ACC, log=True)
ax.set_title("difference where the two decodes disagree", color=FG, fontsize=11, pad=8)
ax.set_xlabel("stock minus float32", color=DIM, fontsize=10)
ax.set_ylabel("samples (log)", color=DIM, fontsize=10)
for s in ax.spines.values():
    s.set_color("#30363d")
ax.tick_params(colors=DIM, labelsize=9)

# --- the staircase, which is the whole point
ax = fig.add_axes([0.695, 0.135, 0.26, 0.30], facecolor=BG)
# The steps are 1/1024 apart, invisible on a full-height axis - so find the flattest
# stretch of the flattest row and show only that. This is the staircase, life size.
g_s, g_f = cs[..., 1], cf[..., 1]
SEG = 70
rows = np.abs(np.diff(g_f, axis=1)).mean(axis=1)
row = int(np.argmin(rows))
starts = np.abs(np.diff(g_f[row]))
seg = int(np.argmin([starts[i:i + SEG].sum() for i in range(len(starts) - SEG)]))
xs = np.arange(seg, seg + SEG)
ax.step(xs, g_s[row, seg:seg + SEG], where="mid", color=WARN, lw=1.6, label="stock (bf16)")
ax.plot(xs, g_f[row, seg:seg + SEG], color=ACC, lw=1.2, label="float32")
ax.set_title("one scanline, zoomed to the quantisation", color=FG, fontsize=11, pad=8)
ax.set_xlabel(f"pixels {seg}-{seg + SEG} of row {row}", color=DIM, fontsize=10)
for s in ax.spines.values():
    s.set_color("#30363d")
ax.tick_params(colors=DIM, labelsize=9)
leg = ax.legend(facecolor=BG, edgecolor="#30363d", fontsize=9, labelcolor=FG)

pa = float(over.mean() * 100); pb = float(under.mean() * 100)
fig.text(0.5, 0.045,
         f"stock: +0.0000 .. +1.0000        float32: {f32.min():+.4f} .. {f32.max():+.4f}"
         f"        outside the clamp: {pa:.3f}% above, {pb:.3f}% below"
         f"        max |difference|: {d.max():.4f}",
         color=DIM, fontsize=12, ha="center")

out = args.out or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "docs", "assets", "decode_compared.png")
fig.savefig(out, facecolor=BG)
print(f"patch at {px},{py} ({P}x{P})")
print(f"levels in [{LO},{HI}]  patch: stock {levels(cs):,}  float32 {levels(cf):,}")
print(f"                       frame: stock {levels(stock):,}  float32 {levels(f32):,}")
print("wrote", out)
