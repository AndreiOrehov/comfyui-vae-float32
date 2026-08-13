# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Andrei Orehov / Andromediastudio - https://andromediastudio.com
# comfyui-vae-float32 - https://github.com/AndreiOrehov/comfyui-vae-float32
"""Measure decode wall-clock for tile_size x precision on the LTX-2.5 video VAE.

Why an empty latent: decode cost is data-independent (same convolutions, same
tensor shapes), so zeros give representative timing and memory without paying for
a sampler first. Only the pixel VALUES are meaningless; the work is not.

Shape matches the reference sequence: 1280x704, 121 frames -> latent [1,128,16,22,40].

Usage:  python bench_tiling.py
Output: vae_float32_tools/bench_tiling_results.md  (+ live progress on stdout)
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

HOST = "http://127.0.0.1:8000"
VAE = "ltx-2.5-video-vae-bf16.safetensors"

# (label, precision, tile_size, tiled). Cheap ones first so numbers land early;
# the slowest case goes last.
CONFIGS = [
    ("warmup (loads the VAE, not timed)", "vae default", 384, True),
    ("bf16  tile 384", "vae default", 384, True),
    ("bf16  tile 768  <- official template", "vae default", 768, True),
    ("fp32  tile 384  <- our default", "float32", 384, True),
    ("fp32  tile 512  <- stock node default", "float32", 512, True),
    ("fp32  tile 768  <- official template in fp32", "float32", 768, True),
]

ap = argparse.ArgumentParser()
ap.add_argument("--width", type=int, default=1280)
ap.add_argument("--height", type=int, default=704, help="must be a multiple of 32")
ap.add_argument("--length", type=int, default=121)
ap.add_argument("--only", default="", help="substring filter over the config labels")
ap.add_argument("--out", default="bench_tiling_results.md")
args = ap.parse_args()
W, H, LEN = args.width, args.height, args.length
if W % 32 or H % 32:
    raise SystemExit(f"{W}x{H}: EmptyLTXVLatentVideo floor-divides by 32, pick multiples of 32")
if args.only:
    CONFIGS = [CONFIGS[0]] + [c for c in CONFIGS[1:] if args.only in c[0]]
OUT = Path(__file__).with_name(args.out)


def post(path, payload):
    req = urllib.request.Request(
        HOST + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read() or b"{}")


def get(path):
    with urllib.request.urlopen(HOST + path, timeout=60) as r:
        return json.loads(r.read())


def graph(precision, tile_size, tiled):
    return {
        "1": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "2": {"class_type": "EmptyLTXVLatentVideo",
              "inputs": {"width": W, "height": H, "length": LEN, "batch_size": 1}},
        "3": {"class_type": "VAEDecodeFloat32", "inputs": {
            "samples": ["2", 0], "vae": ["1", 0],
            "keep_out_of_range": True, "precision": precision, "tiled": tiled,
            "tile_size": tile_size, "overlap": 64,
            "temporal_size": 4096, "temporal_overlap": 8}},
        "4": {"class_type": "PreviewAny", "inputs": {"source": ["3", 1]}},
    }


def free_memory():
    try:
        post("/free", {"unload_models": False, "free_memory": True})
    except Exception as e:
        print("  (free failed:", e, ")")


def vram_free_gb():
    d = get("/system_stats")["devices"][0]
    return d["vram_free"] / 2**30, d["vram_total"] / 2**30


def run(precision, tile_size, tiled):
    free_memory()
    before, total = vram_free_gb()
    t0 = time.time()
    pid = post("/prompt", {"prompt": graph(precision, tile_size, tiled)})["prompt_id"]
    report, err = None, None
    while True:
        hist = get(f"/history/{pid}")
        if pid in hist:
            h = hist[pid]
            status = h.get("status", {})
            if status.get("status_str") == "error" or not status.get("completed", True):
                err = json.dumps(status)[:400]
            outs = h.get("outputs", {}).get("4", {})
            txt = outs.get("text") or outs.get("string")
            report = (txt[0] if isinstance(txt, list) and txt else None)
            break
        time.sleep(2)
    dt = time.time() - t0
    after, _ = vram_free_gb()
    return dt, before, after, total, report, err


def main():
    rows = []
    for label, precision, tile, tiled in CONFIGS:
        print(f"\n=== {label} ===", flush=True)
        dt, before, after, total, report, err = run(precision, tile, tiled)
        print(f"    {dt:7.1f} s | VRAM free {before:.1f} -> {after:.1f} / {total:.1f} GB", flush=True)
        if err:
            print("    ERROR:", err, flush=True)
        if report:
            print("   ", report.replace("\n", " | ")[:200], flush=True)
        if not label.startswith("warmup"):
            rows.append((label, dt, before, after, err, report))

    lines = [
        "# Tiling benchmark — LTX-2.5 video VAE",
        "",
        f"Decode only, empty latent {W}x{H}x{LEN} "
        f"(latent [1,128,{(LEN - 1) // 8 + 1},{H // 32},{W // 32}]), "
        "`temporal_size 4096`, `overlap 64`.",
        "Wall clock from POST /prompt to history completion, VAE preloaded by a warmup run.",
        "",
        "| config | decode time | VRAM free before -> after |",
        "|---|---|---|",
    ]
    for label, dt, before, after, err, _ in rows:
        note = " ⚠️ ERROR" if err else ""
        lines.append(f"| {label} | **{dt:.1f} s**{note} | {before:.1f} -> {after:.1f} GB |")
    base = next((r[1] for r in rows if "fp32  tile 384" in r[0]), None)
    if base:
        lines += ["", "Relative to fp32 / tile 384:", ""]
        for label, dt, *_ in rows:
            lines.append(f"- {label}: {dt / base:.2f}x")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\nwrote", OUT, flush=True)


if __name__ == "__main__":
    main()
