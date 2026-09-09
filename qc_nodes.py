# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Andrei Orehov / Andromediastudio - https://andromediastudio.com
# comfyui-vae-float32 - https://github.com/AndreiOrehov/comfyui-vae-float32
"""Ingest QC for AI-generated video - the check a VFX pipeline TD runs before a plate is accepted.

Runway, Seedance, Kling and friends hand you an mp4 and no report. What arrives is routinely
not what the shot list says it is:

  * 8-bit 4:2:0 with no colour tags at all, so every reader downstream guesses BT.601 and the
    grade starts from the wrong primaries;
  * frames that are held or literally repeated, because the generator ran out of motion - a
    24 fps clip that is really 12 fps in places, which conforms and then judders on a pan;
  * a black frame or a one-frame luma jump at a stitch point;
  * NaN, once a workflow has been through a float32 decode and an upscaler.

None of that is visible in a scrub-through at speed, and all of it is cheap to measure. This
node measures it, prints one verdict, and draws a contact sheet of the frames it flagged so the
number can be looked at rather than believed.

Nothing here writes to the image: the batch is read, never returned - wire the source image on
to the rest of the graph directly.
"""

import json as _json
import logging
import os
import re
import shutil
import subprocess

import numpy as np
import torch

import folder_paths

logger = logging.getLogger("vae_float32")

try:                                        # normal import, inside the installed pack
    from .nodes import CATEGORY_ANDRO, _SLOT, _luma
except ImportError:                         # standalone import (tests, tooling) - nodes.py drags
    CATEGORY_ANDRO = "ANDRO"                # in comfy.utils, which only exists inside ComfyUI.
    _SLOT = "  "

    def _luma(a):
        return a[..., 0] * 0.2126 + a[..., 1] * 0.7152 + a[..., 2] * 0.0722

# A frame delta below this reads as "the same picture again". 1/1024 rather than 0.0 because a
# re-encode never gives back a bit-exact repeat: h264 quantisation moves a duplicated frame by a
# few thousandths, and demanding exact equality would report zero duplicates on every mp4 ever.
_DUP_EPS = 1.0 / 1024.0

# The long side the temporal statistics run at. Frame-to-frame deltas and mean luma are averages
# over the whole frame, so they converge long before full resolution - and at 4K the difference is
# a QC pass that takes a minute instead of a second. Level counting does NOT use this: quantisation
# lives in the individual samples, and a subsample would throw away exactly the evidence.
_TEMPORAL_LONG_SIDE = 512

# Frames sampled for the distinct-value count. Eight is enough to catch the grid (a quantised
# source is quantised in every frame) and cheap enough to stay honest about full resolution.
_LEVEL_FRAMES = 8

# Samples used to test which quantisation grid the values sit on.
_GRID_SAMPLES = 200_000

# Half a step of the finest grid tested is 7.6e-6; float32 rounding near 1.0 is 1.2e-7. 1e-6 sits
# between the two, so it absorbs the arithmetic without ever letting one grid pass for another.
_GRID_TOL = 1e-6

# ffprobe's way of saying "there is no tag here". An untagged file is not a broken file - it is a
# file whose meaning depends on who opens it, which is worse, because nothing errors.
_UNTAGGED = {"", "unknown", "reserved", "n/a", "none", "unspecified"}

_MAX_LIST = 30                              # flagged-frame lists truncated to this in the report
_SHEET_COLS = 4
_SHEET_MAX = 24
_SHEET_OK = 8


# --------------------------------------------------------------------------- probe


def _pix_fmt_depth(pix_fmt):
    """Bits per component implied by an ffmpeg pixel format name, or None.

    Deliberately conservative: the formats an AI video generator emits are yuv420p and, rarely,
    yuv420p10le / yuv444p10le. Anything the pattern does not recognise is reported as unknown
    rather than guessed at - a wrong bit depth in a QC report is worse than a missing one.
    """
    pf = (pix_fmt or "").lower()
    if not pf:
        return None
    m = re.search(r"(\d{1,2})(le|be)$", pf)
    if m:
        bits = int(m.group(1))
        if bits > 16 and pf[:3] in ("rgb", "bgr", "gbr"):
            return bits // 3                # rgb48le is 16 bits per component, not 48
        return bits
    if re.match(r"^(yuv|yuvj|gray|gbr|rgb|bgr|nv|pal|ya)", pf):
        return 8
    return None


def _ffprobe(path):
    """Container truth for one video file: {..} on success, or {"error": "..."}.

    Kept separate from the pixel statistics on purpose. The tensor says what the decoder produced;
    the container says what the file CLAIMS. Most ingest surprises are the gap between the two.
    """
    exe = shutil.which("ffprobe")
    if exe is None:
        return {"error": "ffprobe not found on PATH"}
    if not os.path.isfile(path):
        return {"error": f"file not found: {path}"}
    cmd = [exe, "-v", "error", "-select_streams", "v:0", "-show_entries",
           "stream=codec_name,pix_fmt,width,height,r_frame_rate,nb_frames,"
           "color_space,color_transfer,color_primaries,color_range",
           "-of", "json", path]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"error": f"ffprobe failed: {exc}"}
    if out.returncode != 0:
        return {"error": f"ffprobe exit {out.returncode}: {out.stderr.strip()[:200]}"}
    try:
        streams = _json.loads(out.stdout).get("streams") or []
    except ValueError as exc:
        return {"error": f"ffprobe returned unparsable json: {exc}"}
    if not streams:
        return {"error": "no video stream"}

    s = streams[0]
    fps = None
    rate = s.get("r_frame_rate") or ""
    if "/" in rate:
        num, den = rate.split("/", 1)
        try:
            fps = float(num) / float(den) if float(den) else None
        except ValueError:
            fps = None
    nb = s.get("nb_frames")
    try:
        nb = int(nb)
    except (TypeError, ValueError):
        nb = None

    tags = {k: (s.get(k) or "") for k in ("color_space", "color_transfer", "color_primaries")}
    untagged = [k for k, v in tags.items() if v.strip().lower() in _UNTAGGED]
    return {
        "codec_name": s.get("codec_name"),
        "pix_fmt": s.get("pix_fmt"),
        "pix_fmt_bits": _pix_fmt_depth(s.get("pix_fmt")),
        "width": s.get("width"),
        "height": s.get("height"),
        "r_frame_rate": rate,
        "fps": fps,
        "nb_frames": nb,
        "color_space": tags["color_space"],
        "color_transfer": tags["color_transfer"],
        "color_primaries": tags["color_primaries"],
        "color_range": s.get("color_range") or "",
        "untagged": untagged,
    }


# --------------------------------------------------------------------------- measurement


def _levels(arr):
    """Distinct values per channel on evenly spaced frames, and which grid they sit on.

    Two separate questions, and both matter:

      * how many distinct values a channel actually holds - the ceiling on what any grade can
        pull out of the plate;
      * whether those values land on a regular k/m grid - which names the format the picture
        came out of, whatever the container now says it is.

    Grids are tested 255 -> 1023 -> 65535 and the FIRST fit wins, because 255 divides 65535
    (65535 = 255 x 257): every 8-bit value is also exactly a 16-bit value, so testing the coarse
    grid last would report 8-bit material as 16-bit.
    """
    n = arr.shape[0]
    idx = np.unique(np.linspace(0, n - 1, min(_LEVEL_FRAMES, n)).round().astype(int))
    per_channel, samples = [], []
    for i in idx:
        frame = arr[i]
        for c in range(frame.shape[-1]):
            v = frame[..., c].ravel()
            v = v[np.isfinite(v)]
            if v.size:
                per_channel.append(int(np.unique(v).size))
                samples.append(v)
    max_levels = max(per_channel) if per_channel else 0

    grid, grid_share = None, {}
    if samples:
        pool = np.concatenate(samples)
        if pool.size > _GRID_SAMPLES:
            pool = pool[:: max(1, pool.size // _GRID_SAMPLES)]
        pool = pool.astype(np.float64)
        for m in (255, 1023, 65535):
            share = float(np.mean(np.abs(np.round(pool * m) / m - pool) <= _GRID_TOL))
            grid_share[m] = share
            if grid is None and share >= 0.999:
                grid = m

    if max_levels <= 256:
        implied = "8-bit source"
    elif max_levels <= 1024:
        implied = "10-bit source"
    else:
        implied = "float/16-bit source"
    bits = {None: None, 255: 8, 1023: 10, 65535: 16}[grid]
    return {"max_distinct_per_channel": max_levels, "implied": implied,
            "frames_sampled": [int(i) for i in idx], "grid": grid, "grid_bits": bits,
            "grid_fit": {str(k): v for k, v in grid_share.items()}}


def _runs(flags):
    """Longest run of consecutive True in a boolean list, and where it starts."""
    best, best_at, cur, cur_at = 0, -1, 0, -1
    for i, f in enumerate(flags):
        if f:
            if cur == 0:
                cur_at = i
            cur += 1
            if cur > best:
                best, best_at = cur, cur_at
        else:
            cur = 0
    return best, best_at


def _measure(arr, held_ratio, flash_jump, black_level):
    """Every pixel statistic the report is built from. arr is (N,H,W,3) float32, as delivered."""
    n, h, w = arr.shape[0], arr.shape[1], arr.shape[2]

    finite = np.isfinite(arr)
    nan_count = int(np.isnan(arr).sum())
    inf_count = int(np.isinf(arr).sum())
    bad_frames = [int(i) for i in range(n) if not finite[i].all()]
    total = int(arr.size)
    below = float(np.count_nonzero(arr < 0.0) / total * 100.0)      # NaN compares False, so it
    above = float(np.count_nonzero(arr > 1.0) / total * 100.0)      # never counts as clipping

    # Non-finite samples are neutralised for the temporal maths only. Left in, a single NaN
    # poisons its frame's mean and every delta touching it, and the report would then blame the
    # neighbouring frames for a fault that belongs to one.
    clean = np.where(finite, arr, 0.0).astype(np.float32)

    step = max(1, int(np.ceil(max(h, w) / float(_TEMPORAL_LONG_SIDE))))
    small = clean[:, ::step, ::step, :]
    y = _luma(small)                                                # (N, h', w')

    mean_luma = y.reshape(n, -1).mean(axis=1).astype(np.float64)
    d_luma = np.diff(mean_luma) if n > 1 else np.zeros(0)
    delta = (np.abs(np.diff(y, axis=0)).reshape(max(n - 1, 0), -1).mean(axis=1).astype(np.float64)
             if n > 1 else np.zeros(0))

    black = [int(i) for i in range(n) if mean_luma[i] < black_level]
    # Indexed at i+1: the jump is a property of the frame you land ON, which is the frame an
    # editor would pull. d_luma[i] is frame i+1 minus frame i.
    flash = [int(i + 1) for i in range(d_luma.size) if abs(d_luma[i]) > flash_jump]
    cut = np.abs(d_luma) > flash_jump
    flicker = float(np.std(d_luma[~cut])) if d_luma.size and (~cut).any() else 0.0

    median_delta = float(np.median(delta)) if delta.size else 0.0
    dup = [int(i + 1) for i in range(delta.size) if delta[i] < _DUP_EPS]
    # The median guard matters: on a locked-off shot the median delta is already near zero, and
    # held_ratio x ~0 flags the entire clip as held. Below 1/255 of movement there is nothing to
    # measure a held frame against, so the check declines to answer rather than answer wrongly.
    held_threshold = held_ratio * median_delta
    held_valid = median_delta > 1.0 / 255.0
    held = ([int(i + 1) for i in range(delta.size) if delta[i] < held_threshold]
            if held_valid else [])
    held_flags = [False] * n
    for i in held:
        held_flags[i] = True
    longest_held, longest_at = _runs(held_flags)

    return {
        "frames": n, "height": h, "width": w, "channels": int(arr.shape[-1]),
        "temporal_subsample": step,
        "nan": nan_count, "inf": inf_count, "nonfinite_frames": bad_frames,
        "pct_below_0": below, "pct_above_1": above, "pct_outside_unit": below + above,
        "mean_luma": [float(v) for v in mean_luma],
        "luma_delta": [float(v) for v in d_luma],
        "frame_delta": [float(v) for v in delta],
        "mean_luma_overall": float(mean_luma.mean()) if n else 0.0,
        "median_frame_delta": median_delta,
        "held_threshold": float(held_threshold), "held_check_valid": bool(held_valid),
        "black_frames": black, "flash_frames": flash, "flicker": flicker,
        "duplicate_frames": dup, "held_frames": held,
        "held_pct": (100.0 * len(held) / n) if n else 0.0,
        "longest_held_run": int(longest_held), "longest_held_run_at": int(longest_at),
    }


# --------------------------------------------------------------------------- contact sheet


def _font():
    from PIL import ImageFont
    for name in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, 12)
        except OSError:
            continue
    return ImageFont.load_default()


def _sheet(arr, picks, thumb_w=224):
    """Contact sheet of the flagged frames, labelled with number and reason.

    A list of frame numbers is not evidence. Half of what this node flags is a judgement call at
    the margin - a held frame on a slow push is fine, a held frame mid-pan is not - and the only
    way to settle that is to look at the frames it named, side by side, which is what this is.
    """
    from PIL import Image, ImageDraw

    n, h, w = arr.shape[0], arr.shape[1], arr.shape[2]
    if not picks or n == 0:
        return torch.zeros((1, 8, 8, 3), dtype=torch.float32)

    thumb_h = max(1, int(round(thumb_w * h / float(w))))
    label_h, pad, bg = 18, 6, 0.09
    cols = min(_SHEET_COLS, len(picks))
    rows = int(np.ceil(len(picks) / float(cols)))
    cell_w, cell_h = thumb_w + pad, thumb_h + label_h + pad
    sheet = np.full((rows * cell_h + pad, cols * cell_w + pad, 3), bg, np.float32)
    font = _font()

    for k, (idx, reason) in enumerate(picks):
        frame = np.nan_to_num(arr[idx], nan=0.0, posinf=1.0, neginf=0.0)
        pic = Image.fromarray((np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8), "RGB")
        pic = pic.resize((thumb_w, thumb_h), Image.BILINEAR)

        strip = Image.new("RGB", (thumb_w, label_h), (18, 18, 22))
        draw = ImageDraw.Draw(strip)
        text = f"{idx:05d}  {reason}"
        # Trimmed by measurement rather than by a character count: the label is drawn in whatever
        # font was found, and a truncation guess would either overflow or waste half the strip.
        while text and draw.textlength(text, font=font) > thumb_w - 6:
            text = text[:-1]
        draw.text((3, 3), text, fill=(235, 235, 235), font=font)

        r, c = divmod(k, cols)
        y0 = pad + r * cell_h
        x0 = pad + c * cell_w
        sheet[y0:y0 + thumb_h, x0:x0 + thumb_w] = np.asarray(pic, np.float32) / 255.0
        sheet[y0 + thumb_h:y0 + thumb_h + label_h, x0:x0 + thumb_w] = (np.asarray(strip, np.float32)
                                                                      / 255.0)
    return torch.from_numpy(sheet).unsqueeze(0)


def _picks(m, n):
    """Which frames go on the sheet, worst first, one entry per frame with the reasons merged."""
    order = [("black", m["black_frames"]), ("nan/inf", m["nonfinite_frames"]),
             ("flash", m["flash_frames"]), ("dup", m["duplicate_frames"]),
             ("held", m["held_frames"])]
    reasons = {}
    for name, frames in order:
        for i in frames:
            reasons.setdefault(i, []).append(name)
    if not reasons:
        picks = [(int(i), "ok") for i in
                 np.unique(np.linspace(0, n - 1, min(_SHEET_OK, n)).round().astype(int))]
        return picks, False
    rank = [nm for nm, _ in order]
    ranked = sorted(reasons.items(), key=lambda kv: (rank.index(kv[1][0]), kv[0]))
    return [(i, "+".join(r)) for i, r in ranked[:_SHEET_MAX]], True


# --------------------------------------------------------------------------- report


def _trim(values):
    head = ", ".join(str(v) for v in values[:_MAX_LIST])
    return head + (f", ... (+{len(values) - _MAX_LIST} more)" if len(values) > _MAX_LIST else "")


class ANDROVideoQC:
    """Ingest QC for a generated clip: levels, range, black/flash frames, held frames, colour."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "The clip to check, straight off the loader. Read only - nothing "
                               "comes back out, so wire the source on to the graph directly."}),
            },
            "optional": {
                "source_path": ("STRING", {
                    "default": "",
                    "tooltip": "The video file the batch was loaded from. With it, ffprobe adds "
                               "what the CONTAINER claims - codec, pix_fmt, fps, and the colour "
                               "tags - which is the half of ingest QC no pixel statistic can "
                               "reach. Leave empty and everything else still runs."}),
                "expected_fps": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 240.0, "step": 0.001,
                    "tooltip": "What the shot list says. 0 = do not check. Only checkable from "
                               "ffprobe: an IMAGE batch has no timebase."}),
                "expected_frames": ("INT", {
                    "default": 0, "min": 0, "max": 1_000_000,
                    "tooltip": "Expected frame count. 0 = do not check. A generator that returns "
                               "96 frames for a 4 s 24 fps order has silently changed the edit."}),
                "expected_width": ("INT", {
                    "default": 0, "min": 0, "max": 16384,
                    "tooltip": "Expected width in pixels. 0 = do not check."}),
                "expected_height": ("INT", {
                    "default": 0, "min": 0, "max": 16384,
                    "tooltip": "Expected height in pixels. 0 = do not check."}),
                "held_ratio": ("FLOAT", {
                    "default": 0.2, "min": 0.01, "max": 1.0, "step": 0.01,
                    "tooltip": "A frame counts as HELD when its mean absolute delta to the "
                               "previous frame falls below this fraction of the clip's median "
                               "delta. 0.2 = 'moved less than a fifth as much as this clip "
                               "normally moves'. Relative on purpose - an absolute threshold "
                               "would call a locked-off shot a stall and a whip pan clean. The "
                               "check switches itself off when the median delta is under 1/255, "
                               "because there is then no motion to be held against."}),
                "flash_jump": ("FLOAT", {
                    "default": 0.25, "min": 0.01, "max": 1.0, "step": 0.01,
                    "tooltip": "Mean luma jump between neighbouring frames, in 0..1 units, that "
                               "reads as a flash or a hard cut. Those frames are also excluded "
                               "from the flicker figure, so one legitimate cut does not make an "
                               "otherwise steady clip look unstable."}),
                "black_level": ("FLOAT", {
                    "default": 0.03, "min": 0.0, "max": 1.0, "step": 0.005,
                    "tooltip": "Mean luma below which a frame is called black. Always a FAIL: a "
                               "black frame in a generated clip is a dropped frame, not a "
                               "creative choice."}),
                "write_json": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Also write the JSON report to disk, so a batch of clips leaves a "
                               "trail that can be diffed later."}),
                "json_prefix": ("STRING", {
                    "default": "qc/report",
                    "tooltip": "Path under the ComfyUI output folder, without the extension."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "IMAGE", "BOOLEAN")
    RETURN_NAMES = ("report", "json", "sheet", "pass")
    OUTPUT_TOOLTIPS = ("The verdict and one line per check, as Markdown - readable on the canvas "
                       "and pasteable into a delivery note.",
                       "The same measurements as JSON, with every flagged-frame list complete "
                       "rather than truncated. For logging a batch of clips.",
                       "Contact sheet of the flagged frames, each labelled with its number and "
                       "why it was flagged - or eight evenly spaced frames when nothing was.",
                       "True only on PASS. WARN and FAIL are both False, so this can gate a "
                       "downstream branch without reading the text.")
    FUNCTION = "run"
    CATEGORY = CATEGORY_ANDRO
    OUTPUT_NODE = True
    SEARCH_ALIASES = ["VideoQC", "Video QC", "qc", "ingest check", "plate check", "held frames",
                      "duplicate frames", "black frame", "flicker", "colour tags", "color tags",
                      "bit depth check video", "runway kling seedance check"]
    DESCRIPTION = ("Ingest QC for AI-generated video: effective bit depth and quantisation grid, "
                   "range and NaN, black / flash / duplicate / held frames, flicker, resolution "
                   "and fps against expectations, and the container's colour tags via ffprobe. "
                   "One verdict, plus a labelled contact sheet of every frame it flagged.")

    def run(self, images, source_path="", expected_fps=0.0, expected_frames=0, expected_width=0,
            expected_height=0, held_ratio=0.2, flash_jump=0.25, black_level=0.03,
            write_json=False, json_prefix="qc/report"):
        arr = images.detach().float().cpu().numpy()
        if arr.ndim != 4 or arr.shape[-1] < 3:
            msg = f"expected an (N,H,W,3) IMAGE batch, got {tuple(arr.shape)}"
            logger.warning("[vae_float32] video qc: %s", msg)
            return {"ui": {"text": [msg]},
                    "result": (msg, _json.dumps({"error": msg}), torch.zeros((1, 8, 8, 3)), False)}
        arr = arr[..., :3]

        m = _measure(arr, held_ratio, flash_jump, black_level)
        lv = _levels(arr)
        probe = _ffprobe(source_path.strip()) if source_path.strip() else None
        n = m["frames"]

        # Reconcile the two depth answers before anything is reported, because on its own the
        # level count is a trap. A yuv420p file reaches an IMAGE batch through a YUV -> RGB
        # matrix (ComfyUI's loader asks swscale for gbrpf32le), and that matrix maps the 8-bit
        # grid onto a much finer one: measured on two real Seedance clips, 8-bit h264 comes out
        # as ~59 000 distinct values per channel sitting exactly on k/65535, which would read as
        # "16-bit source" and be wrong. Whenever the container states a depth, IT is the
        # authority on the source and the level count describes the conversion.
        lv["reconcile"] = ""
        lv["source_bits"] = lv["grid_bits"]
        if probe and not probe.get("error") and probe.get("pix_fmt_bits"):
            lv["source_bits"] = probe["pix_fmt_bits"]
            implied_bits = lv["grid_bits"] or 32
            if implied_bits > probe["pix_fmt_bits"]:
                lv["reconcile"] = (
                    f"the container says {probe['pix_fmt_bits']}-bit ({probe['pix_fmt']}), and it "
                    f"wins: a subsampled YUV source is matrixed to RGB on load, which lands the "
                    f"values on a finer grid than the file ever held. Read the level count as the "
                    f"decode path's, not the plate's.")
            elif implied_bits < probe["pix_fmt_bits"]:
                lv["reconcile"] = (
                    f"fewer levels than the container's {probe['pix_fmt_bits']}-bit "
                    f"({probe['pix_fmt']}) promises - the depth is there in the file and empty.")

        # ---- verdict. Collected as sentences rather than flags: a QC report that says FAIL
        # without saying what failed just moves the investigation somewhere else.
        fails, warns = [], []
        if m["nan"] or m["inf"]:
            fails.append(f"{m['nan']} NaN and {m['inf']} Inf samples across "
                         f"{len(m['nonfinite_frames'])} frame(s)")
        if m["black_frames"]:
            fails.append(f"{len(m['black_frames'])} black frame(s) "
                         f"(mean luma < {black_level:g})")

        spec = []
        if expected_width and m["width"] != expected_width:
            fails.append(f"width {m['width']} != expected {expected_width}")
            spec.append(f"width {m['width']} vs {expected_width} MISMATCH")
        elif expected_width:
            spec.append(f"width {m['width']} ok")
        if expected_height and m["height"] != expected_height:
            fails.append(f"height {m['height']} != expected {expected_height}")
            spec.append(f"height {m['height']} vs {expected_height} MISMATCH")
        elif expected_height:
            spec.append(f"height {m['height']} ok")
        if expected_frames and n != expected_frames:
            fails.append(f"{n} frames != expected {expected_frames}")
            spec.append(f"frames {n} vs {expected_frames} MISMATCH")
        elif expected_frames:
            spec.append(f"frames {n} ok")
        if expected_fps:
            got = (probe or {}).get("fps")
            if got is None:
                spec.append(f"fps unverifiable (no ffprobe data) vs {expected_fps:g}")
                warns.append("fps could not be checked: an IMAGE batch carries no timebase, and "
                             "no source_path was probed")
            elif abs(got - expected_fps) > 0.01:
                fails.append(f"fps {got:.3f} != expected {expected_fps:g}")
                spec.append(f"fps {got:.3f} vs {expected_fps:g} MISMATCH")
            else:
                spec.append(f"fps {got:.3f} ok")

        if probe and not probe.get("error"):
            if probe["untagged"]:
                warns.append("untagged colour ({}): readers will assume BT.601 / unknown transfer"
                             .format(", ".join(t.replace("color_", "") for t in probe["untagged"])))
            if probe.get("pix_fmt_bits") and probe["pix_fmt_bits"] >= 10 \
                    and lv["max_distinct_per_channel"] <= 256:
                warns.append(f"container says {probe['pix_fmt_bits']}-bit ({probe['pix_fmt']}) but "
                             f"the pixels hold {lv['max_distinct_per_channel']} levels per channel "
                             f"- the extra depth is an empty promise")
        if m["held_frames"] and m["held_pct"] > 10.0:
            warns.append(f"{len(m['held_frames'])} held frame(s) = {m['held_pct']:.1f}% of the "
                         f"clip: the generator ran out of motion, so the real frame rate is lower "
                         f"than the file's")
        if m["duplicate_frames"]:
            warns.append(f"{len(m['duplicate_frames'])} duplicate frame(s)")
        if m["pct_outside_unit"] > 1.0:
            warns.append(f"{m['pct_outside_unit']:.3f}% of samples outside [0,1]")

        verdict = "FAIL" if fails else ("WARN" if warns else "PASS")
        passed = verdict == "PASS"

        # ---- report
        L = [f"# ANDRO Video QC - **{verdict}**", ""]
        for f in fails:
            L.append(f"- FAIL: {f}")
        for w in warns:
            L.append(f"- WARN: {w}")
        if not fails and not warns:
            L.append("- nothing flagged")
        L += ["", f"**clip** {n} frames, {m['width']}x{m['height']}, "
                  f"dtype {images.dtype}, temporal stats at 1/{m['temporal_subsample']} scale"]
        grid = (f"grid k/{lv['grid']} ({lv['grid_bits']}-bit), "
                f"{lv['grid_fit'][str(lv['grid'])] * 100:.2f}% of samples fit"
                if lv["grid"] else "no k/255, k/1023 or k/65535 grid fits - continuous values")
        L.append(f"**levels** max {lv['max_distinct_per_channel']} distinct values per channel "
                 f"over {len(lv['frames_sampled'])} frames -> {lv['implied']}; {grid}")
        if lv["reconcile"]:
            L.append(f"{_SLOT}^ {lv['reconcile']}")
        L.append(f"**range** {m['pct_below_0']:.4f}% below 0, {m['pct_above_1']:.4f}% above 1; "
                 f"NaN {m['nan']}, Inf {m['inf']}")
        L.append(f"**luma** mean {m['mean_luma_overall']:.4f}; black frames "
                 f"{len(m['black_frames'])} (< {black_level:g}); flashes/cuts "
                 f"{len(m['flash_frames'])} (|d| > {flash_jump:g}); flicker "
                 f"{m['flicker']:.5f} (std of luma delta off the cuts)")
        held_note = (f"held {len(m['held_frames'])} = {m['held_pct']:.1f}% "
                     f"(< {m['held_threshold']:.6f} = {held_ratio:g}x median), longest run "
                     f"{m['longest_held_run']}"
                     + (f" from frame {m['longest_held_run_at']}"
                        if m["longest_held_run"] else "")
                     if m["held_check_valid"] else
                     "held: not checked, median delta below 1/255 (nothing moves in this clip)")
        L.append(f"**temporal** median |delta| {m['median_frame_delta']:.6f}; duplicates "
                 f"{len(m['duplicate_frames'])} (< 1/1024); {held_note}")
        L.append(f"**spec** {'; '.join(spec) if spec else 'nothing to check against'}")
        if probe is None:
            L.append("**probe** no source_path given - container tags unchecked")
        elif probe.get("error"):
            L.append(f"**probe** unavailable: {probe['error']}")
        else:
            # Built by concatenation rather than nested f-strings: PEP 701 only landed in 3.12 and
            # this pack still has to import on the 3.10 and 3.11 ComfyUI builds in the wild.
            fps_note = " = {:.3f} fps".format(probe["fps"]) if probe["fps"] else ""
            nb_note = probe["nb_frames"] if probe["nb_frames"] is not None else "n/a"
            L.append(f"**probe** {probe['codec_name']} {probe['pix_fmt']} "
                     f"({probe['pix_fmt_bits'] or '?'}-bit) {probe['width']}x{probe['height']} "
                     f"@ {probe['r_frame_rate']}{fps_note}, nb_frames {nb_note}")
            L.append(f"**colour** space={probe['color_space'] or '-'} "
                     f"transfer={probe['color_transfer'] or '-'} "
                     f"primaries={probe['color_primaries'] or '-'} "
                     f"range={probe['color_range'] or '-'}"
                     + ("  <- untagged" if probe["untagged"] else ""))

        flagged = [("black", m["black_frames"]), ("nan/inf", m["nonfinite_frames"]),
                   ("flash", m["flash_frames"]), ("duplicate", m["duplicate_frames"]),
                   ("held", m["held_frames"])]
        if any(v for _, v in flagged):
            L += ["", "**flagged frames** (0-based)"]
            for name, frames in flagged:
                if frames:
                    L.append(f"- {name} ({len(frames)}): {_trim(frames)}")

        payload = {"verdict": verdict, "pass": passed, "fails": fails, "warns": warns,
                   "source_path": source_path.strip(), "levels": lv, "probe": probe,
                   "spec": {"expected_fps": expected_fps, "expected_frames": expected_frames,
                            "expected_width": expected_width, "expected_height": expected_height,
                            "notes": spec},
                   "thresholds": {"held_ratio": held_ratio, "flash_jump": flash_jump,
                                  "black_level": black_level, "dup_eps": _DUP_EPS},
                   "measurements": m}
        js = _json.dumps(payload, indent=2)

        if write_json:
            path = os.path.join(folder_paths.get_output_directory(),
                                json_prefix.strip() or "qc/report")
            path = path if path.lower().endswith(".json") else path + ".json"
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(js)
            L += ["", f"json written to `{path}`"]

        picks, _flagged_any = _picks(m, n)
        sheet = _sheet(arr, picks)
        report = "\n".join(L)
        logger.info("[vae_float32] video qc: %s", verdict)
        return {"ui": {"text": [report]}, "result": (report, js, sheet, passed)}


NODE_CLASS_MAPPINGS = {"ANDROVideoQC": ANDROVideoQC}
NODE_DISPLAY_NAME_MAPPINGS = {"ANDROVideoQC": "ANDRO Video QC"}
