# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Andrei Orehov / Andromediastudio - https://andromediastudio.com
# comfyui-vae-float32 - https://github.com/AndreiOrehov/comfyui-vae-float32
"""Honest float32 out of any ComfyUI VAE, and the tools to prove it.

ComfyUI finishes every VAE decode with (comfy/sd.py:502)

    process_output = lambda image: image.add_(1.0).div_(2.0).clamp_(0.0, 1.0)

and runs most VAEs in bfloat16. Two things follow, and neither is visible from
inside a graph:

  * whatever the decoder emitted outside [0,1] is gone - on a real LTX-2.5
    generation that is -0.0715 .. +1.0445, i.e. specular highlights and shadow
    detail, deleted before any node downstream can see them;
  * the values that survive carry bf16 precision. Measured across [0.2, 0.3] of
    one frame: 77 distinct levels, step 1/1024. Decoded in fp32 instead: 3.3
    million levels. No float32 container recovers that after the fact.

These nodes remove both losses, write the result as real float32 EXR, and give
you the measurements to check any of it yourself.
"""

import logging
import os

import numpy as np
import torch

import comfy.utils
import folder_paths

logger = logging.getLogger("vae_float32")

# One house name for the whole pack, so a graph shows at a glance which nodes are ours: it is the
# node menu's category, the prefix of every display name, and the prefix of every class key. ANDRO is
# short for Andromediastudio; kept to five characters because it sits in front of every node title.
CATEGORY_ANDRO = "ANDRO"

# The default process_output, minus the clamp. In-place (add_/div_) on purpose:
# comfy/sd.py:1215 calls process_output for its side effect and throws the return
# value away, so a pure-functional replacement would silently do nothing there.
_NO_CLAMP = lambda image: image.add_(1.0).div_(2.0)   # noqa: E731
_IDENTITY = lambda image: image                        # noqa: E731

_SLOT = "  "

# Tells "input not connected" apart from "connected but not evaluated yet", which a lazy
# input reports as None. Same trick the core's Soft Switch uses.
_MISSING = object()


# --------------------------------------------------------------------------- helpers


def _unclamped_process_output(vae):
    """A clamp-free stand-in for THIS vae's process_output, or None if unrecognised.

    Not every VAE uses the [-1,1] -> [0,1] default. TAEHV / lighttae (sd.py:894,
    906), MiniMax H3 (976) and StageA (540) already emit [0,1] and set identity -
    substituting the default there would rescale the image and wreck it. So probe
    the real function with -1/0/1 instead of assuming.
    """
    probe = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)
    try:
        got = vae.process_output(probe.clone()).float()
    except Exception:
        return None
    if torch.allclose(got, torch.tensor([0.0, 0.5, 1.0]), atol=1e-4):
        return _NO_CLAMP
    if torch.allclose(got, torch.tensor([-1.0, 0.0, 1.0]), atol=1e-4):
        return _IDENTITY
    return None


def _round_tile(px):
    """Nearest multiple of 32 at or below px - the step the tile widgets use."""
    return max(64, int(px) // 32 * 32)


def _tile_fits_in_vram(vae, latent, tile_size, dtype):
    """(bytes needed, bytes free, verdict) for decoding ONE tile - or None if unknowable.

    verdict is "fits" / "tight" / "spills", deliberately three states rather than a
    yes/no. Calibration against everything this pack has actually measured (RTX 5090,
    30.3 GiB reported free, LTX-2.5, 1280x704x121):

        float32 tile 384 -> 14.5 GiB    44 s
        float32 tile 512 -> 25.7 GiB    42 s
        float32 tile 768 -> 57.9 GiB  1247 s   <- the cliff
        bf16    tile 768 -> 28.9 GiB    12.1 s <- 95% of free, and perfectly fine

    That last row is why a single threshold cannot work: a hard cut at 0.9 condemns a
    run that demonstrably took 12 seconds, and a cut at 1.0 leaves no room for the
    assembled output batch. So say "tight" in between and let the user decide, and never
    state more confidence than four points from one GPU can support. The free figure is
    also read at graph time, before the model is fully resident, so it is an upper bound.
    """

    try:
        import comfy.model_management as mm
        comp = vae.spacial_compression_decode()
        side = max(1, tile_size // comp)                # tile measured in LATENT samples
        shape = list(latent.shape)
        shape[-2], shape[-1] = side, side              # one tile, not the whole frame
        need = float(vae.memory_used_decode(tuple(shape), dtype))
        free = float(mm.get_free_memory(vae.device))
    except Exception:
        return None                                    # unknown VAE shape or no such API
    if not need or not free:
        return None
    ratio = need / free
    verdict = "fits" if ratio < 0.85 else ("tight" if ratio < 1.0 else "spills")
    return need, free, verdict


def _luma(a):
    return a[..., 0] * 0.2126 + a[..., 1] * 0.7152 + a[..., 2] * 0.0722


def _sharpness(y):
    """mean |Laplacian| - high-frequency energy of one frame."""
    lap = (-4.0 * y[1:-1, 1:-1] + y[:-2, 1:-1] + y[2:, 1:-1]
           + y[1:-1, :-2] + y[1:-1, 2:])
    return float(np.abs(lap).mean())


def _range_report(t, label):
    a = t.detach().float().cpu().numpy().ravel()
    n = a.size
    sample = a if n <= 4_000_000 else a[:: max(1, n // 4_000_000)]
    p = np.percentile(sample, [0.1, 1, 50, 99, 99.9])
    below = float((a < 0.0).mean() * 100.0)
    above = float((a > 1.0).mean() * 100.0)
    return (
        f"{label}: min={a.min():+.6f} max={a.max():+.6f} mean={a.mean():+.6f}\n"
        f"{_SLOT}p0.1={p[0]:+.5f} p1={p[1]:+.5f} p50={p[2]:+.5f} "
        f"p99={p[3]:+.5f} p99.9={p[4]:+.5f}\n"
        f"{_SLOT}outside [0,1]: {below:.4f}% below 0, {above:.4f}% above 1 "
        f"({below + above:.4f}% would be lost to the standard clamp)"
    )


def _box_mean(a, k):
    """Mean over a k x k window, via an integral image - no scipy, and O(1) per pixel.

    Written out rather than pulled from a library because this pack ships to people who
    already have to install an EXR backend; one more hard dependency for a box filter
    is not worth it.
    """
    k = max(1, int(k) | 1)                              # odd, so the window has a centre
    r = k // 2
    pad = np.pad(a, r, mode="edge")
    integral = pad.cumsum(0).cumsum(1)
    integral = np.pad(integral, ((1, 0), (1, 0)))
    h, w = a.shape
    total = (integral[k:k + h, k:k + w] - integral[0:h, k:k + w]
             - integral[k:k + h, 0:w] + integral[0:h, 0:w])
    return total / float(k * k)


def _local_std(y, k=7):
    """Standard deviation in a k x k window. E[x^2] - E[x]^2, clipped at 0 for rounding."""
    mean = _box_mean(y, k)
    mean_sq = _box_mean(y * y, k)
    return np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))


def _quantisation_step(values):
    """Smallest non-zero gap between neighbouring distinct values, or nan.

    This is what actually reveals the format: bfloat16 across [0.2, 0.3] can only land on
    77 values, spaced 1/1024 apart, and that is the FORMAT's entire grid there rather than
    a property of the picture.
    """
    uniq = np.unique(values)
    if uniq.size < 4:
        return float("nan"), uniq.size
    gaps = np.diff(uniq)
    gaps = gaps[gaps > 0]
    return (float(gaps.min()) if gaps.size else float("nan")), uniq.size


def _effective_bits(step):
    """Bits implied by a quantisation step across the unit interval."""
    if not np.isfinite(step) or step <= 0:
        return float("nan")
    return float(np.log2(1.0 / step))


def _banding_risk(frame_luma, step, smooth_k=9):
    """Mask of where banding will show, plus the share of the frame at risk.

    Learned the hard way: amplifying the difference 25x produced a black frame. On dense
    noisy material (neon, rain, skin) even eight bits survive an aggressive grade, because
    the noise dithers the quantisation away. What breaks is the opposite kind of area -
    sky, haze, smooth gradients.

    So this is decided against the measured quantisation STEP, not against percentiles of
    the frame. Percentiles cannot work here: on a perfectly smooth ramp every pixel has the
    same slope, so "steeper than the 60th percentile" is false everywhere and the very case
    we are hunting scores zero. The physics instead:

      * a band is one quantisation level held across several pixels, so its width in pixels
        is step/slope - visible when that is 2 pixels or more, i.e. slope <= step/2;
      * where slope is 0 there is no transition to band at all;
      * local noise of the order of the step dithers the edge and the band disappears,
        so risk needs residual < step.
    """
    if not np.isfinite(step) or step <= 0:
        return np.zeros_like(frame_luma, dtype=bool), 0.0
    gy, gx = np.gradient(frame_luma)
    slope = np.hypot(gx, gy)
    smooth = _box_mean(frame_luma, smooth_k)
    residual = _local_std(frame_luma - smooth, smooth_k)     # texture, with the ramp removed
    risk = (slope > 0) & (slope <= step * 0.5) & (residual < step)
    return risk, float(risk.mean() * 100.0)


def _ssim(a, b, k=7):
    """Mean SSIM over a k x k window, on luma, with the usual C1/C2 stabilisers.

    Here because PSNR measures error ENERGY: a difference spread thinly over a whole frame
    and one concentrated in the sky score alike, and the second is the one that bands.
    """
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_a, mu_b = _box_mean(a, k), _box_mean(b, k)
    # Clipped at zero because E[x^2] - E[x]^2 is only non-negative in exact arithmetic: in float32 a
    # flat window comes out slightly negative, which drives the denominator through zero and returns
    # values in the tens of thousands. Seen on a real decode before this line existed - SSIM read
    # -100341.29. Covariance is left alone; it is legitimately signed.
    var_a = np.maximum(_box_mean(a * a, k) - mu_a * mu_a, 0.0)
    var_b = np.maximum(_box_mean(b * b, k) - mu_b * mu_b, 0.0)
    cov = _box_mean(a * b, k) - mu_a * mu_b
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2)
    # den >= c1 * c2 > 0 now, so the clamp is a belt-and-braces guard rather than the thing holding
    # the result together.
    return float(np.mean(num / np.maximum(den, 1e-12)))


def _histogram_image(values, width=512, height=160, bins=256):
    """Log-scaled histogram as an IMAGE, with the [0,1] limits marked.

    Log scale on purpose: the out-of-range tails are a fraction of a percent of the samples
    (measured 0.01-0.34%), and on a linear axis they are invisible - which is the whole
    reason people believe nothing is being lost.
    """
    img = np.full((height, width, 3), 0.09, np.float32)
    lo, hi = float(values.min()), float(values.max())
    if not np.isfinite(lo) or hi <= lo:
        return torch.from_numpy(img).unsqueeze(0)
    counts, edges = np.histogram(values, bins=bins, range=(lo, hi))
    scaled = np.log1p(counts.astype(np.float64))
    scaled = scaled / max(scaled.max(), 1e-9)
    for i, v in enumerate(scaled):
        x0 = int(i / bins * width)
        x1 = max(x0 + 1, int((i + 1) / bins * width))
        top = height - 1 - int(v * (height - 4))
        img[top:, x0:x1] = (0.30, 0.64, 1.0)
    for value, colour in ((0.0, (0.45, 0.16, 0.16)), (1.0, (0.45, 0.16, 0.16))):
        if lo <= value <= hi:                            # where the stock clamp cuts
            x = int((value - lo) / (hi - lo) * (width - 1))
            img[:, x] = colour
    return torch.from_numpy(img).unsqueeze(0)


def _plot(series, width=512, height=160, marks=()):
    """A tiny line chart as an IMAGE tensor - no matplotlib dependency."""
    img = np.full((height, width, 3), 0.09, np.float32)
    if len(series) < 2:
        return torch.from_numpy(img).unsqueeze(0)
    v = np.asarray(series, np.float32)
    lo, hi = float(v.min()), float(v.max())
    span = max(hi - lo, 1e-9)
    xs = (np.linspace(0, width - 1, len(v))).astype(int)
    ys = (height - 1 - (v - lo) / span * (height - 6) - 3).astype(int)
    for m in marks:                                   # vertical marker lines
        x = int(m / max(len(v) - 1, 1) * (width - 1))
        if 0 <= x < width:
            img[:, x] = (0.45, 0.16, 0.16)
    for i in range(len(v) - 1):                       # polyline
        x0, x1, y0, y1 = xs[i], xs[i + 1], ys[i], ys[i + 1]
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        for s in range(steps + 1):
            x = int(x0 + (x1 - x0) * s / steps)
            y = int(np.clip(y0 + (y1 - y0) * s / steps, 0, height - 1))
            img[y, x] = (0.30, 0.64, 1.0)
    return torch.from_numpy(img).unsqueeze(0)


# --------------------------------------------------------------------------- decode


class ANDROVAEDecode:
    """VAE Decode that keeps values outside [0,1] and can force a float32 decode."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT", {
                    "tooltip": "The latent to decode - the same input the stock VAE Decode takes, so "
                               "this node drops straight into an existing graph in its place. "
                               "Nested latents are unbound to their first element, as the stock "
                               "node does."}),
                "vae": ("VAE", {
                    "tooltip": "The VAE that belongs to the model. Any VAE works: this node probes "
                               "the VAE's own output transform instead of assuming the default one, "
                               "so VAEs that already emit [0,1] are left alone rather than rescaled."}),
                "keep_out_of_range": ("BOOLEAN", {
                    "default": True, "label_on": "keep (no clamp)", "label_off": "clamp to [0,1]",
                    "tooltip": "ComfyUI ends every decode with clamp_(0.0, 1.0) (comfy/sd.py:502), "
                               "and the decoder does emit outside that: measured -0.0715 .. +1.0445 "
                               "on a real LTX-2.5 generation, 0.01-0.34% of samples depending on the "
                               "shot. Those are specular highlights and shadow detail, deleted "
                               "before any node downstream can see them. ON keeps them - only "
                               "useful if what follows can carry them (EXR, or Remap Range first). "
                               "OFF reproduces stock ComfyUI exactly, for an A/B."}),
                "precision": (["vae default", "float32"], {
                    "default": "float32",
                    "tooltip": "'vae default' is bfloat16 on most VAEs, which is coarser than it "
                               "sounds: across [0.2, 0.3] of a frame it can represent 77 distinct "
                               "values, and that is the FORMAT's entire grid there, not a property "
                               "of the picture. float32 gives ~3.3 million in the same window. Costs "
                               "roughly 3x the decode time and 2x the VAE's VRAM. Nothing recovers "
                               "this afterwards - a float32 container around bf16 values is empty "
                               "precision."}),
                "tiled": ("BOOLEAN", {
                    # On by default since 1.3.0. A float32 decode holds the VAE at twice its usual
                    # weight and every intermediate at four bytes a sample, so the whole-frame path
                    # runs out of VRAM at resolutions the stock bf16 decode swallows. The defaults
                    # around it are chosen so this costs no visible seam: temporal_size stays high
                    # enough that nothing is cut along time (that is the cut that leaves a soft
                    # frame), and only the spatial split - the one overlap can actually blend - is
                    # used. Turn it off for a small still where the frame fits whole.
                    "default": True,
                    "tooltip": "Decode in tiles instead of whole frames. On by default: float32 "
                               "decoding needs far more VRAM than the stock bf16 path. Only the "
                               "SPATIAL split is active with these defaults, and overlap blends "
                               "it - temporal_size is left high on purpose, because cutting along "
                               "time is what leaves a soft frame on every seam."}),
            },
            "optional": {
                "tile_size": ("INT", {
                    "default": 384, "min": 64, "max": 4096, "step": 32,
                    "tooltip": "Spatial tile in PIXELS, and the knob to cut when you run out of "
                               "memory - cut it before ever touching temporal_size. Cost is a "
                               "CLIFF, not a slope: while the decode fits in VRAM the tile size is "
                               "nearly free, and the moment it stops fitting, weights start paging "
                               "and the same decode takes tens of times longer, with no error and "
                               "no warning - just a progress bar that stops moving. WHERE that "
                               "cliff sits depends on your card, your VAE and the frame size, so "
                               "there is no universal safe number; this node estimates it for YOUR "
                               "machine and says so in the report. One measured example, RTX 5090 "
                               "32GB on LTX-2.5 at 1280x704x121 in float32: 384 = 44s, 512 = 42s, "
                               "768 = 1247s. In bfloat16 the same run is 12.1s at both 384 and 768, "
                               "which is why ComfyUI's LTX-2.5 template ships 768 - right for bf16, "
                               "ruinous in float32 on that card."}),
                "overlap": ("INT", {
                    "default": 64, "min": 0, "max": 4096, "step": 32,
                    "tooltip": "Pixels of overlap between neighbouring spatial tiles, blended so the "
                               "join does not show. 64 against a 384 tile is enough in practice: "
                               "gradient excess at the tile boundaries measures 1.03-1.05x, well "
                               "under the 1.30x ANDRO Seam Check needs before it calls anything a "
                               "peak. Raise it only if Seam Check reports REGULAR vertical or "
                               "horizontal peaks - a single strong line is content, not a seam. "
                               "Bigger overlap means more pixels decoded twice, so it costs time."}),
                "temporal_size": ("INT", {
                    "default": 4096, "min": 8, "max": 4096, "step": 4,
                    "tooltip": "Video VAEs only: how many frames are decoded per temporal tile. "
                               "LEAVE IT HIGH. At 4096 nothing is cut along time at all, which is "
                               "the point - a diffusion decoder has no context at a temporal tile "
                               "edge, so the blend leaves a visibly SOFTER frame on every seam, "
                               "evenly spaced and easy to miss. Measured: temporal_size 32 put a "
                               "soft frame every 24 frames. Overlap softens that but never removes "
                               "it. If you are short on memory, cut tile_size instead - the spatial "
                               "seam is the one overlap can genuinely blend away. ANDRO Seam Check "
                               "finds these."}),
                "temporal_overlap": ("INT", {
                    "default": 8, "min": 4, "max": 4096, "step": 4,
                    "tooltip": "Frames of overlap between temporal tiles. Does nothing while "
                               "temporal_size stays at 4096, because then there is only one "
                               "temporal tile. It cannot rescue temporal tiling either: more "
                               "overlap softens the seam frame, it never removes it, since the "
                               "decoder still had no context at that edge."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "range_report")
    OUTPUT_TOOLTIPS = ("The decoded batch, still carrying values outside [0,1] when that is on.",
                       "What the decode actually did: measured range, percentiles, how much the "
                       "stock clamp would have deleted, and the dtypes it really ran in.")
    FUNCTION = "decode"
    CATEGORY = CATEGORY_ANDRO
    # Every name this node used to answer to, plus the words someone types when they do not know the
    # pack exists. Renaming a node otherwise makes it unfindable for everyone who learned the old name.
    SEARCH_ALIASES = ["VAEDecodeFloat32", "VAE Decode float32", "vae decode no clamp",
                      "unclamped decode", "decode without clamp", "hdr decode"]
    DESCRIPTION = ("VAE Decode without the [0,1] clamp, optionally in float32. Works with any VAE - "
                   "it probes the VAE's own output transform rather than assuming one.")

    def decode(self, samples, vae, keep_out_of_range, precision, tiled,
               tile_size=384, overlap=64, temporal_size=4096, temporal_overlap=8):
        latent = samples["samples"]
        if latent.is_nested:
            latent = latent.unbind()[0]

        fsm = vae.first_stage_model
        try:
            model_dtype = next(fsm.parameters()).dtype
        except StopIteration:
            model_dtype = None

        saved_process_output = vae.process_output
        saved_vae_dtype = vae.vae_dtype
        cast_back = None
        notes = []

        if keep_out_of_range:
            replacement = _unclamped_process_output(vae)
            if replacement is None:
                notes.append("unrecognised process_output on this VAE - clamp left in place")
                logger.warning("[vae_float32] %s", notes[-1])
            else:
                vae.process_output = replacement
                if replacement is _IDENTITY:
                    notes.append("this VAE already outputs [0,1] natively, nothing was clamping")

        if precision == "float32":
            if torch.float32 not in getattr(vae, "working_dtypes", [torch.float32]):
                notes.append("VAE does not list float32 as a working dtype - kept its own precision")
                logger.warning("[vae_float32] %s", notes[-1])
            else:
                try:
                    if model_dtype is not None and model_dtype != torch.float32:
                        fsm.to(torch.float32)
                        cast_back = model_dtype
                    # Set even when the weights are ALREADY float32: decode() casts the incoming latent
                    # to vae_dtype (comfy/sd.py:1206), so leaving it at bfloat16 quantises the input
                    # before the weights ever see it. Seen on Flux's ae.safetensors.
                    vae.vae_dtype = torch.float32
                except Exception as e:                 # quantised weights refuse the cast
                    notes.append(f"float32 cast failed ({type(e).__name__}), using {model_dtype}")
                    logger.warning("[vae_float32] %s", notes[-1])

        try:
            try:
                notes.append(f"decoded with weights={next(fsm.parameters()).dtype} "
                             f"vae_dtype={vae.vae_dtype}")
            except StopIteration:
                pass
            if tiled:
                fit = _tile_fits_in_vram(vae, latent, tile_size, torch.float32
                                         if precision == "float32" else vae.vae_dtype)
                if fit is not None:
                    need, free, verdict = fit
                    budget = f"tile {tile_size}: ~{need / 2**30:.1f} GiB needed against " \
                             f"{free / 2**30:.1f} GiB free"
                    if verdict == "fits":
                        notes.append(f"{budget} - fits")
                    elif verdict == "tight":
                        notes.append(f"{budget} - TIGHT. It may well run, but there is little room "
                                     f"left; if this decode crawls, that is why.")
                        logger.warning("[vae_float32] %s", notes[-1])
                    else:
                        notes.append(
                            f"{budget} - WILL NOT FIT. Weights start paging and the decode can take "
                            f"tens of times longer with no error raised (measured on one card: 42 s "
                            f"became 1247 s). Cut tile_size - try {_round_tile(tile_size * 0.66)} - "
                            f"before touching temporal_size.")
                        logger.warning("[vae_float32] %s", notes[-1])
                if tile_size < overlap * 4:
                    overlap = tile_size // 4
                if temporal_size < temporal_overlap * 2:
                    temporal_overlap = temporal_overlap // 2
                t_comp = vae.temporal_compression_decode()
                if t_comp is not None:
                    t_size = max(2, temporal_size // t_comp)
                    t_over = max(1, min(t_size // 2, temporal_overlap // t_comp))
                else:
                    t_size = t_over = None
                comp = vae.spacial_compression_decode()
                images = vae.decode_tiled(latent, tile_x=tile_size // comp, tile_y=tile_size // comp,
                                          overlap=overlap // comp, tile_t=t_size, overlap_t=t_over)
            else:
                images = vae.decode(latent)
        finally:
            vae.process_output = saved_process_output
            vae.vae_dtype = saved_vae_dtype
            if cast_back is not None:
                fsm.to(cast_back)

        if len(images.shape) == 5:                     # combine batches, as stock VAEDecode does
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        images = images.float()

        report = _range_report(images, f"decode ({precision}, "
                                       f"{'no clamp' if keep_out_of_range else 'clamped'})")
        for n in notes:
            report += f"\n{_SLOT}note: {n}"
        logger.info("[vae_float32] %s", report.replace("\n", " "))
        return (images, report)


class ANDROVAEEncode:
    """VAE Encode in float32, and without silently clipping an out-of-range plate.

    The stock encode casts the image to the VAE's working dtype (usually bf16)
    before it reaches the weights. Feeding it a float32 EXR plate throws the
    precision away at the door.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pixels": ("IMAGE", {
                    "tooltip": "The image to encode. A float32 EXR plate keeps its precision here "
                               "only if precision is float32."}),
                "vae": ("VAE", {
                    "tooltip": "The VAE that belongs to the model. A VAE is trained together with "
                               "its model - pairing one model's latents with another's VAE does not "
                               "fail loudly, it just decodes wrong."}),
                "precision": (["vae default", "float32"], {
                    "default": "float32",
                    "tooltip": "'vae default' casts the plate to the VAE's dtype (usually bfloat16) "
                               "before the weights see it, which throws away a float32 EXR's "
                               "precision at the door."}),
            },
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "report")
    OUTPUT_TOOLTIPS = ("The encoded latent.",
                       "The range of what was handed in, measured BEFORE the latent exists, so "
                       "out-of-range input is seen rather than inferred afterwards.")
    FUNCTION = "encode"
    CATEGORY = CATEGORY_ANDRO
    SEARCH_ALIASES = ["VAEEncodeFloat32", "VAE Encode float32", "encode exr", "float32 encode"]
    DESCRIPTION = ("VAE Encode in float32, the other end of the round trip. Reports the incoming "
                   "range before encoding, so a plate that is already out of range is visible.")

    def encode(self, pixels, vae, precision):
        fsm = vae.first_stage_model
        try:
            model_dtype = next(fsm.parameters()).dtype
        except StopIteration:
            model_dtype = None
        saved_dtype = vae.vae_dtype
        cast_back = None
        notes = []

        if precision == "float32" and torch.float32 in getattr(vae, "working_dtypes", [torch.float32]):
            try:
                if model_dtype is not None and model_dtype != torch.float32:
                    fsm.to(torch.float32)
                    cast_back = model_dtype
                vae.vae_dtype = torch.float32
            except Exception as e:
                notes.append(f"float32 cast failed ({type(e).__name__})")

        try:
            latent = vae.encode(pixels[:, :, :, :3])
        finally:
            vae.vae_dtype = saved_dtype
            if cast_back is not None:
                fsm.to(cast_back)

        a = pixels.detach().float().cpu().numpy()
        report = (f"encode ({precision}): input min={a.min():+.5f} max={a.max():+.5f}, "
                  f"latent {tuple(latent.shape)} {latent.dtype}")
        for n in notes:
            report += f"\n{_SLOT}note: {n}"
        logger.info("[vae_float32] %s", report.replace("\n", " "))
        return ({"samples": latent}, report)


# --------------------------------------------------------------------------- measure


class ANDRORangeStats:
    """How much of a batch lives outside [0,1], and how finely it is quantised."""

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE", {
                                 "tooltip": "The batch to measure. It comes back out unchanged, so "
                                            "this node can sit anywhere in a chain."}),
                             "label": ("STRING", {
                                 "default": "stats",
                                 "tooltip": "Free text, printed at the head of the report - so two "
                                            "of these in one graph can be told apart."})}}

    # New outputs are appended, never inserted: a saved workflow wires links by index, so adding
    # histogram/banding_mask at the end leaves every existing graph connected as it was.
    RETURN_TYPES = ("IMAGE", "STRING", "IMAGE", "MASK")
    RETURN_NAMES = ("image", "report", "histogram", "banding_mask")
    OUTPUT_TOOLTIPS = ("The image, passed through untouched - wire it onward to measure mid-chain.",
                       "Min/max/mean, percentiles, the share outside [0,1] the stock clamp would "
                       "delete, the measured quantisation step as EFFECTIVE BITS, and how much of "
                       "the frame is at risk of banding.",
                       "Log-scaled histogram with the [0,1] clamp limits marked in red. Log because "
                       "the out-of-range tails are a fraction of a percent - on a linear axis they "
                       "are invisible, which is why people believe nothing is lost.",
                       "Where banding will appear first: real gradient, too little noise to dither "
                       "it. Wire it to a preview to see which parts of the shot need the precision.")
    FUNCTION = "run"
    CATEGORY = CATEGORY_ANDRO
    OUTPUT_NODE = True
    SEARCH_ALIASES = ["ImageRangeStats", "Image Range Stats", "range stats", "measure range",
                      "precision probe", "how many levels", "bit depth check"]
    DESCRIPTION = ("Measure a batch: range, percentiles, how much sits outside [0,1], and how many "
                   "distinct levels survive in a narrow window. bfloat16 gives 77 there; float32 "
                   "gives millions. Passes the image through, so it can sit mid-chain.")

    def run(self, image, label):
        report = _range_report(image, label)
        arr = image.detach().float().cpu().numpy()
        a = arr.ravel()

        step = float("nan")
        window = a[(a > 0.2) & (a < 0.3)]
        if window.size > 16:
            step, uniq = _quantisation_step(window)
            bits = _effective_bits(step)
            report += (f"\n{_SLOT}precision probe in [0.2,0.3]: {uniq} distinct values, "
                       f"smallest step {step:.3e} = {bits:.1f} EFFECTIVE BITS "
                       f"(bfloat16 gives 10.0 / 77 values, 8-bit gives 8.0 / ~26)")
            # A window can only hold as many distinct values as it has samples. When the count
            # comes close to that ceiling, the step being measured is the picture's, not the
            # format's - saying "21 bits" there would be measuring the sample count.
            if uniq > window.size * 0.5:
                report += (f"\n{_SLOT}  ^ SATURATED: {uniq} distinct values out of {window.size} "
                           f"samples in the window - the grid is fuller than the frame can show, "
                           f"so this is a floor on the precision, not a measurement of it")

        risk_pct = 0.0
        mask = np.zeros(arr.shape[:3], np.float32)
        if arr.ndim == 4 and np.isfinite(step):
            for i in range(arr.shape[0]):
                risk, pct = _banding_risk(_luma(arr[i]), step)
                mask[i] = risk.astype(np.float32)
                risk_pct = max(risk_pct, pct)
            if risk_pct >= 0.5:
                report += (f"\n{_SLOT}banding risk: {risk_pct:.1f}% of the worst frame carries a "
                           f"gradient with less local noise than the quantisation step - nothing "
                           f"dithers it, so that is where bands appear first under a grade")
            else:
                report += (f"\n{_SLOT}banding risk: {risk_pct:.1f}% - this material dithers itself "
                           f"(noise, texture), so quantisation stays invisible here. float32 buys "
                           f"headroom for a second pass, not visible fidelity on this shot")

        report += f"\n{_SLOT}dtype={image.dtype} shape={tuple(image.shape)}"
        logger.info("[vae_float32] %s", report.replace("\n", " "))
        return {"ui": {"text": [report]},
                "result": (image, report, _histogram_image(a), torch.from_numpy(mask))}


class ANDROCompare:
    """Numeric A/B of two batches: how far apart, where, and by how much.

    For answering "did that setting actually change anything, and is the change
    real or just noise" without exporting and diffing by hand.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_a": ("IMAGE", {
                    "tooltip": "First batch. Frame counts may differ - the shorter one decides how "
                               "many frames are compared - but the frame SIZE must match."}),
                "image_b": ("IMAGE", {
                    "tooltip": "Second batch, compared against A. The usual pairing is one decode "
                               "with keep_out_of_range ON and one with it OFF, or float32 against "
                               "'vae default' - same seed, one setting changed, so whatever the "
                               "report shows can only have come from that setting."}),
                "amplify": ("FLOAT", {
                    "default": 20.0, "min": 1.0, "max": 200.0, "step": 1.0,
                    "tooltip": "Gain on the difference image, because the raw difference is almost "
                               "always too dark to see. Be warned what this is worth: the clamp/"
                               "precision differences this pack is about stayed BLACK at 25x, and "
                               "only became visible around 100-200x - and there they sit on sky, "
                               "haze and smooth gradients, i.e. exactly where banding shows up "
                               "first. Trust the numbers in the report over this picture."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "MASK")
    RETURN_NAMES = ("difference", "report", "worst_zones")
    OUTPUT_TOOLTIPS = ("|A-B| multiplied by amplify - black means identical, and it stays black "
                       "for differences too small to see even at high gain.",
                       "max/mean deviation, PSNR and SSIM, where in the frame the difference "
                       "actually sits, the five worst frames, and a verdict on whether float32 is "
                       "worth it for THIS shot.",
                       "The tiles where the difference is concentrated. A high global PSNR with a "
                       "small bright patch here means the damage is real but local - usually sky "
                       "or haze, which is exactly where banding starts.")
    FUNCTION = "run"
    CATEGORY = CATEGORY_ANDRO
    OUTPUT_NODE = True
    SEARCH_ALIASES = ["ImageCompareNumeric", "Image Compare", "compare images", "a/b compare",
                      "difference", "psnr", "diff two images"]
    DESCRIPTION = ("Numeric A/B of two batches: max and mean deviation, PSNR, which frame is worst, "
                   "plus an amplified difference image. Answers 'did that setting change anything, "
                   "and is the change real' without exporting and diffing by hand.")

    def run(self, image_a, image_b, amplify):
        a = image_a.detach().float().cpu().numpy()
        b = image_b.detach().float().cpu().numpy()
        n = min(a.shape[0], b.shape[0])
        if a.shape[1:] != b.shape[1:]:
            msg = f"shape mismatch: {a.shape[1:]} vs {b.shape[1:]} - cannot compare"
            logger.warning("[vae_float32] %s", msg)
            return {"ui": {"text": [msg]},
                    "result": (image_a, msg, torch.zeros(a.shape[:3], dtype=torch.float32))}

        d = a[:n] - b[:n]
        ad = np.abs(d)
        differing = float((ad > 0).mean() * 100.0)
        mse = float((d ** 2).mean())
        psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")
        per_frame = ad.reshape(n, -1).mean(axis=1)
        worst_frames = [int(i) + 1 for i in np.argsort(per_frame)[::-1][:5]]

        lines = [
            f"A vs B over {n} frame(s)",
            f"{_SLOT}max |diff| = {ad.max():.6f}   mean |diff| = {ad.mean():.6e}",
            f"{_SLOT}differing samples: {differing:.3f}%   PSNR = {psnr:.2f} dB",
            f"{_SLOT}A: {a.min():+.5f}..{a.max():+.5f}   B: {b.min():+.5f}..{b.max():+.5f}",
        ]

        # SSIM on the worst frame, not the average: PSNR is error ENERGY, so a difference smeared
        # thinly over the whole frame and one concentrated in the sky score the same. SSIM answers
        # the second question - did the STRUCTURE change - which is what banding actually is.
        ref = worst_frames[0] - 1
        ssim = _ssim(_luma(a[ref]), _luma(b[ref]))
        lines.append(f"{_SLOT}SSIM on the worst frame ({worst_frames[0]}): {ssim:.6f}")
        # Only worth printing when the frames actually differ from each other. On a uniform
        # difference the sort order is arbitrary, and listing it as "the worst five" invites the
        # reader to find a pattern in what is really just descending frame numbers.
        spread = float(per_frame.max() - per_frame.min())
        if n > 1 and spread > per_frame.mean() * 0.05:
            lines.append(f"{_SLOT}worst 5 frames by mean |diff|: {worst_frames} - evenly spaced "
                         f"numbers here mean a tiling seam rather than a precision difference")
        elif n > 1:
            lines.append(f"{_SLOT}all {n} frames differ by the same amount (spread {spread:.2e}) - "
                         f"no single frame is worse, so this is not a seam")

        # Where the difference sits. A single global number cannot tell "spread evenly over the
        # frame" from "all of it in one patch of sky", and only the second is worth acting on.
        tiles, mask = 8, np.zeros(a.shape[:3], np.float32)
        h, w = a.shape[1], a.shape[2]
        cell = ad[ref].mean(axis=2)
        grid = np.zeros((tiles, tiles), np.float64)
        for ty in range(tiles):
            for tx in range(tiles):
                y0, y1 = ty * h // tiles, (ty + 1) * h // tiles
                x0, x1 = tx * w // tiles, (tx + 1) * w // tiles
                grid[ty, tx] = cell[y0:y1, x0:x1].mean()
        hottest = float(grid.max())
        if hottest > 0:
            concentration = hottest / max(float(grid.mean()), 1e-12)
            ty, tx = np.unravel_index(int(np.argmax(grid)), grid.shape)
            lines.append(f"{_SLOT}hottest zone: tile ({tx + 1},{ty + 1}) of {tiles}x{tiles}, "
                         f"{concentration:.1f}x the frame average")
            if concentration > 3.0:
                lines.append(f"{_SLOT}  ^ CONCENTRATED, not spread: the global PSNR above is "
                             f"diluted by clean areas and understates what happens in that zone")
            # The mask marks every tile at least half as bad as the worst one.
            for ty in range(tiles):
                for tx in range(tiles):
                    if grid[ty, tx] >= hottest * 0.5:
                        y0, y1 = ty * h // tiles, (ty + 1) * h // tiles
                        x0, x1 = tx * w // tiles, (tx + 1) * w // tiles
                        mask[:, y0:y1, x0:x1] = 1.0

        # The verdict this node exists for. Thresholds come from what this pack measured rather
        # than from theory: the clamp/precision difference stayed invisible at 25x amplification
        # and only showed at 100-200x, so "big enough to see" starts around 1/255.
        if ad.max() == 0:
            lines.append(f"{_SLOT}VERDICT: bit-identical - whatever was changed did not reach the "
                         f"pixels")
        elif ad.max() < 1.0 / 4096:
            lines.append(f"{_SLOT}VERDICT: difference is below a 12-bit step. Real, but no delivery "
                         f"format carries it - keep the simpler path")
        elif ssim > 0.999 and hottest > 0 and hottest / max(float(grid.mean()), 1e-12) < 3.0:
            lines.append(f"{_SLOT}VERDICT: small and evenly spread. Survives a normal grade; float32 "
                         f"is insurance for a second pass, not a visible win on this shot")
        else:
            lines.append(f"{_SLOT}VERDICT: structural and localised - this is the case where the "
                         f"cheaper path shows up in the picture, usually as banding in the flat "
                         f"areas. Worth the float32 decode here")

        report = "\n".join(lines)
        logger.info("[vae_float32] %s", report.replace("\n", " "))
        vis = np.clip(np.abs(d) * amplify, 0.0, 1.0)
        return {"ui": {"text": [report]},
                "result": (torch.from_numpy(vis), report, torch.from_numpy(mask))}


class ANDROSeamCheck:
    """Find the artefacts tiled decoding leaves behind.

    Temporal tiling is the dangerous one: a diffusion decoder has no context at a
    temporal tile edge, so the blend leaves a visibly SOFTER frame on every seam -
    periodic, and easy to miss. Comparing consecutive frames does not catch it,
    because the softening is smooth rather than a jump. This measures per-frame
    sharpness and looks for local dips, then checks for vertical/horizontal seams
    from spatial tiling.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "The decoded sequence to inspect. Needs enough frames for a pattern "
                               "to show - a handful of frames cannot prove periodicity."}),
                "dip_threshold": ("FLOAT", {
                    "default": 0.93, "min": 0.5, "max": 0.999, "step": 0.005,
                    "tooltip": "A frame is called soft when its sharpness drops below this fraction "
                               "of the previous frame's - 0.93 means a 7% fall. Lower it to catch "
                               "fainter seams at the cost of flagging ordinary motion blur; raise "
                               "it if a fast-moving shot reports dips everywhere. Isolated dips are "
                               "usually content: only REGULARLY spaced ones are a tiling seam, and "
                               "the report says which it found."}),
                "check_spatial": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Also look for vertical/horizontal seams left by SPATIAL tiling. "
                               "Only regular, repeating peaks are reported - a single strong edge "
                               "is content, not a seam."}),
            },
            "optional": {
                "predict_from_temporal_size": ("INT", {
                    "default": 0, "min": 0, "max": 4096, "step": 4,
                    "tooltip": "Set this to the temporal_size the decode used and the node will "
                               "PREDICT where soft frames must land, instead of only finding them "
                               "afterwards. The spacing is arithmetic, not luck: tile_t = "
                               "temporal_size / the VAE's temporal compression, and a seam falls "
                               "every (tile_t - overlap_t) x compression frames. 0 = predict "
                               "nothing, just measure what is in the batch."}),
                "predict_from_temporal_overlap": ("INT", {
                    "default": 8, "min": 0, "max": 4096, "step": 4,
                    "tooltip": "The temporal_overlap that went with it. Only read when "
                               "predict_from_temporal_size is not 0."}),
                "temporal_compression": ("INT", {
                    "default": 8, "min": 1, "max": 64, "step": 1,
                    "tooltip": "How many output frames the VAE packs into one latent frame - 8 for "
                               "LTX-2.5. Wrong value here shifts the predicted spacing by the same "
                               "factor, so check it against your VAE rather than assuming."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("sharpness_plot", "report")
    OUTPUT_TOOLTIPS = ("Per-frame sharpness as a plot, with the soft frames marked - a temporal "
                       "seam shows up as evenly spaced dips.",
                       "Which frames went soft, whether their spacing is regular enough to be a "
                       "tiling seam rather than motion, and what to change if it is.")
    FUNCTION = "run"
    CATEGORY = CATEGORY_ANDRO
    OUTPUT_NODE = True
    SEARCH_ALIASES = ["TileSeamCheck", "Tile Seam Check", "seam check", "tiling artefacts",
                      "soft frames", "temporal tiling", "blurry every n frames"]
    DESCRIPTION = ("Find what tiled decoding left behind. Temporal tiling is the dangerous one: the "
                   "decoder has no context at a tile edge, so every seam gets a softer frame - "
                   "periodic and easy to miss. Comparing neighbouring frames does not catch it.")

    def run(self, images, dip_threshold, check_spatial, predict_from_temporal_size=0,
            predict_from_temporal_overlap=8, temporal_compression=8):
        a = images.detach().float().cpu().numpy()
        n = a.shape[0]
        lines = []

        predicted = []
        if predict_from_temporal_size:
            # Same arithmetic the decoder does (see ANDROVAEDecode.decode), so the answer is
            # derived from the settings rather than fitted to the pictures: a temporal tile is
            # temporal_size/compression latent frames, consecutive tiles advance by tile minus
            # overlap, and every advance lands one blended - visibly softer - output frame.
            t_comp = max(1, int(temporal_compression))
            t_size = max(2, predict_from_temporal_size // t_comp)
            t_over = max(1, min(t_size // 2, predict_from_temporal_overlap // t_comp))
            step = max(1, (t_size - t_over) * t_comp)
            if predict_from_temporal_size >= n * t_comp:
                lines.append(f"prediction: temporal_size {predict_from_temporal_size} covers all "
                             f"{n} frames in one tile - no temporal seam is possible here")
            else:
                predicted = list(range(step + 1, n + 1, step))
                lines.append(f"prediction: tile_t={t_size} latent frames, overlap_t={t_over} -> a "
                             f"soft frame every {step} output frames, i.e. at {predicted[:8]}"
                             f"{' ...' if len(predicted) > 8 else ''}")
                lines.append(f"{_SLOT}raise temporal_size (or cut tile_size instead) to remove "
                             f"them - overlap only softens a seam, it never removes it")

        sharp = np.array([_sharpness(_luma(a[i])) for i in range(n)])
        med = float(np.median(sharp)) if n else 0.0
        dips = [i + 1 for i in range(1, n - 1) if sharp[i] < sharp[i - 1] * dip_threshold]
        lines.append(f"temporal: {n} frame(s), median sharpness {med:.5f}")
        if dips:
            lines.append(f"{_SLOT}soft frames at {dips}")
            # Motion in the shot produces isolated soft frames too, so do not demand that EVERY gap
            # matches. Look for the largest subset sitting on one regular grid instead.
            best = (0, None, None)
            for i, start in enumerate(dips):
                for period in {int(d) for d in np.diff(dips)} | {24}:
                    if period < 4:
                        continue
                    on_grid = [f for f in dips[i:] if abs((f - start) % period) <= 1
                               or abs((f - start) % period - period) <= 1]
                    if len(on_grid) > best[0]:
                        best = (len(on_grid), period, on_grid)
            count, period, on_grid = best
            if count >= 3:
                stray = [f for f in dips if f not in on_grid]
                lines.append(f"{_SLOT}PERIODIC: {count} of them every {period} frames "
                             f"({on_grid}) - that is a temporal tiling seam. Raise temporal_size "
                             f"(or turn temporal tiling off) and cut tile_size instead.")
                if stray:
                    lines.append(f"{_SLOT}off-grid, most likely motion in the shot: {stray}")
            else:
                lines.append(f"{_SLOT}no regular spacing - these look like content, not tiling")
        else:
            lines.append(f"{_SLOT}no sudden softening found")

        # Prediction and measurement are worth confronting: agreement means the settings explain
        # the softness, and disagreement is information too - a prediction with nothing measured
        # says the seam is below this threshold, not that it is absent.
        if predicted:
            hit = [f for f in predicted if any(abs(f - d) <= 1 for d in dips)]
            if hit and len(hit) >= max(1, len(predicted) // 2):
                lines.append(f"{_SLOT}CONFIRMED: {len(hit)} of {len(predicted)} predicted seam "
                             f"frames actually went soft - the temporal tiling is the cause")
            elif hit:
                lines.append(f"{_SLOT}partly confirmed: {len(hit)} of {len(predicted)} predicted "
                             f"frames went soft; the rest may be hidden by motion in the shot")
            else:
                lines.append(f"{_SLOT}predicted seams were NOT measurable at dip_threshold "
                             f"{dip_threshold} - either the blend is gentler than the threshold or "
                             f"the compression figure is wrong for this VAE")

        if check_spatial and n:
            # A single strong column means nothing - a hard edge in the CONTENT does that. A tiling
            # seam is REGULAR, so only report one when several excess peaks share a spacing.
            take = a[: min(n, 8)]
            gx = np.stack([np.abs(np.diff(_luma(f), axis=1)).mean(axis=0) for f in take]).mean(axis=0)
            gy = np.stack([np.abs(np.diff(_luma(f), axis=0)).mean(axis=1) for f in take]).mean(axis=0)
            for axis, sig in (("x", gx), ("y", gy)):
                k = 9
                smooth = np.convolve(sig, np.ones(k) / k, mode="same")
                excess = sig / np.maximum(smooth, 1e-9)
                excess[:6] = excess[-6:] = 1.0                 # frame border is not a seam
                peaks = [int(i) for i in np.where(excess > 1.30)[0]]
                merged = []
                for p in peaks:                                # collapse adjacent columns
                    if not merged or p - merged[-1] > 4:
                        merged.append(p)
                spacing = None
                if len(merged) >= 3:
                    gaps = np.diff(merged)
                    if gaps.std() < 3.0 and gaps.mean() > 16:
                        spacing = float(gaps.mean())
                peak = float(excess.max())
                if spacing:
                    lines.append(f"spatial {axis}: SEAM - {len(merged)} regular peaks every "
                                 f"{spacing:.0f}px (max {peak:.2f}x). Raise overlap or tile_size.")
                else:
                    lines.append(f"spatial {axis}: no periodic seam "
                                 f"(max local excess {peak:.2f}x, {len(merged)} isolated peaks - "
                                 f"content edges look like this too)")

        report = "\n".join(lines)
        logger.info("[vae_float32] %s", report.replace("\n", " "))
        plot = _plot(sharp / max(med, 1e-9), marks=[d - 1 for d in dips])
        return {"ui": {"text": [report]}, "result": (plot, report)}


# --------------------------------------------------------------------------- output


class ANDRORemapRange:
    """Fit out-of-range values into [0,1] on purpose, instead of letting a clamp do it.

    An 8/10-bit writer clips anything above 1.0. When that matters - a highlight
    you want to keep - map it down first and decide how.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "The batch to fit into [0,1]. Only meaningful on an unclamped "
                               "decode - a clamped one has nothing left outside the range."}),
                "mode": (["clip", "scale to fit", "reinhard highlights", "filmic rolloff",
                          "report only"],
                         {"default": "filmic rolloff",
                          "tooltip": "clip = what the stock path does anyway, everything above 1.0 "
                                     "becomes 1.0. scale to fit = keep every value, move the whole "
                                     "range into [0,1] - nothing is lost but the whole picture "
                                     "loses contrast, including the parts that were fine. reinhard "
                                     "highlights = compress what is above 1.0 only. filmic rolloff "
                                     "= leave everything below the knee untouched and bend only "
                                     "the top into the remaining headroom, which is what a film "
                                     "shoulder does and what a colourist expects. report only = "
                                     "measure, change nothing."}),
            },
            "optional": {
                "knee": ("FLOAT", {
                    "default": 0.8, "min": 0.1, "max": 1.0, "step": 0.01,
                    "tooltip": "Where the filmic rolloff starts bending. Below it the picture is "
                               "passed through untouched - that is the point: mid-tones and skin "
                               "keep their contrast while only the top gets compressed. 0.8 leaves "
                               "the top fifth of the range to absorb the overshoot."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "report")
    OUTPUT_TOOLTIPS = ("The remapped image, safe to hand to an 8/10-bit writer.",
                       "The range before and after, so it is on record what the fit cost.")
    FUNCTION = "run"
    CATEGORY = CATEGORY_ANDRO
    SEARCH_ALIASES = ["RemapRange", "Remap Range", "remap", "tone map", "fit to 0-1",
                      "highlight rolloff", "reinhard"]
    DESCRIPTION = ("Fit out-of-range values into [0,1] deliberately instead of letting a clamp do "
                   "it. Use before any 8/10-bit writer when a highlight above 1.0 is worth keeping.")

    def run(self, image, mode, knee=0.8):
        x = image.detach().float()
        lo, hi = float(x.min()), float(x.max())
        above = float((x > 1.0).float().mean() * 100.0)
        below = float((x < 0.0).float().mean() * 100.0)

        if mode == "clip":
            out = x.clamp(0.0, 1.0)
        elif mode == "scale to fit":
            span = max(hi - min(lo, 0.0), 1e-9)
            out = (x - min(lo, 0.0)) / span if (hi > 1.0 or lo < 0.0) else x
        elif mode == "reinhard highlights":
            out = torch.where(x > 1.0, 1.0 + torch.log1p(x - 1.0) * 0.25, x).clamp(min=0.0)
            out = out / max(float(out.max()), 1.0)
        elif mode == "filmic rolloff":
            # Everything under the knee passes through EXACTLY, and the range above it is bent
            # asymptotically into what headroom is left. Chosen over a global scale because a
            # scale pays for a few overshooting samples by flattening the entire picture -
            # measured overshoot is 0.01-0.34% of samples, so that is a poor trade.
            k = min(max(float(knee), 0.01), 0.999)
            headroom = 1.0 - k
            if hi > k:
                # Reinhard with a white point, applied to the range ABOVE the knee only. An
                # exponential shoulder was the obvious choice and is wrong here: it approaches 1.0
                # asymptotically, so the brightest sample landed at 0.94 and the white point turned
                # grey. This form maps the input maximum to exactly 1.0 - at u = W it reduces to
                # (W+1)/(1+W) - while still leaving everything below the knee untouched.
                u = ((x - k) / headroom).clamp(min=0.0)
                w = (hi - k) / headroom
                bent = k + headroom * (u * (1.0 + u / (w * w)) / (1.0 + u))
                out = torch.where(x > k, bent, x).clamp(min=0.0)
            else:
                out = x.clamp(min=0.0)
        else:
            out = x

        new_lo, new_hi = float(out.min()), float(out.max())
        lines = [f"remap [{mode}]: {lo:+.5f}..{hi:+.5f} -> {new_lo:+.5f}..{new_hi:+.5f}"]
        if above or below:
            lines.append(f"{_SLOT}input carried {above:.4f}% above 1.0 and {below:.4f}% below 0.0")

        # The price, in stops, because that is the unit the decision is actually made in.
        if hi > 1.0:
            stops_in = float(np.log2(max(hi, 1e-9)))
            stops_kept = float(np.log2(max(new_hi, 1e-9))) if new_hi > 0 else float("-inf")
            lines.append(f"{_SLOT}highlights: input peaked {stops_in:+.2f} stops above 1.0, "
                         f"output peaks {stops_kept:+.2f}")
        touched = float((torch.abs(out - x) > 1e-6).float().mean() * 100.0)
        lines.append(f"{_SLOT}cost: {touched:.2f}% of samples were moved")
        if mode == "scale to fit" and hi > 1.0:
            lines.append(f"{_SLOT}  ^ EVERY sample moved, including the correctly exposed ones - "
                         f"that is what a global scale costs. 'filmic rolloff' pays only in the "
                         f"highlights")
        if mode == "clip":
            lost = above + below
            lines.append(f"{_SLOT}  ^ {lost:.4f}% of samples were flattened to the limits and are "
                         f"now unrecoverable - identical to what stock ComfyUI does")

        report = "\n".join(lines)
        logger.info("[vae_float32] %s", report.replace("\n", " "))
        return (out, report)


# What a colourspace label means in the two terms a reader actually needs, plus the standard
# EXR chromaticities for those primaries. This is a LABEL: nothing here converts a pixel. An
# untagged EXR is read as linear by every compositor that opens it, which for a decoded SDR
# frame is wrong twice - the transfer curve, and for ACES material the primaries as well.
_CHROMATICITIES = {
    "Rec.709":           (0.640, 0.330, 0.300, 0.600, 0.150, 0.060, 0.3127, 0.3290),
    "ACES AP1":          (0.713, 0.293, 0.165, 0.830, 0.128, 0.044, 0.32168, 0.33767),
    "ACES AP0":          (0.7347, 0.2653, 0.0, 1.0, 0.0001, -0.0770, 0.32168, 0.33767),
    "ARRI Wide Gamut 3": (0.6840, 0.3130, 0.2210, 0.8480, 0.0861, -0.1020, 0.3127, 0.3290),
}

# colorspace label -> (transfer function, primaries)
_COLORSPACES = {
    "srgb_display":   ("sRGB piecewise",    "Rec.709"),
    "rec709_display": ("BT.1886 gamma 2.4", "Rec.709"),
    "linear_rec709":  ("linear",            "Rec.709"),
    "acescg":         ("linear",            "ACES AP1"),
    "acescct":        ("ACEScct log",       "ACES AP1"),
    "aces2065_1":     ("linear",            "ACES AP0"),
    "logc3":          ("ARRI LogC3 EI800",  "ARRI Wide Gamut 3"),
    "unspecified":    ("unspecified",       "unspecified"),
}


def _write_exr(path, rgb, half, attrs=None, clipped=None):
    """Write one HxWx3 float array as EXR. Returns the backend used.

    cv2's EXR codec is compiled in but disabled unless OPENCV_IO_ENABLE_OPENEXR=1
    was set BEFORE cv2 was imported (opencv/opencv#21326), which no ComfyUI
    launcher does - cv2.imwrite then writes nothing while returning quietly. The
    OpenEXR module has no such gate, so it goes first.

    attrs are written as custom header attributes, so the file records how its own
    pixels were made. clipped, when given, goes in as a second layer holding exactly
    what the stock [0,1] clamp would have deleted - the loss travels WITH the picture
    instead of living in a screenshot someone has to be shown.

    Only the OpenEXR backend supports either; the cv2 and tifffile fallbacks write the
    picture alone, and the caller is told which backend ran.
    """
    data = np.ascontiguousarray(rgb.astype(np.float16 if half else np.float32))
    try:
        import OpenEXR
        header = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}
        header.update(attrs or {})
        channels = {"RGB": data}
        if clipped is not None:
            channels["clipped"] = np.ascontiguousarray(
                clipped.astype(np.float16 if half else np.float32))
        with OpenEXR.File(header, channels) as f:
            f.write(path)
        return "OpenEXR"
    except ImportError:
        pass
    try:
        import cv2
        exr_type = cv2.IMWRITE_EXR_TYPE_HALF if half else cv2.IMWRITE_EXR_TYPE_FLOAT
        ok = cv2.imwrite(path, data[:, :, ::-1].copy(),
                         [int(cv2.IMWRITE_EXR_TYPE), int(exr_type),
                          int(cv2.IMWRITE_EXR_COMPRESSION), int(cv2.IMWRITE_EXR_COMPRESSION_ZIP)])
        if ok and os.path.exists(path):
            return "cv2"
    except Exception:
        pass
    import tifffile                                   # last resort: 32-bit float TIFF
    tifffile.imwrite(os.path.splitext(path)[0] + ".tif", data)
    return "tifffile(.tif)"


class ANDROSaveEXR:
    """Write an IMAGE batch as a float32 EXR sequence, values untouched."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "The batch to write. Values are written exactly as they arrive - "
                               "nothing is clamped, scaled or tone-mapped on the way out."}),
                "filename_prefix": ("STRING", {
                    "default": "float32/frame",
                    "tooltip": "Subfolder and file stem. Frames are numbered from 1 as "
                               "stem.00001.exr, so 'float32/frame' gives float32/frame.00001.exr."}),
                "half_float": ("BOOLEAN", {
                    "default": False, "label_on": "16f (half)", "label_off": "32f (full)",
                    "tooltip": "16f halves the file size and still carries everything outside "
                               "[0,1] - what it costs is precision, roughly 11 bits of mantissa "
                               "against 24. That is far above bfloat16, so a 16f EXR of a float32 "
                               "decode keeps most of what the decode won. Use 32f when the frames "
                               "are going into a grade that will stretch them hard, 16f when disk "
                               "or a downstream tool argues."}),
            },
            "optional": {
                "output_folder": ("STRING", {
                    "default": "",
                    "tooltip": "Absolute path to write to, e.g. D:/renders/shot_04. Empty means the "
                               "ComfyUI output directory. Use this to drop a sequence straight into "
                               "a project folder instead of fishing it out of output/ later; "
                               "filename_prefix still applies inside it."}),
                "write_metadata": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Record how these pixels were made in the EXR header: measured "
                               "range, how much sat outside [0,1], bit depth, and - if you wire "
                               "the decode report in - the dtypes the decode actually ran in. Six "
                               "months later the file still answers 'was this the float32 pass?' "
                               "on its own. Needs the OpenEXR backend; the TIFF fallback drops it."}),
                "decode_report": ("STRING", {
                    "default": "", "forceInput": True,
                    "tooltip": "Wire ANDRO VAE Decode's range_report here and its contents are "
                               "stored in the file's header. Optional - without it the metadata "
                               "still carries everything measurable from the pixels themselves."}),
                "clipped_layer": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Add a second EXR layer named 'clipped' holding exactly what the "
                               "stock [0,1] clamp would have deleted - zero everywhere it would "
                               "have kept the value. The evidence then travels inside the file "
                               "instead of in a screenshot. Costs roughly double the file size, "
                               "and readers that ignore extra layers are unaffected."}),
                "colorspace": (list(_COLORSPACES.keys()), {
                    "default": "srgb_display",
                    "tooltip": "What these pixels ARE, written into the header. A decoded SDR "
                               "generation is display-referred sRGB/Rec.709 gamma, not linear - "
                               "that is what most VAEs produce (SD, Flux, Wan, LTX SDR), so "
                               "srgb_display is the honest default. Two exceptions: LTX-2.5 HDR "
                               "decodes are ACEScct, and the LTX-2.3 HDR IC-LoRA is LogC3. This "
                               "writes a label - no pixel is converted."}),
                "colorspace_note": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "Free text appended to the label, e.g. 'Flux.2 decode' or 'after "
                               "OCIO ColorSpace to ACEScg'."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("folder",)
    OUTPUT_TOOLTIPS = ("The folder actually written to - wire it onward to a node that reads the "
                       "sequence back.",)
    FUNCTION = "save"
    CATEGORY = CATEGORY_ANDRO
    OUTPUT_NODE = True
    SEARCH_ALIASES = ["SaveEXRFloat32", "Save EXR float32", "save exr", "write exr", "exr sequence",
                      "openexr", "save 32 bit"]
    DESCRIPTION = ("Write an IMAGE batch as a real float32 EXR sequence, values untouched - "
                   "including everything outside [0,1]. Falls back to a 32-bit float TIFF if no "
                   "EXR backend is available, and refuses to claim a write it cannot verify.")

    def save(self, images, filename_prefix, half_float, output_folder="", write_metadata=True,
             decode_report="", clipped_layer=False, colorspace="srgb_display",
             colorspace_note=""):
        base = output_folder.strip() or folder_paths.get_output_directory()
        full = os.path.join(base, filename_prefix)
        folder, stem = os.path.dirname(full), os.path.basename(full) or "frame"
        os.makedirs(folder, exist_ok=True)

        arr = images.detach().float().cpu().numpy()
        backend, written = None, 0

        attrs = None
        if write_metadata:
            lo_pct = float((arr < 0.0).mean() * 100.0)
            hi_pct = float((arr > 1.0).mean() * 100.0)
            attrs = {
                "andro/writer": "comfyui-vae-float32 (ANDRO Save EXR)",
                "andro/bitDepth": "16f" if half_float else "32f",
                "andro/range": f"{arr.min():+.6f} .. {arr.max():+.6f}",
                "andro/outsideUnitRange": f"{lo_pct:.4f}% below 0, {hi_pct:.4f}% above 1",
                "andro/frames": str(arr.shape[0]),
            }
            if decode_report.strip():
                # Stored verbatim: it already states the dtypes the decode ran in and what the
                # clamp would have removed, and paraphrasing it here would let the two drift.
                attrs["andro/decodeReport"] = decode_report.strip()

            # The file stops being silent about what its numbers mean. The andro/* strings are
            # readable by anyone; the standard chromaticities attribute is the part a compositor
            # can act on. Neither touches a pixel - the transfer curve is stated, not applied.
            note = colorspace_note.strip()
            transfer, primaries = _COLORSPACES.get(colorspace, _COLORSPACES["unspecified"])
            attrs["andro/colorspace"] = f"{colorspace} - {note}" if note else colorspace
            attrs["andro/transfer"] = transfer
            attrs["andro/primaries"] = primaries
            if primaries in _CHROMATICITIES:
                # A plain 8-float tuple, and nothing else: the OpenEXR 3.x binding refuses a list
                # or a numpy array here (and its error message misnames it a "6-tuple").
                attrs["chromaticities"] = _CHROMATICITIES[primaries]

        # A 121-frame EXR sequence is a slow, silent stretch otherwise - the node just looks hung.
        progress = comfy.utils.ProgressBar(arr.shape[0])
        for i in range(arr.shape[0]):
            path = os.path.join(folder, f"{stem}.{i + 1:05d}.exr")
            # What the stock clamp would have deleted, and zero wherever it would have kept the
            # value - so the layer is literally the loss, not a second copy of the picture.
            clipped = (arr[i] - np.clip(arr[i], 0.0, 1.0)) if clipped_layer else None
            backend = _write_exr(path, arr[i], half_float, attrs=attrs, clipped=clipped)
            written += 1
            progress.update(1)
            if i == 0:                                  # never report an unverified write
                probe = path if os.path.exists(path) else os.path.splitext(path)[0] + ".tif"
                if not os.path.exists(probe) or os.path.getsize(probe) == 0:
                    raise RuntimeError(
                        f"EXR write produced no file at {path} (backend {backend}). Install the "
                        "OpenEXR python module, or start ComfyUI with OPENCV_IO_ENABLE_OPENEXR=1.")

        lo = float((arr < 0.0).mean() * 100.0)
        hi = float((arr > 1.0).mean() * 100.0)
        msg = (f"wrote {written} frame(s) via {backend} "
               f"({'16f' if half_float else '32f'}) to {folder}\n"
               f"{_SLOT}range carried: min={arr.min():+.6f} max={arr.max():+.6f} "
               f"({lo:.4f}% below 0, {hi:.4f}% above 1)")
        if backend != "OpenEXR" and (write_metadata or clipped_layer):
            msg += (f"\n{_SLOT}note: metadata and the clipped layer need the OpenEXR backend - "
                    f"{backend} wrote the picture only")
        elif write_metadata:
            msg += f"\n{_SLOT}header carries the provenance attributes (andro/*)"
        if clipped_layer and backend == "OpenEXR":
            msg += f"\n{_SLOT}second layer 'clipped' holds what the stock clamp would have deleted"
        if write_metadata:
            cs_note = colorspace_note.strip()
            cs_transfer, cs_primaries = _COLORSPACES.get(colorspace,
                                                         _COLORSPACES["unspecified"])
            cs_label = f"{colorspace} - {cs_note}" if cs_note else colorspace
            msg += (f"\n{_SLOT}colorspace: {cs_label} | transfer: {cs_transfer} | "
                    f"primaries: {cs_primaries}")
        else:
            msg += f"\n{_SLOT}colorspace: not written (write_metadata off)"
        logger.info("[vae_float32] %s", msg.replace("\n", " "))
        return {"ui": {"text": [msg]}, "result": (folder,)}


class ANDROAudioSwitch:
    """Pick between an optional latent and a fallback - without a required wire.

    Written for LTX's audio branch, where LTXVConcatAVLatent demands an audio
    latent: muting the LoadAudio -> encode chain breaks the graph, and a plain
    boolean switch does not help either, because ComfyUI validates every node in
    the prompt before execution - a missing wav still fails the run through
    dependent_outputs even with the switch pointing elsewhere.

    Taking the optional branch as an OPTIONAL input solves it properly: mute that
    chain and the input is simply gone, leaving the fallback. Nothing to validate,
    no placeholder file. Useful anywhere an input is mandatory but you want it to
    be skippable.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio_source": ("BOOLEAN", {
                    "default": True, "label_on": "external", "label_off": "generated",
                    "tooltip": "Which branch wins when both are connected. Switching is enough — "
                               "there is no need to mute anything. If only one branch is connected, "
                               "that one is used whatever this says."}),
            },
            "optional": {
                "generated_audio": ("LATENT", {
                    "lazy": True,
                    "tooltip": "The latent the model composes on its own - for LTX that is an empty "
                               "audio latent. Lazy: while the switch points at the other branch, "
                               "this one is never computed at all, so it costs nothing to leave "
                               "wired up."}),
                "external_audio": ("LATENT", {
                    "lazy": True,
                    "tooltip": "Your own audio, encoded to a latent. Mute or delete its chain and "
                               "this input simply disappears."}),
            },
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "mode")
    OUTPUT_TOOLTIPS = ("Whichever branch won.",
                       "Which branch that was, and whether it was the one the switch asked for - "
                       "so a fallback is visible in the log instead of silent.")
    FUNCTION = "pick"
    CATEGORY = CATEGORY_ANDRO
    SEARCH_ALIASES = ["AudioLatentSwitch", "Audio Latent Switch", "latent switch", "audio switch",
                      "optional latent", "switch without muting", "generated or external audio"]
    DESCRIPTION = ("Pick between two latents with one toggle, no muting required. Both inputs are "
                   "optional and lazy: the branch that loses is never computed, and either side "
                   "can be muted or deleted without breaking the graph.")

    def check_lazy_status(self, audio_source, generated_audio=_MISSING, external_audio=_MISSING):
        """Ask for one branch only, so the other is never computed.

        A lazy input arrives as None while it is connected but unevaluated, which is
        why absent inputs default to _MISSING here rather than None - otherwise the
        two cases are indistinguishable.
        """
        wanted, other = (("external_audio", "generated_audio") if audio_source
                         else ("generated_audio", "external_audio"))
        values = {"generated_audio": generated_audio, "external_audio": external_audio}
        if values[wanted] is _MISSING:              # the preferred branch is not wired at all
            return [other] if values[other] is None else []
        return [wanted] if values[wanted] is None else []

    def pick(self, audio_source, generated_audio=_MISSING, external_audio=_MISSING):
        # Both branches optional on purpose: either can be muted, bypassed or deleted and the
        # survivor is used. Requiring one of them made disabling THAT side fail with ComfyUI's
        # "missing a required input", which reads like a broken node rather than a muted branch.
        wanted, other = (("external", "generated") if audio_source else ("generated", "external"))
        values = {"generated": generated_audio, "external": external_audio}
        got = {k: v for k, v in values.items() if v is not _MISSING and v is not None}

        if wanted in got:
            mode, out = wanted, got[wanted]
        elif other in got:
            mode, out = f"{other} ({wanted} not connected)", got[other]
        else:
            raise RuntimeError(
                "Audio Latent Switch: neither generated_audio nor external_audio is connected. "
                "Connect one of them, or un-mute the branch you meant to use.")

        logger.info("[vae_float32] latent switch: %s", mode)
        return (out, mode)


class ANDROLoadAudio:
    """Load Audio that treats a missing file as silence instead of killing the prompt.

    ComfyUI validates every node in a prompt before any of it runs, and the stock
    LoadAudio rejects a filename that is not in the input folder. So a graph that
    merely CONTAINS an audio branch cannot run without the file - even when a
    switch downstream was never going to use it, and even when the branch is lazy
    (measured: a lazy input still fails validation). That is what forces the
    mute-the-whole-chain ritual, and why one un-muted node in the chain breaks it.

    The fix is to own the validation. VALIDATE_INPUTS taking `audio_file` makes
    ComfyUI skip its own combo check for that widget (execution.py:1019), so an
    unknown filename passes, reaches execute(), and turns into silence plus a line
    in the report. The branch then costs nothing and breaks nothing when unused.
    """

    NONE = "(none - silence)"

    @classmethod
    def INPUT_TYPES(s):
        try:
            files = folder_paths.filter_files_content_types(
                os.listdir(folder_paths.get_input_directory()), ["audio", "video"])
        except Exception:
            files = []
        return {
            "required": {
                "audio_file": ([s.NONE] + sorted(files), {
                    "tooltip": "A file from the input folder. A name that does not exist here - on "
                               "another machine, say - yields silence and a note, not an error."}),
                "silence_seconds": ("FLOAT", {
                    "default": 5.0, "min": 0.1, "max": 3600.0, "step": 0.1,
                    "tooltip": "How long the substituted silence is when no file is selected or the "
                               "file is missing. Match it to the clip you are generating - some "
                               "audio-conditioned models size their latent from the audio, so "
                               "silence that is too short can shorten the result."}),
                "sample_rate": ("INT", {
                    "default": 48000, "min": 8000, "max": 192000, "step": 1000,
                    "tooltip": "Sample rate of the GENERATED SILENCE only - a real file always keeps "
                               "its own rate, whatever this says, and the report states which rate "
                               "was actually loaded. Set it to what the downstream audio encoder "
                               "expects (48000 for LTX)."}),
            },
            "optional": {
                "path_override": ("STRING", {
                    "default": "",
                    "tooltip": "Absolute path, for audio that lives outside the input folder. "
                               "Wins over audio_file when set."}),
                "resample_to_sample_rate": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Resample a loaded file to the sample_rate above. Off by default "
                               "because resampling is never free and most graphs do not need it - "
                               "but when the downstream encoder wants one rate and the file has "
                               "another, the mismatch otherwise surfaces as a confusing failure "
                               "much further down the graph. The report always states both rates, "
                               "whether or not this is on."}),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "report")
    OUTPUT_TOOLTIPS = ("The loaded audio, or silence of the requested length if there was no file.",
                       "Which file was loaded and its length / rate / channels - or why silence "
                       "was substituted. Nothing is ever swapped for silence quietly.")
    FUNCTION = "load"
    CATEGORY = CATEGORY_ANDRO
    SEARCH_ALIASES = ["LoadAudioOptional", "Load Audio (optional)", "load audio optional",
                      "optional audio", "missing wav", "audio without breaking the graph"]
    DESCRIPTION = ("Load Audio that treats a missing file as silence instead of killing the prompt. "
                   "ComfyUI validates every node before anything runs, so the stock LoadAudio makes "
                   "an unused audio branch fail the whole graph; this one owns its validation.")

    @classmethod
    def VALIDATE_INPUTS(cls, audio_file):
        # Deliberately permissive - see the class docstring. Naming the widget here is
        # what makes ComfyUI hand its combo check over to us.
        return True

    def _silence(self, seconds, sample_rate):
        n = max(1, int(round(seconds * sample_rate)))
        return {"waveform": torch.zeros(1, 2, n), "sample_rate": int(sample_rate)}

    def load(self, audio_file, silence_seconds, sample_rate, path_override="",
             resample_to_sample_rate=False):
        wanted = path_override.strip() or ("" if audio_file == self.NONE else audio_file)
        if not wanted:
            report = f"no file selected - {silence_seconds:g}s of silence at {sample_rate} Hz"
            logger.info("[vae_float32] %s", report)
            return (self._silence(silence_seconds, sample_rate), report)

        path = wanted
        if not os.path.isabs(path):
            try:
                path = folder_paths.get_annotated_filepath(wanted)
            except Exception:
                path = os.path.join(folder_paths.get_input_directory(), wanted)

        if not os.path.exists(path):
            report = (f"'{wanted}' not found - {silence_seconds:g}s of silence instead. "
                      f"Nothing failed; pick a file, or leave it if this branch is unused.")
            logger.warning("[vae_float32] %s", report)
            return (self._silence(silence_seconds, sample_rate), report)

        try:
            from comfy_extras.nodes_audio import load as _load     # same decoder as stock LoadAudio
            waveform, sr = _load(path)
        except Exception as e:
            report = f"could not decode '{wanted}' ({type(e).__name__}: {e}) - using silence"
            logger.warning("[vae_float32] %s", report)
            return (self._silence(silence_seconds, sample_rate), report)

        secs = waveform.shape[-1] / float(sr)
        lines = [f"loaded '{os.path.basename(path)}': {secs:.2f}s, {sr} Hz, {waveform.shape[0]}ch"]

        if sr != int(sample_rate):
            # Stated either way. Audio length frequently decides clip length, and a rate mismatch
            # that is silently carried downstream fails somewhere far from its cause.
            if resample_to_sample_rate:
                try:
                    import torchaudio.functional as AF
                    waveform = AF.resample(waveform, sr, int(sample_rate))
                    old_sr, sr = sr, int(sample_rate)
                    secs = waveform.shape[-1] / float(sr)
                    lines.append(f"{_SLOT}resampled {old_sr} -> {sr} Hz, now {secs:.2f}s")
                except Exception as e:
                    lines.append(f"{_SLOT}resample to {sample_rate} Hz FAILED "
                                 f"({type(e).__name__}: {e}) - passing the original {sr} Hz through")
                    logger.warning("[vae_float32] %s", lines[-1])
            else:
                lines.append(f"{_SLOT}note: file is {sr} Hz but sample_rate says {sample_rate} - "
                             f"the file's own rate is passed on. Turn resample_to_sample_rate on "
                             f"if the encoder downstream needs {sample_rate}")

        report = "\n".join(lines)
        logger.info("[vae_float32] %s", report.replace("\n", " "))
        return ({"waveform": waveform.unsqueeze(0), "sample_rate": sr}, report)


# The keys here are what a saved workflow stores, not the display names - renaming one silently breaks
# every graph that used it. Hence _RENAMED below: the old keys stay registered forever, pointing at
# DEPRECATED subclasses so they keep loading old workflows without cluttering the node menu.
NODE_CLASS_MAPPINGS = {
    "ANDROVAEDecode": ANDROVAEDecode,
    "ANDROLoadAudio": ANDROLoadAudio,
    "ANDROAudioSwitch": ANDROAudioSwitch,
    "ANDROVAEEncode": ANDROVAEEncode,
    "ANDRORangeStats": ANDRORangeStats,
    "ANDROCompare": ANDROCompare,
    "ANDROSeamCheck": ANDROSeamCheck,
    "ANDRORemapRange": ANDRORemapRange,
    "ANDROSaveEXR": ANDROSaveEXR,
}

# Second reason for the prefix, besides making the pack recognisable: these keys live in ONE global
# namespace shared by every installed pack, and the originals were generic enough to collide. Any
# other pack registering "RemapRange" or "ImageRangeStats" would have silently replaced ours,
# whichever loaded last.
NODE_DISPLAY_NAME_MAPPINGS = {
    "ANDROVAEDecode": "ANDRO VAE Decode",
    "ANDROAudioSwitch": "ANDRO Audio Switch",
    "ANDROLoadAudio": "ANDRO Load Audio",
    "ANDROVAEEncode": "ANDRO VAE Encode",
    "ANDRORangeStats": "ANDRO Range Stats",
    "ANDROCompare": "ANDRO Compare",
    "ANDROSeamCheck": "ANDRO Seam Check",
    "ANDRORemapRange": "ANDRO Remap Range",
    "ANDROSaveEXR": "ANDRO Save EXR",
}

# old key -> new class, for workflows saved before 1.3.0.
_RENAMED = {
    "VAEDecodeFloat32": ANDROVAEDecode,
    "VAEEncodeFloat32": ANDROVAEEncode,
    "ImageRangeStats": ANDRORangeStats,
    "ImageCompareNumeric": ANDROCompare,
    "TileSeamCheck": ANDROSeamCheck,
    "RemapRange": ANDRORemapRange,
    "SaveEXRFloat32": ANDROSaveEXR,
    "LoadAudioOptional": ANDROLoadAudio,
    "AudioLatentSwitch": ANDROAudioSwitch,
}

for _old, _new in _RENAMED.items():
    # A subclass rather than the class itself: DEPRECATED is read off the class (server.py's
    # node_info), so pointing both keys at one object would mark the current node deprecated too.
    # SEARCH_ALIASES emptied on the alias: the aliases are already on the current node, and leaving
    # them here would show the retired copy alongside it in every search that matches.
    NODE_CLASS_MAPPINGS[_old] = type(_old, (_new,), {"DEPRECATED": True, "SEARCH_ALIASES": []})
    # Same display name as the current node, with nothing appended. web/andromedia_color.js rewrites
    # these types as a graph loads, so a canvas user never meets the alias at all; the name only shows
    # up where that migration cannot run - an API prompt posted straight to /prompt with the old key.
    # Labelling it "(old name)" there just decorates a node that is working correctly.
    NODE_DISPLAY_NAME_MAPPINGS[_old] = NODE_DISPLAY_NAME_MAPPINGS[_new.__name__]
