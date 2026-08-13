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

import folder_paths

logger = logging.getLogger("vae_float32")

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


class VAEDecodeFloat32:
    """VAE Decode that keeps values outside [0,1] and can force a float32 decode."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
                "keep_out_of_range": ("BOOLEAN", {
                    "default": True, "label_on": "keep (no clamp)", "label_off": "clamp to [0,1]",
                    "tooltip": "Off reproduces stock ComfyUI exactly. On keeps the decoder's "
                               "undershoot/overshoot so an EXR can carry it."}),
                "precision": (["vae default", "float32"], {
                    "default": "float32",
                    "tooltip": "Most VAEs run in bfloat16, which quantises the output to roughly 10 "
                               "bits. float32 costs ~3x decode time and 2x VAE VRAM."}),
                "tiled": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Tiled decoding, for frame counts / resolutions that will not fit."}),
            },
            "optional": {
                "tile_size": ("INT", {"default": 384, "min": 64, "max": 4096, "step": 32,
                                      "tooltip": "Cut THIS before reaching for temporal tiling."}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 32}),
                "temporal_size": ("INT", {"default": 4096, "min": 8, "max": 4096, "step": 4,
                                          "tooltip": "Video VAEs only. Leave high: a temporal tile "
                                                     "edge has no context and the blend leaves a "
                                                     "soft frame on every seam. Use Tile Seam Check."}),
                "temporal_overlap": ("INT", {"default": 8, "min": 4, "max": 4096, "step": 4}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "range_report")
    FUNCTION = "decode"
    CATEGORY = "vae_float32"
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


class VAEEncodeFloat32:
    """VAE Encode in float32, and without silently clipping an out-of-range plate.

    The stock encode casts the image to the VAE's working dtype (usually bf16)
    before it reaches the weights. Feeding it a float32 EXR plate throws the
    precision away at the door.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pixels": ("IMAGE",),
                "vae": ("VAE",),
                "precision": (["vae default", "float32"], {"default": "float32"}),
            },
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "report")
    FUNCTION = "encode"
    CATEGORY = "vae_float32"

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


class ImageRangeStats:
    """How much of a batch lives outside [0,1], and how finely it is quantised."""

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",),
                             "label": ("STRING", {"default": "stats"})}}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "report")
    FUNCTION = "run"
    CATEGORY = "vae_float32"
    OUTPUT_NODE = True

    def run(self, image, label):
        report = _range_report(image, label)
        a = image.detach().float().cpu().numpy().ravel()
        window = a[(a > 0.2) & (a < 0.3)]
        if window.size > 16:
            uniq = np.unique(window)
            step = float(np.min(np.diff(uniq))) if uniq.size > 3 else float("nan")
            report += (f"\n{_SLOT}precision probe in [0.2,0.3]: {uniq.size} distinct values, "
                       f"smallest step {step:.3e} (8-bit would give ~26 / step 3.9e-03)")
        report += f"\n{_SLOT}dtype={image.dtype} shape={tuple(image.shape)}"
        logger.info("[vae_float32] %s", report.replace("\n", " "))
        return {"ui": {"text": [report]}, "result": (image, report)}


class ImageCompareNumeric:
    """Numeric A/B of two batches: how far apart, where, and by how much.

    For answering "did that setting actually change anything, and is the change
    real or just noise" without exporting and diffing by hand.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "amplify": ("FLOAT", {"default": 20.0, "min": 1.0, "max": 200.0, "step": 1.0,
                                      "tooltip": "Gain applied to the difference image so small "
                                                 "deviations become visible."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("difference", "report")
    FUNCTION = "run"
    CATEGORY = "vae_float32"
    OUTPUT_NODE = True

    def run(self, image_a, image_b, amplify):
        a = image_a.detach().float().cpu().numpy()
        b = image_b.detach().float().cpu().numpy()
        n = min(a.shape[0], b.shape[0])
        if a.shape[1:] != b.shape[1:]:
            msg = f"shape mismatch: {a.shape[1:]} vs {b.shape[1:]} - cannot compare"
            logger.warning("[vae_float32] %s", msg)
            return {"ui": {"text": [msg]}, "result": (image_a, msg)}

        d = a[:n] - b[:n]
        ad = np.abs(d)
        differing = float((ad > 0).mean() * 100.0)
        mse = float((d ** 2).mean())
        psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")
        worst_frame = int(np.argmax(ad.reshape(n, -1).max(axis=1))) + 1

        report = (
            f"A vs B over {n} frame(s)\n"
            f"{_SLOT}max |diff| = {ad.max():.6f}   mean |diff| = {ad.mean():.6e}\n"
            f"{_SLOT}differing samples: {differing:.3f}%   PSNR = {psnr:.2f} dB\n"
            f"{_SLOT}largest deviation in frame {worst_frame}\n"
            f"{_SLOT}A: {a.min():+.5f}..{a.max():+.5f}   B: {b.min():+.5f}..{b.max():+.5f}"
        )
        logger.info("[vae_float32] %s", report.replace("\n", " "))
        vis = np.clip(np.abs(d) * amplify, 0.0, 1.0)
        return {"ui": {"text": [report]},
                "result": (torch.from_numpy(vis), report)}


class TileSeamCheck:
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
                "images": ("IMAGE",),
                "dip_threshold": ("FLOAT", {"default": 0.93, "min": 0.5, "max": 0.999, "step": 0.005,
                                            "tooltip": "A frame counts as a dip when its sharpness "
                                                       "falls below this fraction of the previous "
                                                       "frame's."}),
                "check_spatial": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("sharpness_plot", "report")
    FUNCTION = "run"
    CATEGORY = "vae_float32"
    OUTPUT_NODE = True

    def run(self, images, dip_threshold, check_spatial):
        a = images.detach().float().cpu().numpy()
        n = a.shape[0]
        lines = []

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


class RemapRange:
    """Fit out-of-range values into [0,1] on purpose, instead of letting a clamp do it.

    An 8/10-bit writer clips anything above 1.0. When that matters - a highlight
    you want to keep - map it down first and decide how.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["clip", "scale to fit", "reinhard highlights", "report only"],
                         {"default": "scale to fit"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "report")
    FUNCTION = "run"
    CATEGORY = "vae_float32"

    def run(self, image, mode):
        x = image.detach().float()
        lo, hi = float(x.min()), float(x.max())
        if mode == "clip":
            out = x.clamp(0.0, 1.0)
        elif mode == "scale to fit":
            span = max(hi - min(lo, 0.0), 1e-9)
            out = (x - min(lo, 0.0)) / span if (hi > 1.0 or lo < 0.0) else x
        elif mode == "reinhard highlights":
            out = torch.where(x > 1.0, 1.0 + torch.log1p(x - 1.0) * 0.25, x).clamp(min=0.0)
            out = out / max(float(out.max()), 1.0)
        else:
            out = x
        report = (f"remap [{mode}]: {lo:+.5f}..{hi:+.5f} -> "
                  f"{float(out.min()):+.5f}..{float(out.max()):+.5f}")
        logger.info("[vae_float32] %s", report)
        return (out, report)


def _write_exr(path, rgb, half):
    """Write one HxWx3 float array as EXR. Returns the backend used.

    cv2's EXR codec is compiled in but disabled unless OPENCV_IO_ENABLE_OPENEXR=1
    was set BEFORE cv2 was imported (opencv/opencv#21326), which no ComfyUI
    launcher does - cv2.imwrite then writes nothing while returning quietly. The
    OpenEXR module has no such gate, so it goes first.
    """
    data = np.ascontiguousarray(rgb.astype(np.float16 if half else np.float32))
    try:
        import OpenEXR
        header = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}
        with OpenEXR.File(header, {"RGB": data}) as f:
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


class SaveEXRFloat32:
    """Write an IMAGE batch as a float32 EXR sequence, values untouched."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "float32/frame"}),
                "half_float": ("BOOLEAN", {
                    "default": False, "label_on": "16f (half)", "label_off": "32f (full)",
                    "tooltip": "Half still carries out-of-range values, with less precision."}),
            },
            "optional": {
                "output_folder": ("STRING", {
                    "default": "", "tooltip": "Absolute path, or empty for the ComfyUI output dir."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("folder",)
    FUNCTION = "save"
    CATEGORY = "vae_float32"
    OUTPUT_NODE = True

    def save(self, images, filename_prefix, half_float, output_folder=""):
        base = output_folder.strip() or folder_paths.get_output_directory()
        full = os.path.join(base, filename_prefix)
        folder, stem = os.path.dirname(full), os.path.basename(full) or "frame"
        os.makedirs(folder, exist_ok=True)

        arr = images.detach().float().cpu().numpy()
        backend, written = None, 0
        for i in range(arr.shape[0]):
            path = os.path.join(folder, f"{stem}.{i + 1:05d}.exr")
            backend = _write_exr(path, arr[i], half_float)
            written += 1
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
        logger.info("[vae_float32] %s", msg.replace("\n", " "))
        return {"ui": {"text": [msg]}, "result": (folder,)}


class AudioLatentSwitch:
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
                    "tooltip": "The latent the model makes on its own — an empty audio latent for "
                               "LTX."}),
                "external_audio": ("LATENT", {
                    "lazy": True,
                    "tooltip": "Your own audio, encoded to a latent. Mute or delete its chain and "
                               "this input simply disappears."}),
            },
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "mode")
    FUNCTION = "pick"
    CATEGORY = "vae_float32"

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


class LoadAudioOptional:
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
                    "tooltip": "Length of the silence used when there is no file."}),
                "sample_rate": ("INT", {"default": 48000, "min": 8000, "max": 192000, "step": 1000}),
            },
            "optional": {
                "path_override": ("STRING", {
                    "default": "",
                    "tooltip": "Absolute path, for audio that lives outside the input folder. "
                               "Wins over audio_file when set."}),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "report")
    FUNCTION = "load"
    CATEGORY = "vae_float32"

    @classmethod
    def VALIDATE_INPUTS(cls, audio_file):
        # Deliberately permissive - see the class docstring. Naming the widget here is
        # what makes ComfyUI hand its combo check over to us.
        return True

    def _silence(self, seconds, sample_rate):
        n = max(1, int(round(seconds * sample_rate)))
        return {"waveform": torch.zeros(1, 2, n), "sample_rate": int(sample_rate)}

    def load(self, audio_file, silence_seconds, sample_rate, path_override=""):
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
        report = f"loaded '{os.path.basename(path)}': {secs:.2f}s, {sr} Hz, {waveform.shape[0]}ch"
        logger.info("[vae_float32] %s", report)
        return ({"waveform": waveform.unsqueeze(0), "sample_rate": sr}, report)


NODE_CLASS_MAPPINGS = {
    "VAEDecodeFloat32": VAEDecodeFloat32,
    "LoadAudioOptional": LoadAudioOptional,
    "AudioLatentSwitch": AudioLatentSwitch,
    "VAEEncodeFloat32": VAEEncodeFloat32,
    "ImageRangeStats": ImageRangeStats,
    "ImageCompareNumeric": ImageCompareNumeric,
    "TileSeamCheck": TileSeamCheck,
    "RemapRange": RemapRange,
    "SaveEXRFloat32": SaveEXRFloat32,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VAEDecodeFloat32": "VAE Decode (float32, no clamp)",
    "AudioLatentSwitch": "Audio Latent Switch (generated / external)",
    "LoadAudioOptional": "Load Audio (optional)",
    "VAEEncodeFloat32": "VAE Encode (float32)",
    "ImageRangeStats": "Image Range Stats",
    "ImageCompareNumeric": "Image Compare (numeric)",
    "TileSeamCheck": "Tile Seam Check",
    "RemapRange": "Remap Range",
    "SaveEXRFloat32": "Save EXR (float32)",
}
