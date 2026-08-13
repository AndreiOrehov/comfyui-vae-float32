<div align="center">

<img src="docs/assets/cover.png" width="880" alt="comfyui-vae-float32 - distinct levels per frame, stock decode vs float32, across six VAEs">

# comfyui-vae-float32

**Every VAE decode in ComfyUI throws away two things, and neither is visible from inside a graph.**
<br>
**This pack gives them back - and gives you the measurements to check that on your own models.**

**By [Andromediastudio](https://andromediastudio.com/).**

![License: MIT](https://img.shields.io/badge/License-MIT-FFD27D.svg)
![ComfyUI](https://img.shields.io/badge/ComfyUI-custom_nodes-5BAEE3.svg)
![Nodes](https://img.shields.io/badge/8_nodes-decode_·_measure_·_EXR-9aa3b2.svg)
![Verified on](https://img.shields.io/badge/verified_on-6_VAEs-3fb950.svg)

</div>

---

## What is lost

### 1. Everything outside [0,1]

`comfy/sd.py:502` finishes every decode with

```python
process_output = lambda image: image.add_(1.0).div_(2.0).clamp_(0.0, 1.0)
```

Decoders routinely emit values past those bounds. On a real LTX-2.5 generation the decode spans
**−0.0715 … +1.0445**: specular highlights and shadow detail, deleted before any node downstream can
see them.

<div align="center">
<img src="docs/assets/clamp_range.png" width="880" alt="Decoded value range per VAE against the [0,1] clamp bounds">
</div>

### 2. Most of the precision

Most VAEs run in bfloat16. Distinct values across `[0.2, 0.3]` of one frame:

| decode | distinct levels | smallest step |
|---|---|---|
| stock (bfloat16) | **77** | 9.77e-04 (1/1024) |
| float32 | **3 354 786** | 2.98e-08 |

bf16 and fp32 decodes of the same latent differ by up to **0.0186** — roughly five steps of an 8-bit
scale. No float32 container recovers this after the fact; the precision has to exist at decode time.

This is not LTX-specific. Same still, same probe, stock vs this pack:

| VAE | stock | ours |
|---|---|---|
| `ltx-2.5-video-vae-bf16` | 77 | 3 354 786 |
| `ae.safetensors` (Flux) | 77 | 223 109 |
| `qwen_image_vae` | 77 | 214 995 |
| `wan2.2_vae` | 77 | 212 585 |
| `hunyuanvideo15_vae_fp16` | 614 | 213 523 |
| `taeltx2_3` (TAEHV) | 77 | 229 150 |

---

## The nodes

All under the **`vae_float32`** category.

### VAE Decode (float32, no clamp)

Drop-in replacement for `VAEDecode` / `VAEDecodeTiled`. Temporarily swaps `vae.process_output` for the
same maths minus the clamp, optionally runs the decoder in float32, and restores both in `finally` —
other graphs in the session are unaffected.

It does **not** assume every VAE uses the `[-1,1] → [0,1]` default. TAEHV / lighttae (`sd.py:894, 906`),
MiniMax H3 (`976`) and StageA (`540`) already emit `[0,1]` and set identity; substituting the default
there would rescale the image and wreck it. The node probes the VAE's own transform with `-1/0/1` and
only replaces shapes it recognises — anything unfamiliar is left alone and reported.

Turn `keep_out_of_range` off to reproduce stock ComfyUI exactly.

### VAE Encode (float32)

The mirror image. Stock encode casts your pixels to the VAE's working dtype (usually bf16) before the
weights see them, so feeding it a float32 plate discards the precision at the door.

### Image Range Stats

How much of a batch is outside `[0,1]`, and how finely it is quantised. Every number in this README
came from this node.

### Image Compare (numeric)

Two batches in, metrics out: max and mean absolute difference, percentage of differing samples, PSNR,
which frame deviates most — plus an amplified difference image. For answering "did that setting change
anything, and is the change real" without exporting and diffing by hand.

### Tile Seam Check

Tiled decoding leaves artefacts, and the temporal kind is easy to miss. A diffusion decoder has no
context at a temporal tile edge, so the blend leaves a visibly **softer frame on every seam** — smooth,
not a jump, which is exactly why a frame-to-frame difference check does not see it.

This node measures per-frame sharpness, finds local dips, and reports only when several of them sit on
one regular grid — motion in the shot produces isolated soft frames too. Real output from a bad setting:

```
temporal: 121 frame(s), median sharpness 0.00738
  soft frames at [12, 25, 49, 66, 73, 97]
  PERIODIC: 4 of them every 24 frames ([25, 49, 73, 97]) - that is a temporal tiling seam.
  off-grid, most likely motion in the shot: [12, 66]
```

Same generation, temporal tiling off: `soft frames at [66] - no regular spacing, these look like
content`. It also checks for spatial seams, again requiring regularity rather than a single strong
column, because a hard edge in the content looks identical to one.

### Remap Range

An 8/10-bit writer clips whatever sits above 1.0. When that matters, map it down deliberately —
`clip`, `scale to fit`, `reinhard highlights`, or `report only`.

### Save EXR (float32)

Writes an EXR sequence through the **OpenEXR module**, not cv2, and verifies the file landed rather
than trusting the writer.

> OpenCV compiles the EXR codec in but leaves it **disabled** unless `OPENCV_IO_ENABLE_OPENEXR=1` is
> set before `cv2` is imported ([opencv#21326](https://github.com/opencv/opencv/issues/21326)). No
> ComfyUI launcher sets it, so any node writing EXR through `cv2.imwrite` silently produces nothing.
> If your pack does that, it is worth checking — this one bit ComfyUI-OCIO too
> ([fix](https://github.com/SlavaSexton/ComfyUI-OCIO/pull/5)).

### Latent Switch (optional input)

Feeds a fallback latent when an optional one is absent. Written for LTX's audio branch, where
`LTXVConcatAVLatent` requires an audio latent — muting the `LoadAudio → encode` chain breaks the graph,
and a plain boolean switch does not help either, because ComfyUI validates every node in the prompt
before execution: a missing wav still kills the run through `dependent_outputs`. An **optional** input
solves it properly — mute the chain and there is nothing left to validate.

---

## Settings that matter

**Tile space, never time.** Measured on a 121-frame 1280×704 clip, fp32 decode, RTX 5090:

| tile_size | temporal_size | wall clock | result |
|---|---|---|---|
| 768 | 4096 | **912 s** | exceeds 32 GB VRAM → dynamic-VRAM offload, crawls |
| 768 | 32 | 60 s | ⚠️ soft frame every 24 frames |
| **384** | **4096** | **60 s** | clean |

The 24-frame period is arithmetic: `tile_t = 32/8 = 4` latent frames, `overlap_t = 1`, so the step is
`3` latent = 24 pixel frames. Cutting the **spatial** tile is what actually solves the memory wall, and
it costs nothing: gradient excess at the spatial tile boundaries measures 1.03–1.05×, i.e. no seam.

**float32 costs about 2.7× the decode time and 2× the VAE's VRAM.** On the clip above that was still
60 s, because the spatial tile was the real constraint.

---

## Start here

**[`example_workflows/01_measure_your_vae.json`](example_workflows/01_measure_your_vae.json)** —
drop it in, point `VAELoader` at any VAE you already have, hit Run. It round-trips one image through
that VAE and decodes the latent twice, stock and float32, side by side:

```
LoadImage ─► VAE Encode (float32) ─┬─► VAE Decode (float32, no clamp) ─┬─► Image Range Stats
                                   │                                   ├─► Save EXR (float32)
                                   │                                   └─► Image Compare ◄─┐
                                   └─► VAEDecode (stock ComfyUI) ─────────► Image Range Stats
```

The two Range Stats readouts are the whole point: same latent, same VAE, and one of them has a few
hundred thousand more levels than the other. No LTX, no video model, nothing to download.

Copy `example_workflows/neon_cyborg_portrait.png` into your `ComfyUI/input/` folder first, or point
`LoadImage` at anything you already have; the shipped `VAELoader` value is `qwen_image_vae.safetensors`,
swap it for a VAE you own. Verified run, exactly as the file ships:

```
ours  - float32, no clamp: min=-0.019557 max=+1.018609   506 744 levels in [0.2,0.3]
stock - ComfyUI decode:    min=+0.000000 max=+1.000000        77 levels
Compare: max |diff| = 0.151855, PSNR 58.14 dB
```

The heavier LTX-2.5 graphs in the same folder show the pack inside a real video pipeline, including
the EXR sequence and the audio switch.

## Install

Clone into `ComfyUI/custom_nodes/` and restart. `pip install OpenEXR` if you want EXR output; the pack
falls back to 32-bit float TIFF otherwise.

## Requirements

`OpenEXR>=3.3`, `tifffile`, `numpy`. Everything else comes with ComfyUI.

## A caveat worth stating

This pack reaches into `vae.process_output`, `vae.vae_dtype` and `first_stage_model` — none of which
are public API. A ComfyUI refactor can break it. The `-1/0/1` probe and the guarded fp32 cast are there
so that a surprise degrades into "no change, with a note in the report" rather than a corrupted image,
but if something looks wrong, compare against stock with **Image Compare (numeric)** first.

## Licence

MIT.

## Screens

![stock vs float32](docs/assets/stock_vs_float32.png)

Both halves come from one latent in a single run: the only difference is the decode path.
