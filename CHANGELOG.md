# Changelog

## 1.4.1 - no processes spawned

The Registry flagged 1.4.0: `ANDRO Video QC` probed the source file by spawning `ffprobe`, and
custom nodes are not allowed to start processes. The probe now reads the container through PyAV -
the library ComfyUI decodes video with - and reports the same fields (codec, pixel format and bit
depth, size, frame rate, frame count, colour space / transfer / primaries / range, untagged
list). FFmpeg's colour enums are mapped to their names; unspecified stays "unknown". The sidecar
manifest's host name now comes from `platform.node()` rather than the socket module. No other
behaviour changed.

## 1.4.0 - the frame knows where it came from, and the plate gets checked before it is trusted

### ANDRO Video QC (new node)

An ingest report for generated video, the checks a pipeline TD runs before a plate is accepted:
effective bit depth (with the container's word winning over a level count that the YUV-to-RGB
decode inflates), range and NaN, black and flash frames, exact duplicates, held frames (a frame that
barely moves against the clip's own median), flicker, resolution / frame count / fps against a spec,
and the colour tags of the source file when `source_path` is given - an untagged file is a WARN,
because every reader will then assume BT.601. Outputs a Markdown report, the same as JSON, a labelled
contact sheet of the flagged frames, and a `pass` boolean to gate the rest of the graph. Measured on
two API-generated clips: both untagged, one with 22 % held frames in a period-4 pattern - a 24 fps
file carrying about 6 fps of motion. `docs/NODES_QC.md`.

### ANDRO Save EXR: frame numbering and a provenance manifest

- `start_frame` (1) and `padding` (5): the defaults reproduce today's `shot.00001.exr` exactly, so
  saved graphs keep writing the same names; `1001` / `4` gives the VFX-standard `shot.1001.exr`.
- The header now records how the frame was made: `andro/created`, `andro/comfyVersion`,
  `andro/packVersion`, your `shot_info`, and a summary read straight out of the graph ComfyUI hands
  the node - every model file (`andro/models`), every seed with its node (`andro/seeds`), the text
  prompts (`andro/prompts`), the full settings of any cloud API node (`andro/apiNodes`), and a
  `andro/workflowHash` that is the same for the same graph whatever the canvas layout. With
  `embed_workflow` on (default) the whole workflow and API graph travel in the header too; turn it
  off for confidential graphs and the summary stays. A sidecar `<stem>.manifest.json` carries the
  same next to the sequence. The workstation name goes only into the sidecar, never into a frame
  that may leave the building.

## 1.3.1 - the EXR says what colour it is

`ANDRO Save EXR` gains `colorspace` and `colorspace_note`. Until now the file was silent about its own
encoding, and an untagged EXR is read as **linear** by every compositor that opens it - which for a decoded
SDR frame is wrong, because what a VAE returns is display-referred sRGB/Rec.709 gamma.

The default is therefore `srgb_display`, the truth for SD, Flux, Wan and LTX SDR decodes. The exceptions are
named in the tooltip: LTX-2.5 HDR decodes are `acescct`, the LTX-2.3 HDR IC-LoRA is `logc3`. Also available:
`rec709_display`, `linear_rec709`, `acescg`, `aces2065_1`, `unspecified`.

The header now carries `andro/colorspace`, `andro/transfer`, `andro/primaries` and - for every known gamut -
the **standard OpenEXR `chromaticities` attribute**, so a reader can act on it instead of guessing. The node's
report states the encoding it wrote.

**Nothing converts.** The pixels are bit-identical to 1.3.0 (verified: max abs diff 0.0 on a round-trip);
Nuke or Resolve must still be told the same colourspace on input. New widgets are appended last, so saved
graphs keep their widget values.

## 1.3.0 - the pack gets a name, and the measurements get opinions

### Every node renamed, and why your old graphs still work

All nine nodes now carry the `ANDRO` prefix, sit in an `ANDRO` category, and share one colour on the
canvas, so it is obvious at a glance which nodes in a graph belong to this pack. `VAE Decode (float32,
no clamp)` is now `ANDRO VAE Decode`; the explanations that used to live in the titles moved into
tooltips, where they belong.

This was not only cosmetic. `NODE_CLASS_MAPPINGS` is one namespace shared by every installed pack, and
the old keys were generic enough to collide — any other pack registering `RemapRange` or
`ImageRangeStats` would have silently replaced ours, whichever loaded last.

**Nothing breaks.** The nine old keys stay registered as deprecated aliases, so a graph saved on 1.2.x
loads. On top of that, a workflow's node types are rewritten to the new names as it loads, including
inside subgraphs, so the old names never appear on the canvas. Save the workflow once and it is
permanent. Searching for `VAEDecodeFloat32` — or "vae decode no clamp", or "unclamped decode" — still
finds the node, because every old name is registered as a search alias.

### Tiling is on by default, and the cliff is now estimated for your machine

`tiled` ships on. float32 decoding is what makes the whole-frame path run out of VRAM in the first
place, and with the shipped defaults only the spatial split is active, which measures no seam
(gradient excess 1.03–1.05× against a 1.30× detection threshold).

1.2.x warned when `tile_size` went above 512. **That was wrong** — it was one card's number stated as
a law. Where the cliff sits depends on the GPU, the VAE and the frame size. The node now asks
`vae.memory_used_decode()` for one tile, compares it against free VRAM, and reports `fits` / `TIGHT` /
`WILL NOT FIT`. Three states rather than two, because the data does not support a single threshold: a
bf16 768 tile needing 95% of free VRAM ran fine in 12.1 s, while a float32 768 tile needing 191% took
1247 s.

### Measurements that answer questions instead of printing numbers

**ANDRO Range Stats** reports the quantisation step as **effective bits** (bf16 reads 10.0), flags
`SATURATED` when the distinct-value count is limited by the sample count rather than by the format,
and adds a **banding-risk mask** plus a log-scaled histogram. Banding is decided from the physics — a
band is one quantisation level held across several pixels, so it needs a gradient *and* local noise
below the step. Validated on six synthetic cases: an 8-bit ramp scores 29.3%, the same ramp with
dithering noise scores 0%.

**ANDRO Compare** adds SSIM, the worst five frames, and an 8×8 zone map with a `worst_zones` mask,
because a global PSNR cannot tell "spread thinly" from "all of it in the sky". Damage confined to one
corner reads as PSNR 46.4 dB — which looks fine — while the node reports the hottest zone at 17.3× the
frame average. Ends with a verdict on whether float32 is worth it for that shot.

**ANDRO Seam Check** predicts seams from the decode settings before the decode, then confronts the
prediction with what it measured. On the documented LTX-2.5 case it predicts soft frames at
`[25, 49, 73, 97, 121]` and confirms four of them.

### Output

**ANDRO Remap Range** gains `filmic rolloff`, now the default: everything below the knee passes
through bit-exact and only the top is bent, with the input peak mapped to exactly 1.0. On one plate
that moves 11.2% of samples against 100% for `scale to fit`. Every mode now reports its price in
stops.

**ANDRO Save EXR** writes provenance into the header (`andro/*`: range, share outside `[0,1]`, bit
depth, and the decode's own report if wired in) and can add a second layer, `clipped`, holding exactly
what the stock clamp would have deleted. Both need the OpenEXR backend; the TIFF fallback says so
rather than dropping them silently. A progress bar now runs during the write.

**ANDRO Load Audio** states duration, rate and channels, and can resample to the target rate instead
of letting a mismatch fail further down the graph.

### Everything else

Tooltips on all 33 inputs and all 9 outputs, up from 19 and 0. Descriptions on all nine nodes, up from
one.

## 1.2.1 - the one to install

No code change. Drops the `Banner` key from `pyproject.toml`: the Registry accepts it and then never
renders it, so leaving it in only suggests an artwork slot that does not exist. This is also the first
version whose number matches a git tag and a GitHub release, which 1.0.0 through 1.2.0 did not.

## 1.2.0

Same pack as 1.0.0 below, renumbered because the Registry already had a 1.1.0 from earlier the same
day, before the docs were audited and the figures were rebuilt. Registry version numbers cannot be
withdrawn or reused, so the way to make the newest thing the newest number is to go past it. Install
1.2.0; 1.0.0 and 1.1.0 are the same afternoon's work in progress.

## 1.0.0 - first release

Nine nodes around one finding: ComfyUI's VAE decode clamps to `[0,1]` and runs in bfloat16, and
neither loss is visible from inside a graph.

### What the stock path costs

Measured, not inferred. The decode leaves `[0,1]` on every generation looked at here, by anywhere
from **−0.0196 … +1.0186** to **−0.0715 … +1.0445** — 0.01–0.34% of samples — and
`clamp_(0.0, 1.0)` in `comfy/sd.py:502` deletes all of it. Across `[0.2, 0.3]` of one frame the stock
decode holds **77** distinct values (step 1/1024, the bfloat16 grid); the same latent decoded in float32
holds **3 354 786** (step 2.98e-08). The two decodes differ by up to **0.0186** — five steps of an
8-bit scale.

Confirmed on eleven unrelated VAEs (LTX-2.5, Flux, Qwen, Wan 2.1/2.2, HunyuanVideo 1.5, MiniMax H3,
TAEHV), so this is a property of the decode path rather than of one model.

### Two traps this had to avoid

**Not every VAE clamps.** TAEHV / lighttae, MiniMax H3 and StageA set `process_output` to identity
because they already emit `[0,1]`. Substituting the default `(x+1)/2` there would rescale the image.
The decode node probes the VAE's own transform with `-1/0/1` and only replaces what it recognises.

**Setting the weights to float32 is not enough.** `decode()` casts the incoming latent to
`vae.vae_dtype` (`comfy/sd.py:1206`), so a VAE whose weights are already float32 while `vae_dtype`
stays bfloat16 — Flux's `ae.safetensors`, as shipped — quantises the input before the weights ever see
it. Both are now set.

### What tiling costs

`tools/bench_tiling.py`, decode only, empty latent, RTX 5090 32 GB:

| precision | tile_size | 1280×704×121 | 1920×1088×121 |
|---|---|---|---|
| bf16 | 384 | 12.1 s | — |
| bf16 | 768 | 12.1 s | — |
| float32 | 384 | 44.2 s | 109.1 s |
| float32 | 512 | 42.4 s | 100.8 s |
| float32 | 768 | **1247 s** | not measured |

In bf16 the tile size is free. In float32 it is a cliff between 512 and 768: the tile stops fitting
and the loader starts paging. ComfyUI's own `video_ltx2_5_i2v` template ships `VAEDecodeTiled` at 768,
which is the right call for bf16 and twenty minutes in float32. float32 itself costs about 3× the
decode time and 2× the VAE's VRAM.

Tiling time, not space, is the other trap: dropping `temporal_size` to 32 cuts a 912 s decode to 60 s
and introduces a soft frame every 24 — `tile_t=4` latent frames with `overlap_t=1` steps 3 latent
frames, and a diffusion decoder has no context at a temporal tile edge. A frame-to-frame difference
check does not catch it, because the softening is smooth; per-frame sharpness with a regular-grid test
does, which is why **Tile Seam Check** exists. Cutting `tile_size` to 384 reaches the same 60 s with
no artefact at all.

### EXR that actually gets written

OpenCV ships the EXR codec disabled unless `OPENCV_IO_ENABLE_OPENEXR=1` predates the `cv2` import
([opencv#21326](https://github.com/opencv/opencv/issues/21326)), which no ComfyUI launcher arranges —
so `cv2.imwrite` on an `.exr` path writes nothing while reporting success. Saving goes through the
OpenEXR module, and the write is verified on disk before the node reports it. The same bug was fixed
upstream in ComfyUI-OCIO ([PR #5](https://github.com/SlavaSexton/ComfyUI-OCIO/pull/5)).

### The audio branch is one toggle

ComfyUI validates every node in a prompt before any of it runs, and stock `LoadAudio` rejects a
filename that is not in the input folder. A lazy consumer does not escape that either; measured on
0.32.0:

```
ComfySwitchNode(switch=false, on_true=LoadAudio("no_such_file.wav"))
  -> HTTP 400  audio - Invalid audio file: no_such_file.wav
```

So an audio branch could only be disabled by muting every node in it, which fails silently the moment
one node is missed. **Load Audio (optional)** owns its validation instead: an unknown file becomes
silence plus a line in its `report`. **Audio Latent Switch** takes both latents as optional and lazy,
so either side can be muted, bypassed or deleted, and the branch you did not pick is never executed.
One toggle, `audio_source`, decides everything.

Verified by full runs of the example graph:

| toggle | file | result |
|---|---|---|
| generated | present | loader never ran, 121 EXR, 145 s |
| external | present | `loaded the wav: 5.00s, 48000 Hz, 2ch`, 121 EXR, 94 s |
| external | **missing** | accepted, silence, 121 EXR, 84 s |

### Worth knowing before filing a bug

- The selection toolbox's ▶ button runs `Comfy.QueueSelectedOutputNodes`, not Run: only the selected
  output nodes are queued, every other one is dropped, and the run still reports success. Measured
  with `PreviewImage` selected — `outputs_to_execute: ['9']`, and both `Image Range Stats` nodes plus
  `Save EXR (float32)` never executed. A plain Run queues all of them.
- `EmptyLTXVLatentVideo` floor-divides by 32, so a height of 1080 silently becomes 1056.
- The example workflow needs its start image in your `input/` folder. The audio filename it carries
  almost certainly does not exist on your machine, and that is fine by design.

### Licence

Apache-2.0. Section 4(d) makes the [NOTICE](NOTICE) file travel with any copy or derivative, so
redistributing the pack keeps the authorship visible. Using the nodes in your own work requires
nothing and costs nothing. Commits made before this release carried an MIT header; the release and
everything after it are Apache-2.0.
