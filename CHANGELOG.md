# Changelog

## 1.1.0 - naming, and what tiling actually costs

### Licence: MIT -> Apache-2.0

Same freedoms, one addition: section 4(d) requires the new [NOTICE](NOTICE) file to travel with any
copy or derivative, so redistributing the pack keeps the authorship visible. Using the nodes in your
own work still costs nothing and requires no credit. Every source file also carries an SPDX header
now, because files get copied one at a time and LICENSE does not follow them.

Releases up to and including the MIT-licensed 1.0.0 keep those terms for anyone who already took them.

### Breaking: Audio Latent Switch inputs renamed

Promoted onto a subgraph, `prefer_external: fallback | external` says nothing — the label lands in a
column of video settings with no hint that it is about audio, and the real toggle sits on the outer
node while the identical-looking one inside the subgraph is inert. Renamed for the reading, not the
mechanics:

| 1.0.0 | 1.1.0 |
|---|---|
| `generated` | `generated_audio` |
| `external` | `external_audio` |
| `prefer_external` (`fallback` / `external`) | `audio_source` (`generated` / `external`) |

Input names are part of the contract, so a 1.0.0 workflow fails with `missing a required input` until
it is migrated: `python tools/migrate_switch_names.py <workflow-or-folder>` rewrites both the UI and
API formats and leaves a `.bak`. A promoted subgraph input keeps its original name — remove the
promotion and re-add it to pick up the new one.

### New: Load Audio (optional) — ninth node

Stock `LoadAudio` fails validation on a filename that is not in the input folder, and ComfyUI
validates every node in a prompt before any of it runs. Marking the consumer lazy does not help;
measured on 0.32.0, a lazy branch is validated all the same:

```
ComfySwitchNode(switch=false, on_true=LoadAudio("no_such_file.wav"))
  -> HTTP 400  audio - Invalid audio file: no_such_file.wav
```

So an audio branch could only be disabled by muting every node in it — a ritual that fails silently
when one node is missed (`TrimAudioDuration: Required input is missing: audio`). This node owns its
validation instead: an unknown file becomes silence plus a line in its `report` output. Same picker
and same decoder as stock, plus `path_override` for files outside the input folder.

### The audio branch is now one toggle

- Both switch inputs are **optional**: 1.0.0 required `generated`, so turning *that* side off failed
  with `missing a required input`, which reads like a broken node rather than a disabled branch.
- Both are **lazy**: only the selected branch is executed, so on `generated` the wav is not decoded
  at all.
- Nothing needs muting. The example ships with the whole chain live and `audio_source` as the only
  control; a missing file downgrades to silence instead of failing the run.

### Fixed

- The shipped `LTX2.5_float32_EXR.json` had `LTXVEmptyLatentAudio` **bypassed** and titled
  "(unused - real audio is fed instead)". It feeds the required branch, so the example failed on load
  with `missing a required input`. Bypass forwards an input of the same type to the output, and an
  empty-latent node has no input to forward. Enabled, and retitled to say it must stay that way.

### Measured

`tools/bench_tiling.py`, decode only, empty latent, RTX 5090 32 GB:

| precision | tile_size | 1280×704×121 | 1920×1088×121 |
|---|---|---|---|
| bf16 | 384 | 12.1 s | — |
| bf16 | 768 | 12.1 s | — |
| float32 | 384 | 44.2 s | 109.1 s |
| float32 | 512 | 42.4 s | 100.8 s |
| float32 | 768 | **1247 s** | not measured |

In bf16 the tile size is free; in float32 it is a cliff between 512 and 768. ComfyUI's own
`video_ltx2_5_i2v` template ships `VAEDecodeTiled` at 768 — correct for bf16, twenty minutes in
float32. float32 itself is stated as ~3× the decode time (was "2.7×").

### Documented

- The selection-toolbox ▶ button runs `Comfy.QueueSelectedOutputNodes`, not Run: every other output
  node is dropped and the run still reports success. Measured with `PreviewImage` selected —
  `outputs_to_execute: ['9']`, and both `Image Range Stats` nodes plus `Save EXR (float32)` never
  executed. A plain Run queues all of them.
- `EmptyLTXVLatentVideo` floor-divides by 32, so a height of 1080 silently becomes 1056.

## 1.0.0 - first release

Eight nodes around one finding: ComfyUI's VAE decode clamps to `[0,1]` and runs in bfloat16, and
neither loss is visible from inside a graph.

### What the stock path costs

Measured, not inferred. On a 121-frame LTX-2.5 generation the decode spans **−0.0715 … +1.0445**, and
`clamp_(0.0, 1.0)` in `comfy/sd.py:502` deletes all of it. Across `[0.2, 0.3]` of one frame the stock
decode holds **77** distinct values (step 1/1024, the bfloat16 grid); the same latent decoded in float32
holds **3 354 786** (step 2.98e-08). The two decodes differ by up to **0.0186** — five steps of an
8-bit scale.

Confirmed on six unrelated VAEs (LTX-2.5, Flux, Qwen, Wan 2.2, Hunyuan Video 1.5, TAEHV), so this is a
property of the decode path rather than of one model.

### Two traps this had to avoid

**Not every VAE clamps.** TAEHV / lighttae, MiniMax H3 and StageA set `process_output` to identity
because they already emit `[0,1]`. Substituting the default `(x+1)/2` there would rescale the image.
The decode node probes the VAE's own transform with `-1/0/1` and only replaces what it recognises.

**Setting the weights to float32 is not enough.** `decode()` casts the incoming latent to
`vae.vae_dtype` (`comfy/sd.py:1206`), so a VAE whose weights are already float32 while `vae_dtype`
stays bfloat16 — Flux's `ae.safetensors`, as shipped — quantises the input before the weights ever see
it. Both are now set.

### Tile Seam Check

Included because the obvious way to buy back speed is wrong. Dropping `temporal_size` to 32 cut a
912 s decode to 60 s and introduced a soft frame every 24 — `tile_t=4` latent frames with
`overlap_t=1` steps 3 latent frames, and a diffusion decoder has no context at a temporal tile edge.
A frame-to-frame difference check does not catch it, because the softening is smooth. Per-frame
sharpness with a regular-grid test does.

Cutting `tile_size` to 384 instead reaches the same 60 s with no artefact at all.

### EXR that actually gets written

OpenCV ships the EXR codec disabled unless `OPENCV_IO_ENABLE_OPENEXR=1` predates the `cv2` import
([opencv#21326](https://github.com/opencv/opencv/issues/21326)), which no ComfyUI launcher arranges —
so `cv2.imwrite` on an `.exr` path writes nothing while reporting success. Saving goes through the
OpenEXR module, and the write is verified on disk before the node reports it.
