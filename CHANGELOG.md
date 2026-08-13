# Changelog

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
