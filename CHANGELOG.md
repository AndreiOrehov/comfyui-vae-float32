# Changelog

## 1.0.0 - first release

Seven nodes around one finding: ComfyUI's VAE decode clamps to `[0,1]` and runs in bfloat16, and
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
