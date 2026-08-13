# Tiling benchmark — LTX-2.5 video VAE

Decode only, no sampler: an empty latent through `VAEDecodeFloat32`, `temporal_size 4096`,
`overlap 64`, VAE preloaded by a warmup run. Wall clock from `POST /prompt` to history completion.

Rig: RTX 5090 32 GB, ComfyUI 0.32.0, Windows 11, PyTorch 2.12.1+cu130.
Reproduce: `python bench_tiling.py [--width W --height H --length N] [--only <label>]`

## 1280×704, 121 frames — latent [1,128,16,22,40]

| precision | tile_size | decode | |
|---|---|---|---|
| bf16 | 384 | 12.1 s | |
| bf16 | 768 | 12.1 s | ComfyUI's official `video_ltx2_5_i2v` template |
| float32 | 384 | 44.2 s | this pack's default |
| float32 | 512 | 42.4 s | stock `VAEDecodeTiled` default |
| float32 | 768 | **1247 s** | 28× — no longer fits, the loader starts paging |

## 1920×1088, 121 frames — latent [1,128,16,34,60]

| precision | tile_size | decode |
|---|---|---|
| float32 | 384 | 109.1 s |
| float32 | 512 | 100.8 s |
| float32 | 768 | not measured — the 1280×704 case already took 21 minutes |

## What the numbers say

- In bf16 the tile size is free. In float32 it is a cliff, and the cliff sits between 512 and 768.
- 2.32× the pixels costs ~2.4× the time: the tile sets the memory ceiling, the frame size then
  spends it linearly. The cliff did not move down at the larger frame.
- 512 is consistently a few percent faster than 384 at both resolutions — fewer tiles, less overlap
  recomputed.
- `EmptyLTXVLatentVideo` floor-divides by 32, so a height of 1080 silently becomes 1056. Use 1088.
