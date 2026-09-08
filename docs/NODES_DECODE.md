# ANDRO VAE Decode and ANDRO VAE Encode

The two nodes that touch the VAE. Everything else in the pack exists to measure what these two do, or to
carry the result somewhere without losing it again.

Wiring map: **[NODES.md](NODES.md)**.

---

## ANDRO VAE Decode

Drop-in replacement for `VAEDecode` and `VAEDecodeTiled`. Same `samples` and `vae` inputs, same `IMAGE` out,
plus a report.

It removes two separate losses that the stock path applies, and both are invisible from inside a graph.

### The clamp

ComfyUI ends every decode with this, at `comfy/sd.py:502`:

```python
process_output = lambda image: image.add_(1.0).div_(2.0).clamp_(0.0, 1.0)
```

The decoder does emit outside `[0,1]`. Measured across five real LTX-2.5 generations, the range ran from
`-0.0196 .. +1.0186` to `-0.0715 .. +1.0445`, with 0.01-0.34% of samples outside the limits depending on the
shot. **The range is a property of the run, not a constant** - do not quote one figure as though it were.

Those samples are specular highlights and shadow detail, deleted before any node downstream can see them.
`keep_out_of_range` keeps them. Turn it off to reproduce stock ComfyUI exactly, which is the honest way to
A/B this pack against the thing it replaces.

The node does **not** assume every VAE uses the `[-1,1] → [0,1]` default. TAEHV and lighttae (`sd.py:894`,
`906`), MiniMax H3 (`976`) and StageA (`540`) already emit `[0,1]` and set identity; substituting the default
there would rescale the image and wreck it. So the node probes the VAE's own transform with `-1/0/1` and only
replaces shapes it recognises. Anything unfamiliar is left alone and said so in the report.

`vae.process_output` and `vae.vae_dtype` are both restored in a `finally`, so other graphs in the same
session are unaffected even if the decode raises.

### The precision

Most VAEs run in bfloat16. That is coarser than it sounds. Across `[0.2, 0.3]` of one frame, bf16 can land on
exactly **77 distinct values**, spaced `1/1024` apart - and that is the format's entire grid in that window,
not a property of the picture. Verified with numpy: 51 representable values below 0.25 at step 1/1024, plus
26 above it at 1/512.

The same window decoded in float32 holds **3 354 786** values, step `2.98e-08`, and the grid is 67% full -
what limits it there is the number of pixels, not the format.

Measured on 11 VAEs; nine of them give exactly 77. The table is in the [README](../README.md).

No float32 container recovers this after the fact. Wrapping bf16 values in a 32-bit file gives empty
precision.

### Tiling, and the cliff

**`tiled` is on by default since 1.3.0.** A float32 decode holds the VAE at twice its usual weight and every
intermediate at four bytes a sample, so the whole-frame path runs out of VRAM at resolutions the stock bf16
decode swallows.

That costs no visible seam, and it is measured rather than hoped for: with the shipped defaults only the
**spatial** split is active, and gradient excess at the spatial tile boundaries comes out at **1.03-1.05x**,
against the 1.30x that `ANDRO Seam Check` needs before it will call something a peak.

The seam people actually hit is the **temporal** one, and it is a different setting. Cutting along time
leaves the decoder with no context at the tile edge, so every seam gets a visibly softer frame. `temporal_size`
therefore ships at 4096 - high enough that nothing is cut along time at all.

Tile size cost is a **cliff, not a slope**. While the decode fits in VRAM the tile size is nearly free; the
moment it stops fitting, weights start paging and the same decode takes tens of times longer, with no error
and no warning. Measured on an RTX 5090 32 GB, LTX-2.5, 1280x704x121:

| precision | tile_size | decode | |
| --- | --- | --- | --- |
| bf16 | 384 | 12.1 s | |
| bf16 | 768 | 12.1 s | ComfyUI's official LTX-2.5 template ships this |
| float32 | 384 | 44.2 s | this pack's default |
| float32 | 512 | 42.4 s | stock `VAEDecodeTiled` default |
| float32 | 768 | **1247 s** | 28x slower |

**Where that cliff sits depends on your card, your VAE and the frame size.** There is no universal safe
number, so the node estimates it for the machine it is running on: it asks `vae.memory_used_decode()` - the
same estimate ComfyUI itself uses to decide how many frames it can batch - for the shape of one tile, and
compares it against `model_management.get_free_memory()`. The report then says one of three things:

| verdict | meaning |
| --- | --- |
| `fits` | under 85% of free VRAM |
| `TIGHT` | 85-100%. It may well run, but if the decode crawls, that is why |
| `WILL NOT FIT` | over free VRAM. Expect paging, and cut `tile_size` |

Three states rather than a yes/no, because a single threshold cannot survive the data. Calibration against
every point this pack has measured (30.3 GiB reported free):

| precision | tile | estimated need | predicted | measured |
| --- | --- | --- | --- | --- |
| float32 | 384 | 14.5 GiB | fits | 44 s |
| float32 | 512 | 25.7 GiB | tight | 42 s |
| float32 | 768 | 57.9 GiB | **spills** | **1247 s** |
| bf16 | 384 | 7.2 GiB | fits | 12.1 s |
| bf16 | 768 | 28.9 GiB | tight | 12.1 s |

That last row is why a hard cut does not work: 28.9 GiB is 95% of free and ran perfectly well in 12 seconds.
A threshold strict enough to condemn the 768 float32 case also condemns that one.

**This is five points from one GPU. It is calibration, not proof.** The free figure is also read before the
model is fully resident, so it is an upper bound. Treat the verdict as a warning, not a guarantee.

### Widgets

| Widget | Default | What it does |
| --- | --- | --- |
| `keep_out_of_range` | on | Off reproduces stock ComfyUI exactly. |
| `precision` | `float32` | `vae default` is bf16 on most VAEs - see above. |
| `tiled` | **on** | Off decodes whole frames; use it for a single still that fits. |
| `tile_size` | 384 | The knob to cut when memory runs out. Cut this before `temporal_size`. |
| `overlap` | 64 | Blended overlap between spatial tiles. 64 against a 384 tile measures no seam. |
| `temporal_size` | 4096 | Leave high. Cutting it is what produces soft frames. |
| `temporal_overlap` | 8 | Does nothing while `temporal_size` is 4096, because there is only one temporal tile. |

### Outputs

- `image:IMAGE` - the decoded batch, still carrying out-of-range values when `keep_out_of_range` is on.
- `range_report:STRING` - measured range, percentiles, what the clamp would have deleted, the dtypes the
  decode really ran in, and the VRAM verdict. Wire it into `ANDRO Save EXR`'s `decode_report` and it is
  stored in the written file's header.

---

## ANDRO VAE Encode

The mirror image, and the answer to "what does encoding actually cost".

**It does not improve the latent's format.** Tracing `comfy/sd.py`, the latent leaves `VAE.encode` as
`vae_output_dtype()` → `intermediate_dtype()` → **float32 already**, regardless of this pack. The only
exception is a ComfyUI started with `--fp16-intermediates`, which makes it float16.

What this node fixes is what happens **on the way in**. The stock path casts your pixels to the VAE's working
dtype before the weights ever see them:

```python
pixels_in = self.process_input(pixel_samples[...]).to(self.vae_dtype)
```

With `vae_dtype` at bf16 - which is the default on most VAEs - a float32 EXR plate is thrown away at the
door. Setting `precision` to `float32` keeps the weights and the incoming pixels in float32 so that does not
happen.

Two more things worth knowing, both visible in the same trace:

- `process_input` is `image * 2.0 - 1.0` for most VAEs, so the input is expected in `[0,1]`. A plate carrying
  1.5 becomes 2.0 and lands outside the VAE's training domain. That is why this node measures the incoming
  range **before** the latent exists, rather than letting you infer it afterwards.
- The alpha channel is dropped: the node passes `pixels[:, :, :, :3]`.

### Widgets

| Widget | Default | What it does |
| --- | --- | --- |
| `precision` | `float32` | `vae default` casts the plate to the VAE's dtype before the weights see it. |

### Outputs

- `latent:LATENT` - the encoded latent.
- `report:STRING` - the range of what was handed in, measured before encoding, plus the latent's shape and
  dtype.
