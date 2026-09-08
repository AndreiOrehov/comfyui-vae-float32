# ANDRO Range Stats, ANDRO Compare and ANDRO Seam Check

The three nodes that measure and change nothing. Every number in this pack's README came out of them.

Wiring map: **[NODES.md](NODES.md)**.

They exist because the losses this pack is about are **not reliably visible**. Pushing the difference by 25x
produced a black frame, twice. On dense noisy material - neon, rain, skin - eight bits survive an aggressive
grade, because the noise dithers the quantisation away. The difference only shows at 100-200x amplification,
and there it sits on sky, haze and smooth gradients. So the honest way to decide anything here is to measure
it, not to look at it.

---

## ANDRO Range Stats

Pass-through: the image comes back out unchanged, so this node sits **in** a chain rather than beside it.

### What the report says

**Range and percentiles**, plus the share outside `[0,1]` that the stock clamp would delete.

**Effective bits.** The precision probe looks at `[0.2, 0.3]`, counts distinct values, takes the smallest gap
between neighbours, and reports `log2(1/step)`. On a bf16 decode that comes out as **77 values, step
9.766e-04, 10.0 effective bits** - matching the figure measured independently with numpy. 8-bit material
gives 8.0 and about 26 values.

**A `SATURATED` warning when the count approaches the number of samples in the window.** A window can only
hold as many distinct values as it has samples, so once the grid is fuller than the frame can show, the step
being measured belongs to the picture rather than to the format. Reporting "21 bits" there would be measuring
the sample count. The line says so instead.

**Banding risk, as a percentage of the worst frame and as a mask.** This is decided against the measured
quantisation step, not against percentiles of the frame - percentiles cannot work here, because on a
perfectly smooth ramp every pixel has the same slope, so "steeper than the 60th percentile" is false
everywhere and the exact case being hunted scores zero. The physics instead:

- a band is one quantisation level held across several pixels, so its width is `step/slope` - visible from
  about 2 pixels, i.e. `slope <= step/2`;
- where slope is 0 there is no transition to band at all;
- local noise of the order of the step dithers the edge away, so risk needs `residual < step`.

Verified on six synthetic cases with known answers:

| material | measured step | effective bits | risk |
| --- | --- | --- | --- |
| smooth ramp, 8-bit | 3.92e-03 | 8.0 | **29.3%** |
| smooth ramp, bf16 | 9.77e-04 | 10.0 | **13.3%** |
| smooth ramp, float32 | 5.87e-04 | 10.7 | 0% |
| pure noise | 1.86e-09 | 29.0 | 0% |
| 8-bit ramp + dithering noise | 1.49e-08 | 26.0 | 0% |
| detailed texture, 8-bit | 3.92e-03 | 8.0 | 0% |

The last two rows are the useful ones: identical bit depth, no risk, because the material dithers itself.
That is why the report ends with a verdict rather than a number - when risk is under 0.5% it says outright
that float32 buys headroom for a second pass here, not visible fidelity.

### Widgets and outputs

| Widget | Default | What it does |
| --- | --- | --- |
| `label` | `stats` | Free text at the head of the report, so two of these in one graph can be told apart. |

- `image:IMAGE` - the input, untouched.
- `report:STRING`
- `histogram:IMAGE` - log-scaled, with the `[0,1]` clamp limits marked in red. Log because the out-of-range
  tails are a fraction of a percent; on a linear axis they are invisible, which is exactly why people believe
  nothing is being lost.
- `banding_mask:MASK` - where bands will appear first.

---

## ANDRO Compare

Two batches in, numbers out. Frame counts may differ - the shorter one decides how many frames are compared -
but frame size must match.

The usual pairing is one decode with `keep_out_of_range` on and one with it off, or `float32` against
`vae default`: same seed, one setting changed, so whatever the report shows can only have come from that
setting.

### What the report says

**Global figures** - max and mean `|diff|`, the share of differing samples, PSNR.

**SSIM on the worst frame.** PSNR measures error *energy*, so a difference smeared thinly across a whole
frame and one concentrated in the sky score alike. SSIM answers the other question - did the structure
change - which is what banding is.

**The worst five frames**, but only when the frames actually differ from each other. On a uniform difference
the sort order is arbitrary, and listing "the worst five" would invite the reader to find a pattern in what
are really just descending frame numbers; the report says so explicitly instead. When there IS spread,
evenly spaced frame numbers here mean a tiling seam rather than a precision difference.

**Where in the frame the difference sits.** The worst frame is divided into an 8x8 grid and the hottest tile
is reported with its concentration factor. Above 3x the report warns that the global PSNR is diluted by clean
areas and understates what happens in that zone.

Verified on three cases:

| case | PSNR | what the node concludes |
| --- | --- | --- |
| identical inputs | inf | bit-identical - the change did not reach the pixels |
| bf16 vs float32, whole frame | 65.5 dB | small and evenly spread; survives a normal grade |
| damage confined to one corner | 46.4 dB | hottest zone **17.3x** the frame average, CONCENTRATED |

That third row is the entire point of the node. 46 dB reads as "fine" and is not.

**A verdict**, in these terms: bit-identical / below a 12-bit step and no delivery format carries it / small
and evenly spread / structural and localised and worth the float32 decode.

### Widgets and outputs

| Widget | Default | What it does |
| --- | --- | --- |
| `amplify` | 20.0 | Gain on the difference image. See the warning below. |

The amplified difference picture is the least trustworthy output in this pack, and the widget's tooltip says
so: the differences this pack is about stayed **black at 25x**. Trust the numbers.

- `difference:IMAGE`, `report:STRING`, `worst_zones:MASK` - every tile at least half as bad as the worst one.

---

## ANDRO Seam Check

Finds what tiled decoding left behind - and, since 1.3.0, predicts it before the decode as well.

### Diagnosis

Temporal tiling is the dangerous one. A diffusion decoder has no context at a temporal tile edge, so the
blend leaves a visibly **softer frame on every seam** - periodic, and easy to miss. Comparing consecutive
frames does not catch it, because the softening is smooth rather than a jump. This node measures per-frame
sharpness (mean `|Laplacian|`) and looks for local dips.

Motion in a shot produces isolated soft frames too, so the node does not demand that every gap match. It
looks for the largest subset sitting on one regular grid, and reports off-grid dips separately as probable
content. Spatial seams are handled the same way: a single strong column means nothing - a hard edge in the
content does that - so a seam is only reported when several excess peaks share a spacing.

### Prediction

Set `predict_from_temporal_size` to the `temporal_size` the decode used, and the node computes where soft
frames must land. The spacing is arithmetic, not luck:

```
tile_t = temporal_size / temporal_compression        (latent frames per tile)
step   = (tile_t - overlap_t) * temporal_compression (output frames between seams)
```

For LTX-2.5 at `temporal_size 32`: `tile_t = 4`, `overlap_t = 1`, step = 3 latent = **24 output frames**.
Run against a 121-frame batch with a softened frame every 24th, the node predicts `[25, 49, 73, 97, 121]`,
then confronts the prediction with the measurement and reports `CONFIRMED: 4 of 5 predicted seam frames
actually went soft`.

Disagreement is information too. If seams were predicted and nothing measurable turned up, the report says
the blend may be gentler than `dip_threshold` **or the compression figure is wrong for your VAE** - it does
not silently conclude "no seam".

With `temporal_size` at its default 4096 the node states that the whole batch fits in one tile and no
temporal seam is possible.

### Widgets and outputs

| Widget | Default | What it does |
| --- | --- | --- |
| `dip_threshold` | 0.93 | A frame counts as soft when sharpness falls below this fraction of the previous frame's. |
| `check_spatial` | on | Also look for vertical/horizontal seams from spatial tiling. |
| `predict_from_temporal_size` | 0 | 0 = measure only. Set it to predict. |
| `predict_from_temporal_overlap` | 8 | Read only when the above is not 0. |
| `temporal_compression` | 8 | Output frames per latent frame - 8 on LTX-2.5. A wrong value shifts the predicted spacing by the same factor. |

- `sharpness_plot:IMAGE` - per-frame sharpness with the soft frames marked.
- `report:STRING`
