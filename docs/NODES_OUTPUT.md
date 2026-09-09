# ANDRO Remap Range and ANDRO Save EXR

The end of the chain: fit the values into a format that cannot hold them, or write a format that can.

Wiring map: **[NODES.md](NODES.md)**.

Colour is not this pack's job. Converting between colourspaces, grading, and writing movie containers is
what [ComfyUI-OCIO](https://github.com/SlavaSexton/ComfyUI-OCIO) does, and these two nodes deliberately stop
short of it: they move values along the axis they already sit on, and write them.

---

## ANDRO Remap Range

An 8- or 10-bit writer clips anything above 1.0. When that matters - a highlight worth keeping - map it down
deliberately instead of letting a clamp decide.

Only meaningful on an unclamped decode. A clamped one has nothing left outside the range.

### The four modes, and what each one costs

Measured on a realistic decode (`-0.0715 .. +1.0445`, 0.26% of samples above 1.0, 0.07% below 0):

| mode | samples moved | mid-tone at 0.5785 becomes | white point |
| --- | --- | --- | --- |
| `clip` | 0.33% | 0.5785 (untouched) | 1.0, and 0.33% flattened beyond recovery |
| `scale to fit` | **100.00%** | 0.5824 (moved) | 1.0 |
| `reinhard highlights` | 99.99% | 0.5723 (moved) | 1.0 |
| **`filmic rolloff`** (default) | **11.21%** | **0.5785 (untouched)** | **1.0** |
| `report only` | 0.00% | 0.5785 | unchanged |

`scale to fit` loses nothing, but it pays for a few overshooting samples by flattening the entire picture,
including the parts that were correctly exposed. With overshoot measured at 0.01-0.34% of samples, that is a
poor trade, and the report says so when you pick it.

`filmic rolloff` leaves everything below the knee **exactly** untouched - verified: maximum change in
`[0, knee)` is `0.0` - and bends only the range above it. It uses Reinhard with a white point applied to the
range above the knee, so the input maximum maps to exactly 1.0.

That last detail was a correction. The obvious choice, an exponential shoulder, approaches 1.0
asymptotically: the brightest sample landed at 0.941 and the white point turned grey. The current form
reaches **1.000000** on the same input.

Negative values are clamped to 0 in every mode. A `[0,1]` container has nowhere else to put them.

### The cost report

Because the decision is made in stops, the report gives them: how far the input peaked above 1.0, where the
output peaks, and what share of samples moved. `clip` additionally states how much is now unrecoverable, and
names the fact that this is identical to what stock ComfyUI does.

### Widgets and outputs

| Widget | Default | What it does |
| --- | --- | --- |
| `mode` | `filmic rolloff` | See the table. |
| `knee` | 0.8 | Where the rolloff starts bending. Below it, the picture passes through untouched. |

- `image:IMAGE`, `report:STRING`

---

## ANDRO Save EXR

Writes an `IMAGE` batch as a real float32 EXR sequence, values untouched - nothing clamped, scaled or
tone-mapped on the way out. Frames are numbered from 1 as `stem.00001.exr`.

### Backends, and why there are three

`cv2`'s EXR codec is compiled in but disabled unless `OPENCV_IO_ENABLE_OPENEXR=1` was set **before** `cv2`
was imported ([opencv/opencv#21326](https://github.com/opencv/opencv/issues/21326)), which no ComfyUI
launcher does. `cv2.imwrite` then writes nothing while returning quietly. So the `OpenEXR` module goes first,
`cv2` second, and a 32-bit float TIFF is the last resort - it carries the same values under a different
extension.

The node **verifies the first file it writes** and raises if nothing landed on disk, rather than reporting a
successful write it cannot prove.

A progress bar runs during the write. A 121-frame sequence is otherwise a long silent stretch in which the
node simply looks hung.

### Provenance metadata

With `write_metadata` on, the header carries `andro/*` attributes: the writer, bit depth, measured range,
the share outside `[0,1]`, and the frame count. Wire `ANDRO VAE Decode`'s `range_report` into the
`decode_report` input and it is stored verbatim as well, so the file records the dtypes the decode actually
ran in.

Verified by writing real files and reading them back:

```
andro/writer            = comfyui-vae-float32 (ANDRO Save EXR)
andro/bitDepth          = 32f
andro/range             = -0.071500 .. +1.044500
andro/outsideUnitRange  = 0.1465% below 0, 0.5859% above 1
andro/frames            = 3
andro/decodeReport      = decode (float32, no clamp): min=-0.071500 max=+1.044500 |
                          weights=torch.float32 vae_dtype=torch.float32
```

This does not overlap OCIO. Its metadata describes the plate - camera, lens, editorial attributes, timecode.
This describes where the pixels came from.

### The colour encoding, stated in the header

An untagged EXR is read as linear by every compositor that opens it. For a decoded SDR frame that is wrong:
what the VAE returns is **display-referred sRGB/Rec.709 gamma**, which is what SD, Flux, Wan and LTX SDR all
produce - hence the `srgb_display` default. The two exceptions: LTX-2.5 HDR decodes are **ACEScct**, and the
LTX-2.3 HDR IC-LoRA is **LogC3**.

The `colorspace` widget writes three readable strings plus the standard OpenEXR `chromaticities` attribute
(skipped for `unspecified`, whose primaries are unknown by definition). Verified by writing files and reading
the headers back:

```
andro/colorspace  = acescg - after OCIO ColorSpace to ACEScg
andro/transfer    = linear
andro/primaries   = ACES AP1
chromaticities    = (0.713, 0.293, 0.165, 0.830, 0.128, 0.044, 0.32168, 0.33767)   # a real
                    # chromaticities attribute, not a string - confirmed in the file's bytes
```

| `colorspace` | `andro/transfer` | `andro/primaries` |
| --- | --- | --- |
| `srgb_display` (default) | `sRGB piecewise` | `Rec.709` |
| `rec709_display` | `BT.1886 gamma 2.4` | `Rec.709` |
| `linear_rec709` | `linear` | `Rec.709` |
| `acescg` | `linear` | `ACES AP1` |
| `acescct` | `ACEScct log` | `ACES AP1` |
| `aces2065_1` | `linear` | `ACES AP0` |
| `logc3` | `ARRI LogC3 EI800` | `ARRI Wide Gamut 3` |
| `unspecified` | `unspecified` | `unspecified` (no `chromaticities`) |

**Nothing is converted.** The widget labels the pixels; it does not transform them. Whatever you pick here
must also be what Nuke or Resolve is told on input, or the file will be misread exactly as before - only now
the header says which answer is correct. Like the rest of the metadata, this needs the OpenEXR backend; the
TIFF fallback writes the picture alone, and the report says so.

### The `clipped` layer

With `clipped_layer` on, the file gets a second layer named `clipped` holding **exactly what the stock
`[0,1]` clamp would have deleted**, and zero everywhere the clamp would have kept the value. Verified on a
written file: the layer's range came out `-0.071500 .. +0.044500` with 0.732% non-zero, and it is zero
across every sample that was already in range.

The point is that the loss then travels inside the file, instead of living in a screenshot someone has to be
shown. Readers that ignore extra layers are unaffected; the cost is roughly double the file size.

Both metadata and the extra layer need the OpenEXR backend. When the TIFF fallback runs, the report says
plainly that the picture was written without them.

### Widgets and outputs

| Widget | Default | What it does |
| --- | --- | --- |
| `filename_prefix` | `float32/frame` | Subfolder and file stem. |
| `half_float` | off (32f) | 16f halves the size and still carries out-of-range values; what it costs is precision, ~11 bits of mantissa against 24 - still far above bf16. |
| `output_folder` | empty | Absolute path, or empty for the ComfyUI output directory. |
| `write_metadata` | on | The `andro/*` header attributes. |
| `clipped_layer` | off | The second layer described above. |
| `colorspace` | `srgb_display` | What the pixels are, written into the header as `andro/colorspace` + `andro/transfer` + `andro/primaries` + the standard `chromaticities`. A label, not a conversion. |
| `colorspace_note` | empty | Free text appended to the label, e.g. `Flux.2 decode`. |

| Input socket | |
| --- | --- |
| `decode_report:STRING*` | Optional. Wire `ANDRO VAE Decode`'s `range_report` here. |

- `folder:STRING` - the folder actually written to.
