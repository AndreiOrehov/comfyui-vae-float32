# ANDRO Video QC

The ingest check. A generator hands you an mp4 and no report; this says whether the plate is
acceptable, and if not, which frames to look at.

Wiring map: **[NODES.md](NODES.md)**.

It is a **read-only** node: the batch goes in, nothing comes back out. Wire the loader's `IMAGE`
onward to the rest of the graph and hang this off the same socket.

Two independent sources of truth are used, and they are kept apart on purpose:

- the **pixels**, which say what the decoder produced;
- the **container**, via PyAV on `source_path`, which says what the file *claims*.

Most ingest surprises live in the gap between the two. Without `source_path` everything except the
fps and colour-tag checks still runs.

---

## The checks

### 1. Effective bit depth

Distinct values per channel, counted on up to **8 evenly spaced frames at full resolution** — full
resolution because quantisation lives in the individual samples, and a subsample throws away exactly
the evidence. The largest count decides: `≤256 → 8-bit`, `≤1024 → 10-bit`, else `float/16-bit`.

The values are then tested against the grids `k/255`, `k/1023` and `k/65535`, in that order, and the
first one that fits ≥ 99.9 % of samples (tolerance `1e-6`) wins. **The order is not cosmetic:**
`65535 = 255 × 257`, so every 8-bit value is also exactly a 16-bit value, and testing the coarse
grid last would report 8-bit material as 16-bit.

> **Read this line together with the probe, never alone.** A `yuv420p` file reaches an `IMAGE` batch
> through a YUV → RGB matrix — ComfyUI's own loader asks swscale for `gbrpf32le` — and that matrix
> lands the values on a far finer grid than the file ever held. Measured on two real Seedance clips:
> 8-bit h264 in, **59 515 and 51 748 distinct values per channel out, 100 % of them on `k/65535`**.
> Taken at face value that reads as a 16-bit source, and it is wrong. So when the container states a
> depth, the node prints a `^` line saying the container wins and the level count describes the
> decode path. It only becomes a WARN in the other direction — container claims 10-bit or more while
> the pixels hold ≤ 256 levels, i.e. depth that is present in the file and empty.

### 2. Range

Share of samples below 0 and above 1, and the NaN / Inf count with the frames they sit in. NaN is
always a FAIL: it survives a float32 decode, an upscaler and an EXR write, and only announces itself
in the compositor. Non-finite samples are zeroed for the temporal maths only — left in, one NaN
poisons its frame's mean and every delta touching it, and the report would then blame the
neighbours.

### 3. Luma: black frames, flashes, flicker

Per-frame mean luma (Rec.709 weights) on the subsampled frames.

| Reading | Rule | Verdict |
| --- | --- | --- |
| black frame | mean luma `< black_level` (0.03) | **FAIL** — a black frame in a generated clip is a dropped frame, not a look |
| flash / cut | `\|Δ mean luma\|` between neighbours `> flash_jump` (0.25) | reported, not a verdict |
| flicker | std of the luma delta **excluding** the cuts | reported |

A flash is indexed at the frame you land *on*, so one black frame produces two entries — the jump
into it and the jump out. That is the honest description of what an editor would pull.

Excluding the cuts from the flicker figure matters: one legitimate hard cut otherwise makes a
perfectly steady clip read as unstable.

### 4. Temporal: duplicates and held frames

Mean absolute frame-to-frame delta, on luma, at `1/n` scale (long side capped at **512 px** — these
are whole-frame averages, they converge long before full resolution, and at 4K the difference is a
QC pass that takes a minute instead of a second).

**Duplicates**: delta `< 1/1024`. Not `== 0`, because a re-encode never gives back a bit-exact
repeat — h264 quantisation moves a duplicated frame by a few thousandths, and demanding exact
equality reports zero duplicates on every mp4 ever made.

**Held**: delta `< held_ratio × median delta` (0.2 = "moved less than a fifth as much as this clip
normally moves"). Relative on purpose — an absolute threshold calls a locked-off shot a stall and a
whip pan clean. **The check switches itself off when the median delta is under `1/255`**, and says
so: below that there is no motion for a frame to be held against, and `0.2 × ~0` would flag the
entire clip.

Duplicates are a subset of held frames by construction, so a frame can be labelled `dup+held`.
The longest consecutive run is reported with its start frame.

Why this is the check that earns the node. Both real clips tested came back with a **period-4** held
pattern — frames 7, 11, 15, 19, 23, 27… and 55, 59, 63, 71, 79… — i.e. roughly one frame in four
barely moves. A 24 fps file whose motion updates every fourth frame is a 6 fps clip in a 24 fps
wrapper. It conforms, it plays, and it judders on the first pan. Nothing about that is visible in a
scrub-through.

### 5. Spec

`expected_width` / `expected_height` / `expected_frames` / `expected_fps`, each `0` = don't check.
**Any mismatch is a FAIL** — this is the check that catches a generator quietly returning 96 frames
for a 4 s 24 fps order, which is a changed edit, not a variation.

`expected_fps` is only answerable from the container probe: an `IMAGE` batch carries no timebase. Asked for
without a probe, the node says the check could not run rather than passing it.

### 6. Colour tags

`color_space`, `color_transfer`, `color_primaries` from the container. Any of them empty, `unknown`,
`reserved` or `N/A` → **WARN: untagged colour: readers will assume BT.601 / unknown transfer.**

Both Seedance clips tested are untagged on all three. That is the normal state of AI-generated
video, and it is the single most expensive thing on this list: the grade starts from the wrong
primaries and nobody gets an error.

---

## The verdict

| | Triggers |
| --- | --- |
| **FAIL** | NaN or Inf; any black frame; any spec mismatch |
| **WARN** | untagged colour; held frames > 10 % of the clip; any duplicate; > 1 % of samples outside [0,1]; container depth ≥ 10-bit while the pixels hold ≤ 256 levels |
| **PASS** | none of the above |

`pass:BOOLEAN` is `True` only for PASS — WARN and FAIL are both `False` — so a downstream branch can
be gated on it without parsing the text. Every FAIL and WARN is a sentence naming what failed: a QC
report that says FAIL without saying why has only moved the investigation somewhere else.

---

## How to read the contact sheet

Up to **24 flagged frames, 4 per row**, worst first: `black → nan/inf → flash → dup → held`. One
entry per frame with the reasons merged, so `00012  dup+held` is one thumbnail, not two. Each label
is the **0-based frame number** and the reason, exactly as the report lists them.

When nothing is flagged the sheet shows **8 evenly spaced frames labelled `ok`** — an empty output
would be indistinguishable from a node that failed to run.

The sheet exists because half of what this node flags is a judgement call at the margin. A held
frame on a slow push is fine; a held frame mid-pan is not. The list of numbers cannot tell you
which, and the frames side by side can.

---

## Widgets and outputs

| Widget | Default | What it does |
| --- | --- | --- |
| `source_path` | `""` | The file the batch was loaded from. Unlocks the probe: codec, pix_fmt, fps and colour tags. |
| `expected_fps` | 0 | 0 = don't check. Needs `source_path`. |
| `expected_frames` | 0 | 0 = don't check. |
| `expected_width` / `expected_height` | 0 | 0 = don't check. |
| `held_ratio` | 0.2 | Fraction of the median delta below which a frame counts as held. |
| `flash_jump` | 0.25 | Mean luma jump between neighbours, in 0..1 units, that reads as a flash or cut. |
| `black_level` | 0.03 | Mean luma below which a frame is called black. |
| `write_json` | off | Also write the JSON report under the ComfyUI output folder. |
| `json_prefix` | `qc/report` | Path under that folder, extension added if missing. |

- `report:STRING` — verdict, one line per check, then the flagged-frame lists (truncated to 30 each).
  Also shown on the node itself, since it is an `OUTPUT_NODE`.
- `json:STRING` — the same measurements with **every list complete**, plus the per-frame mean luma,
  luma delta and frame delta series. For logging a batch of clips and diffing them later.
- `sheet:IMAGE` — the contact sheet, `(1, H, W, 3)`.
- `pass:BOOLEAN` — `True` only on PASS.

## Cost

Measured on this machine (CPU, no GPU): **1.8 s for 121 frames** and **4.6 s for 361 frames** at
1280×720, excluding the load. The distinct-value count at full resolution is the bulk of it, which
is why it is capped at 8 frames.
