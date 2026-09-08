# ANDRO Load Audio and ANDRO Audio Switch

The two nodes in this pack that have nothing to do with float32. They are here because the problem they solve
turned up while building the rest, and because it has no clean solution with stock nodes.

Wiring map: **[NODES.md](NODES.md)**.

---

## The problem both nodes exist for

**ComfyUI validates every node in a prompt before it runs any of it.** Not the nodes it is about to execute -
all of them.

For an audio-conditioned video model such as LTX, that has a consequence people meet as a mystery. LTX's
`LTXVConcatAVLatent` demands an audio latent, so a graph that merely *contains* an audio branch cannot run
without the file, even when a switch downstream was never going to use it. Muting the whole chain is the
usual workaround, and one un-muted node anywhere in it breaks the trick again.

Two details make it worse:

- **A lazy input does not help.** Measured: a lazy input still fails validation.
- **Bypass is not mute.** Bypass passes an input of the same type through to the output; a node with no such
  input - `LTXVEmptyLatentAudio`, for instance - has its output turn into nothing.

The fix is to own the validation, and to make the optional branch genuinely optional.

---

## ANDRO Load Audio

`LoadAudio` that treats a missing file as silence instead of killing the prompt.

Declaring `VALIDATE_INPUTS(audio_file)` makes ComfyUI hand its own combo check for that widget over to this
node (`execution.py:1019`). An unknown filename then passes validation, reaches `execute()`, and turns into
silence plus a line in the report. The branch costs nothing and breaks nothing when unused - and a workflow
moved to another machine, where that `.wav` does not exist, still runs.

Nothing is ever swapped for silence quietly. Every substitution is named in the report, with the reason.

### Rate and duration

The report always states duration, sample rate and channel count, because audio length frequently decides
clip length and today that is only learned after the run.

When the file's rate differs from `sample_rate`, the node says so either way:

- **off** (default): the file's own rate is passed on, with a note naming both rates. A mismatch carried
  silently downstream fails somewhere far from its cause.
- **on**: resampled via `torchaudio.functional.resample`, and the report states the change. Verified:
  44100 → 48000 Hz turned 220500 samples into 240000, duration unchanged at 5.000 s. A failed resample is
  reported and the original is passed through rather than the run being lost.

`sample_rate` applies to generated silence only. A real file always keeps its own rate.

### Widgets and outputs

| Widget | Default | What it does |
| --- | --- | --- |
| `audio_file` | `(none - silence)` | A file from the input folder. An unknown name yields silence and a note, not an error. |
| `silence_seconds` | 5.0 | Length of the substituted silence. |
| `sample_rate` | 48000 | Rate of the generated silence; also the resample target. |
| `path_override` | empty | Absolute path for audio outside the input folder. Wins over `audio_file`. |
| `resample_to_sample_rate` | off | See above. |

- `audio:AUDIO`, `report:STRING`

---

## ANDRO Audio Switch

Picks between two latents with one toggle. No muting, no deleting, no rewiring.

Both inputs are **optional** and **lazy**:

- optional, so either branch can be muted, bypassed or deleted and the survivor is used. Requiring one of
  them made disabling *that* side fail with ComfyUI's "missing a required input", which reads like a broken
  node rather than a muted branch;
- lazy, so the branch that loses is never computed. `check_lazy_status` asks for one side only.

A sentinel object distinguishes "not connected" from "connected but not evaluated yet" - a lazy input arrives
as `None` in the second case, so defaulting absent inputs to `None` would make the two indistinguishable.

If neither side is connected the node raises with an instruction rather than a stack trace.

Nothing about this is specific to audio. It is the generic answer to "this input is mandatory but I want it
to be skippable", and it happens to be named after the graph it was written for.

### Verified behaviour

Full graph runs, LTX-2.5:

| toggle | file present | result |
| --- | --- | --- |
| `generated` | yes | the loader never executed at all, 121 EXR frames, 145 s |
| `external` | yes | `loaded ...: 5.00s, 48000 Hz, 2ch`, 121 EXR frames, 94 s |
| `external` | **no** | graph accepted, silence, 121 EXR frames, 84 s |

The first row is the lazy input doing its job: the branch that lost was not merely ignored, it was not run.

### Widgets and outputs

| Widget | Default | What it does |
| --- | --- | --- |
| `audio_source` | `external` | Which branch wins when both are connected. If only one is connected, that one is used whatever this says. |

| Input socket | |
| --- | --- |
| `generated_audio:LATENT*` | What the model composes on its own - an empty audio latent for LTX. |
| `external_audio:LATENT*` | Your own audio, encoded to a latent. |

- `latent:LATENT` - whichever branch won.
- `mode:STRING` - which one that was, and whether it was the one the switch asked for, so a fallback is
  visible rather than silent.
