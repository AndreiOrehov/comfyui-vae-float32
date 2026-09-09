# The ten nodes: what connects to what

This is the wiring map for the whole pack. It answers three questions and nothing else: what shape is each
node, what may be plugged into it, and what does it plug into. For every widget, every allowed value and
every per-node detail, read the reference for that group:

- **[NODES_DECODE.md](NODES_DECODE.md)** - `ANDRO VAE Decode` and `ANDRO VAE Encode`, the two nodes that
  touch the VAE.
- **[NODES_MEASURE.md](NODES_MEASURE.md)** - `ANDRO Range Stats`, `ANDRO Compare` and `ANDRO Seam Check`,
  which measure rather than change.
- **[NODES_OUTPUT.md](NODES_OUTPUT.md)** - `ANDRO Remap Range` and `ANDRO Save EXR`, the end of the chain.
- **[NODES_AUDIO.md](NODES_AUDIO.md)** - `ANDRO Load Audio` and `ANDRO Audio Switch`, which exist because
  ComfyUI validates a whole prompt before it runs any of it.
- **[NODES_QC.md](NODES_QC.md)** - `ANDRO Video QC`, the ingest check for a clip that arrived from a
  generator rather than from this graph.

Everything below was read out of the pack's own `NODE_CLASS_MAPPINGS`, not from memory. If your install
disagrees, your install is the truth: ask it the same question with `GET /object_info/ANDROVAEDecode` and
friends.

## The shape of every node

An asterisk marks an optional socket. A socket with no asterisk must be connected or the prompt will not
validate. Widgets are left out here on purpose; the group references cover them.

| Node | Input sockets | Outputs |
| --- | --- | --- |
| `ANDRO VAE Decode` | `samples:LATENT`, `vae:VAE` | `image:IMAGE`, `range_report:STRING` |
| `ANDRO VAE Encode` | `pixels:IMAGE`, `vae:VAE` | `latent:LATENT`, `report:STRING` |
| `ANDRO Range Stats` | `image:IMAGE` | `image:IMAGE`, `report:STRING`, `histogram:IMAGE`, `banding_mask:MASK` |
| `ANDRO Compare` | `image_a:IMAGE`, `image_b:IMAGE` | `difference:IMAGE`, `report:STRING`, `worst_zones:MASK` |
| `ANDRO Seam Check` | `images:IMAGE` | `sharpness_plot:IMAGE`, `report:STRING` |
| `ANDRO Remap Range` | `image:IMAGE` | `image:IMAGE`, `report:STRING` |
| `ANDRO Save EXR` | `images:IMAGE`, `decode_report:STRING*` | `folder:STRING` |
| `ANDRO Load Audio` | none, the source is a widget | `audio:AUDIO`, `report:STRING` |
| `ANDRO Audio Switch` | `generated_audio:LATENT*`, `external_audio:LATENT*` | `latent:LATENT`, `mode:STRING` |
| `ANDRO Video QC` | `images:IMAGE` | `report:STRING`, `json:STRING`, `sheet:IMAGE`, `pass:BOOLEAN` |

Five things fall straight out of that table.

**Every node that measures also passes its input through.** `ANDRO Range Stats` returns the same `IMAGE` it
was handed, unchanged, so it can sit in the middle of a chain instead of hanging off the side of it. That is
deliberate: a measurement you have to detour around is a measurement you stop taking.

**Two nodes are the only ones that speak `LATENT`,** and they are mirror images: Encode takes pixels and
gives a latent, Decode takes a latent and gives pixels. `ANDRO Audio Switch` also carries `LATENT`, but it
never looks inside one - it picks between two.

**`ANDRO Save EXR` ends a chain.** Its only output is the folder it wrote, a `STRING` for logging or for
wiring into a node that reads the sequence back. Nothing downstream needs the picture from it.

**`ANDRO Load Audio` starts one,** and has no input sockets at all, because its source is a file you pick.

**Both `ANDRO Audio Switch` inputs are optional AND lazy.** Either side can be muted, bypassed or deleted and
the survivor is used; the branch that loses is never computed. This is the one node in the pack with no
connection to float32 at all - it is here because the problem it solves showed up while building the rest.

## Which socket accepts what

Every socket takes a standard ComfyUI type, so anything in your install that produces that type will connect.

| Type | Wire it from | Notes |
| --- | --- | --- |
| `IMAGE` | `LoadImage`, `VAEDecode`, `ANDRO VAE Decode`, `OCIO Read`, any generation node | The common currency of the pack. |
| `LATENT` | `KSampler`, `EmptyLatentImage`, `VAEEncode`, `ANDRO VAE Encode` | |
| `VAE` | `VAELoader`, `CheckpointLoaderSimple` | Must be the VAE that belongs to the model - it is trained with the transformer and cannot be swapped. |
| `MASK` | produced by `ANDRO Range Stats` and `ANDRO Compare` | Wire to `PreviewImage` through `MaskToImage`, or into any masking node. |
| `AUDIO` | `ANDRO Load Audio`, `LoadAudio`, an audio VAE decode | |
| `STRING` | any report output in this pack | `ANDRO Save EXR`'s `decode_report` input is the only STRING socket that takes one. |

## The one wiring that is not obvious

`ANDRO VAE Decode`'s `range_report` into `ANDRO Save EXR`'s `decode_report`:

```
ANDRO VAE Decode ──image──────────► ANDRO Save EXR
                 └─range_report───► decode_report
```

That second wire is what makes the written EXR carry the dtypes the decode actually ran in, not just the
range measurable from the pixels afterwards. Without it the file still gets provenance metadata; with it the
file can answer "was this the float32 pass?" six months later, on its own.

## A chain that uses most of the pack

```
KSampler ──► ANDRO VAE Decode ──► ANDRO Range Stats ──► ANDRO Remap Range ──► SaveImage
                    │                     │                                   (8-bit delivery)
                    │                     └─banding_mask──► PreviewImage
                    │
                    ├──► ANDRO Seam Check          (did tiling leave marks?)
                    └──► ANDRO Save EXR            (the float32 master)
```

Stock `VAEDecode` on the same latent, wired with this pack's decode into `ANDRO Compare`, is how every number
in the README was produced.
