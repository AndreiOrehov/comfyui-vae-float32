# Attribution

## What this pack builds on

**ComfyUI** (Comfy-Org, GPL-3.0) — the nodes reach into `comfy.sd.VAE`: `process_output`, `vae_dtype`
and `first_stage_model`. Line references throughout the README and CHANGELOG point at
`comfy/sd.py` as of ComfyUI **0.32.0**; they are quoted to explain behaviour, and no ComfyUI code is
copied into this repository.

**OpenEXR** (Academy Software Foundation, BSD-3-Clause) — used through its Python bindings for EXR
output. The `OpenEXR.File` API arrived in 3.3, which is why that is the floor in `requirements.txt`.

**NumPy** (BSD-3-Clause) and **tifffile** (BSD-3-Clause) — measurement maths and the float TIFF
fallback.

## Prior art and thanks

**[ComfyUI-OCIO](https://github.com/SlavaSexton/ComfyUI-OCIO)** by Slava Sexton — the repository this
one is laid out after, and the pack that first put proper colour management and EXR/ProRes output in
front of ComfyUI users. The EXR-writing fix in this pack was contributed back there as
[PR #5](https://github.com/SlavaSexton/ComfyUI-OCIO/pull/5): OpenCV's EXR codec is disabled unless
`OPENCV_IO_ENABLE_OPENEXR=1` predates the `cv2` import, so any pack writing EXR through `cv2.imwrite`
silently writes nothing.

**[ComfyUI-Agent-Kit](https://github.com/SlavaSexton/comfyui-agent-kit)** by Slava Sexton — the
working method behind this pack: measure rather than assume, keep the measuring scripts in the
repository next to the numbers they produced, and state what was verified and what was not.

**[opencv/opencv#21326](https://github.com/opencv/opencv/issues/21326)** — the upstream issue that
explains the disabled EXR codec.

## The measurements

Every number in the README and CHANGELOG was measured on one machine — RTX 5090, ComfyUI 0.32.0,
Windows, PyTorch 2.12.1+cu130 — with the scripts in `tools/`. Timings are hardware-specific; the level
counts and value ranges are properties of the decode path and should reproduce anywhere.
