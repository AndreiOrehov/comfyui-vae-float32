"""Measure what a VAE decode loses, straight against ComfyUI's own code.

Loads a VAE outside the server, decodes the same latent twice - once the way
ComfyUI does it, once without the clamp and in float32 - and prints the range and
the level count for both. This is where the numbers in the README come from.

    python tools/measure_precision.py <comfyui-core-dir> <vae.safetensors> [image.png]
"""
import os
import sys

import numpy as np

CORE, VAE_PATH = sys.argv[1], sys.argv[2]
IMAGE = sys.argv[3] if len(sys.argv) > 3 else None

sys.argv = [sys.argv[0]]
sys.path.insert(0, CORE)
os.chdir(CORE)

import torch  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.utils  # noqa: E402


def levels(a, lo=0.2, hi=0.3):
    w = a[(a > lo) & (a < hi)]
    if w.size < 4:
        return 0, float("nan")
    u = np.unique(w)
    return u.size, float(np.min(np.diff(u)))


def report(tag, t):
    a = t.detach().float().cpu().numpy()
    n, step = levels(a)
    print(f"  {tag:<28} min={a.min():+.6f} max={a.max():+.6f}  "
          f"outside[0,1]={100 * ((a < 0) | (a > 1)).mean():.4f}%  "
          f"levels in [0.2,0.3]={n} (step {step:.2e})")


sd = comfy.utils.load_torch_file(VAE_PATH)
meta = comfy.utils.load_torch_file(VAE_PATH, return_metadata=True)[1]
vae = comfy.sd.VAE(sd=sd, metadata=meta)
print(f"{os.path.basename(VAE_PATH)}: {type(vae.first_stage_model).__name__}, "
      f"vae_dtype={vae.vae_dtype}, working={vae.working_dtypes}")

H, W, FRAMES = 256, 448, 9
if IMAGE:
    from PIL import Image
    im = Image.open(IMAGE).convert("RGB").resize((W, H), Image.LANCZOS)
    px = torch.from_numpy(np.asarray(im).astype(np.float32) / 255.0)
else:                                                  # synthetic full-range chart
    px = torch.zeros(H, W, 3)
    px[H // 4:H // 2] = 1.0
    px[H // 2:3 * H // 4] = torch.linspace(0, 1, W)[None, :, None]
px = px.unsqueeze(0).repeat(FRAMES, 1, 1, 1)

with torch.inference_mode():
    lat = vae.encode(px)

stock = vae.process_output
with torch.inference_mode():
    report("stock (clamped, vae dtype)", vae.decode(lat).clone())

vae.process_output = lambda image: image.add_(1.0).div_(2.0)
if torch.float32 in vae.working_dtypes:
    vae.first_stage_model.to(torch.float32)
    vae.vae_dtype = torch.float32
try:
    with torch.inference_mode():
        report("float32, no clamp", vae.decode(lat).clone())
finally:
    vae.process_output = stock
