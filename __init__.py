# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Andrei Orehov / Andromediastudio - https://andromediastudio.com
# comfyui-vae-float32 - https://github.com/AndreiOrehov/comfyui-vae-float32
"""comfyui-vae-float32 - honest float32 out of any ComfyUI VAE.

See nodes.py for what each node does and why it exists.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Served at /extensions/comfyui-vae-float32/. Only one file lives there: the pack's house colour,
# which has no Python-side equivalent - server.py's node_info() sends the frontend a fixed set of
# fields and colour is not one of them.
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
