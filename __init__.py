# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Andrei Orehov / Andromediastudio - https://andromediastudio.com
# comfyui-vae-float32 - https://github.com/AndreiOrehov/comfyui-vae-float32
"""comfyui-vae-float32 - honest float32 out of any ComfyUI VAE.

See nodes.py for what each node does and why it exists.
"""

from .nodes import NODE_CLASS_MAPPINGS as _CORE_CLASSES
from .nodes import NODE_DISPLAY_NAME_MAPPINGS as _CORE_NAMES
from .qc_nodes import NODE_CLASS_MAPPINGS as _QC_CLASSES
from .qc_nodes import NODE_DISPLAY_NAME_MAPPINGS as _QC_NAMES

# Merged into new dicts rather than .update()-ing the ones nodes.py owns: those objects belong to
# that module, and a second module quietly growing them is how a mapping ends up depending on
# import order. Every key here keeps the ANDRO prefix and the ANDRO category, so web/
# andromedia_color.js - which matches on the category, not on a list of class names - colours the
# new node like the rest of the pack without being touched.
NODE_CLASS_MAPPINGS = {**_CORE_CLASSES, **_QC_CLASSES}
NODE_DISPLAY_NAME_MAPPINGS = {**_CORE_NAMES, **_QC_NAMES}

# Served at /extensions/comfyui-vae-float32/. Only one file lives there: the pack's house colour,
# which has no Python-side equivalent - server.py's node_info() sends the frontend a fixed set of
# fields and colour is not one of them.
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
