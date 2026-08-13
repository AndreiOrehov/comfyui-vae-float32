# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Andrei Orehov / Andromediastudio - https://andromediastudio.com
# comfyui-vae-float32 - https://github.com/AndreiOrehov/comfyui-vae-float32
"""Rename Audio Latent Switch inputs in saved workflows (1.0 -> 1.1).

The node's inputs were renamed so the promoted widget reads as something other
than `prefer_external: fallback | external`, which says nothing once it is
surfaced on a subgraph among video settings:

    generated       -> generated_audio
    external        -> external_audio
    prefer_external -> audio_source        (labels: generated | external)

Input names are part of the contract, so an old workflow otherwise fails with
"missing a required input". This rewrites both formats in place.

A promoted subgraph input keeps the name it was created with — this script
renames it too, but ComfyUI may still show the stale one until you remove the
promotion and re-add it.

Usage:  python migrate_switch_names.py <file-or-dir> [...]
"""
import json
import shutil
import sys
from pathlib import Path

RENAME = {
    "generated": "generated_audio",
    "external": "external_audio",
    "prefer_external": "audio_source",
}


def migrate(doc):
    """Returns the number of renames applied."""
    n = 0

    def fix_ui_node(node):
        nonlocal n
        if node.get("type") != "AudioLatentSwitch":
            return
        for slot in node.get("inputs") or []:
            new = RENAME.get(slot.get("name"))
            if new:
                slot["name"] = new
                if slot.get("localized_name") in RENAME:
                    slot["localized_name"] = RENAME[slot["localized_name"]]
                n += 1

    def fix_promoted(sub):
        nonlocal n
        for slot in sub.get("inputs") or []:
            for key in ("name", "label"):
                if slot.get(key) in RENAME:
                    slot[key] = RENAME[slot[key]]
                    n += 1

    if "nodes" in doc:                                   # UI format
        for node in doc["nodes"]:
            fix_ui_node(node)
        for sub in (doc.get("definitions", {}) or {}).get("subgraphs", []) or []:
            for node in sub.get("nodes", []):
                fix_ui_node(node)
            fix_promoted(sub)
    else:                                                # API format
        for node in doc.values():
            if isinstance(node, dict) and node.get("class_type") == "AudioLatentSwitch":
                inputs = node.get("inputs", {})
                for old, new in RENAME.items():
                    if old in inputs:
                        inputs[new] = inputs.pop(old)
                        n += 1
    return n


def main(targets):
    files = []
    for t in targets:
        p = Path(t)
        files += sorted(p.glob("**/*.json")) if p.is_dir() else [p]
    for f in files:
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  skip {f.name}: {e}")
            continue
        n = migrate(doc)
        if not n:
            continue
        shutil.copy2(f, f.with_suffix(f.suffix + ".bak"))
        f.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  {f}: {n} rename(s), backup at {f.name}.bak")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
