"""Load release weights or the authors' original full-model experiment weights."""

import re

import torch


def load_checkpoint(model, path):
    """Load an MSRSC-Net state dict strictly, translating original key names.

    Only the unused single-frame temporal operator is discarded. Missing,
    unexpected or incompatible active weights raise an error. No download.
    """
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError("Expected a state dict or a dictionary containing state_dict.")
    converted = {}
    for key, value in state.items():
        key = key.removeprefix("module.")
        # In the original single-image model this temporal operator returns x
        # unchanged. It is absent from the image-only release.
        if re.fullmatch(
            r"hmu_[2-5]\.fuse\.0\.(temperal_proj_kv\.weight|temperal_proj\.[02]\.weight)",
            key,
        ):
            continue
        for old, new in (
            ("tra_", "projections."),
            ("siu_", "fusions."),
            ("hmu_", "decoders."),
            ("wall_ribbon_refine_", "scem."),
        ):
            key = re.sub(r"^" + old + r"([2-5])\.", new + r"\1.", key)
        key = re.sub(r"^(decoders\.[2-5])\.fuse\.1\.", r"\1.fuse.", key)
        key = key.replace(".gate_genator.", ".gate_generator.")
        if key.startswith("tra_1."):
            key = "upsample." + key[len("tra_1.") :]
        if key in converted:
            raise ValueError(f"Duplicate checkpoint key after conversion: {key}")
        converted[key] = value
    return model.load_state_dict(converted, strict=True)
