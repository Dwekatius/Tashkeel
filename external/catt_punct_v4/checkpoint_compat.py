from __future__ import annotations

import torch


INPUT_EMBED_KEY = "transformer.encoder.emb.tok_emb.weight"


def normalize_state_dict(state):
    if isinstance(state, dict) and "state_dict" in state:
        return state["state_dict"]
    return state


def load_state_dict_with_vocab_expansion(model, state):
    """Load checkpoints across tokenizer vocab expansions.

    v4 appends punctuation symbols to the original input vocabulary. That
    changes the size of the input embedding matrix, so strict load_state_dict
    from a v3 checkpoint would fail on shape mismatch. This helper preserves all
    overlapping rows and leaves the new punctuation rows at their model-init
    values.
    """

    state = normalize_state_dict(state)
    target_state = model.state_dict()
    patched_state = {}

    for key, value in state.items():
        if key not in target_state:
            patched_state[key] = value
            continue

        target_value = target_state[key]
        if value.shape == target_value.shape:
            patched_state[key] = value
            continue

        if key == INPUT_EMBED_KEY and value.ndim == 2 and target_value.ndim == 2 and value.shape[1] == target_value.shape[1]:
            merged = target_value.clone()
            rows = min(value.shape[0], target_value.shape[0])
            merged[:rows] = value[:rows].to(dtype=merged.dtype)
            patched_state[key] = merged
            continue

        patched_state[key] = value

    return model.load_state_dict(patched_state, strict=False)
