#!/usr/bin/env python3
"""Extract the refusal directions the sweep edits with, once, on a GPU.

Writes ``data/refusal_directions.pt`` and ``data/directions.json`` beside it. The basename
is fixed: ``torch.save`` names zip entries after the file stem, so renaming moves the hash.

Run:  python scripts/extract_directions.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from abliteration import (  # noqa: E402
    collect_mean_last_token_states,
    refusal_directions,
    save_directions,
)
from abliteration.edit import target_matrices  # noqa: E402
from generation.qwen import N_LAYERS, load_model  # noqa: E402
from study.datasets import (  # noqa: E402
    DATA_DIR,
    FREEZE_MANIFEST_JSON,
    REFUSAL_DIRECTIONS_PT,
    load_extraction_harmful,
    load_extraction_harmless,
)
from study.manifest import sha256_file, write_manifest  # noqa: E402

DIRECTIONS_JSON = DATA_DIR / "directions.json"
BATCH = 32
PADDING_CHECK_PROMPTS = 16  # P1 needs one unbatched forward per prompt; 16 shows the same


def check_structure(model) -> None:
    """The one check the unit tests cannot make: they run against a 3-layer stub."""
    found = len(list(target_matrices(model)))
    expected = 1 + 2 * N_LAYERS
    if found != expected:
        raise SystemExit(f"expected {expected} target matrices, found {found}")


def check_datasets() -> None:
    """Extraction must run on the committed corpora, not a working copy."""
    frozen = json.loads(FREEZE_MANIFEST_JSON.read_text())["artifacts"]
    for name in ("extraction_harmful.csv", "extraction_harmless.csv"):
        if sha256_file(DATA_DIR / name) != frozen[name]["sha256"]:
            raise SystemExit(f"{name} has drifted from the freeze manifest")


def extract(model, tokenizer, harmful, harmless):
    return refusal_directions(
        collect_mean_last_token_states(model, tokenizer, harmful, batch_size=BATCH),
        collect_mean_last_token_states(model, tokenizer, harmless, batch_size=BATCH),
    )


def check_directions(directions) -> None:
    """What ``refusal_directions`` does not already raise on: it rejects a zero-norm
    layer, and unit norm is how it builds every row."""
    import torch

    if torch.isnan(directions).any():
        raise SystemExit("directions contain NaN")
    adjacent = torch.nn.functional.cosine_similarity(directions[:-1], directions[1:], dim=-1)
    if (adjacent > 1 - 1e-9).any():
        raise SystemExit("two adjacent layers produced an identical direction")


def padding_cosine(model, tokenizer, prompts) -> float:
    """P1: batched last-token pooling must match unbatched. Left pads consume position_ids
    in the forward-only path, so this is verified rather than assumed."""
    import torch

    batched = collect_mean_last_token_states(model, tokenizer, prompts, batch_size=len(prompts))
    single = collect_mean_last_token_states(model, tokenizer, prompts, batch_size=1)
    return torch.nn.functional.cosine_similarity(batched, single, dim=-1).min().item()


def split_half_cosines(model, tokenizer, harmful, harmless) -> list[float]:
    """P3: directions from disjoint halves of each side. The corpus is frozen at 128, so a
    low value cannot change it — it is reported, and escalated if very low."""
    import torch

    half = len(harmful) // 2
    first = extract(model, tokenizer, harmful[:half], harmless[:half])
    second = extract(model, tokenizer, harmful[half:], harmless[half:])
    return torch.nn.functional.cosine_similarity(first, second, dim=-1).tolist()


def main() -> None:
    check_datasets()
    harmful, harmless = load_extraction_harmful(), load_extraction_harmless()
    model, tokenizer = load_model()
    check_structure(model)

    directions = extract(model, tokenizer, harmful, harmless)
    check_directions(directions)
    sha256 = save_directions(directions, REFUSAL_DIRECTIONS_PT)

    padding = padding_cosine(model, tokenizer, harmful[:PADDING_CHECK_PROMPTS])
    split_half = split_half_cosines(model, tokenizer, harmful, harmless)

    # Only the cosines: the tensor hash identifies what they describe, and everything
    # else about this run is in the freeze manifest or derivable from the model.
    write_manifest(DIRECTIONS_JSON, {
        "sha256": sha256,
        "padding_cosine_min": padding,
        "split_half_cosine": split_half,
    })
    print(f"wrote {REFUSAL_DIRECTIONS_PT}\nwrote {DIRECTIONS_JSON}")
    print(f"  padding cosine (min over layers): {padding:.6f}")
    print(f"  split-half cosine: min {min(split_half):.3f} at layer {split_half.index(min(split_half))}")


if __name__ == "__main__":
    main()
