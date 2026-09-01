#!/usr/bin/env python3
"""Extract the refusal directions the sweep edits with, once per target, on a GPU.

Writes ``refusal_directions.pt`` and ``directions.json`` into the target's own artifact
directory. The basename is fixed: ``torch.save`` names zip entries after the file stem,
so renaming moves the hash.

Run:  python scripts/extract_directions.py <model-slug>
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from abliteration import (  # noqa: E402
    check_inventory,
    collect_mean_last_token_states,
    refusal_directions,
    save_directions,
)
from generation import target  # noqa: E402
from study.datasets import (  # noqa: E402
    load_extraction_harmful,
    load_extraction_harmless,
    model_dir,
)
from study.manifest import write_manifest  # noqa: E402

BATCH = 32
PADDING_CHECK_PROMPTS = 16  # P1 needs one unbatched forward per prompt; 16 shows the same


def extract(model, tokenizer, build_prompt, harmful, harmless):
    return refusal_directions(
        collect_mean_last_token_states(
            model, tokenizer, harmful, build_prompt=build_prompt, batch_size=BATCH
        ),
        collect_mean_last_token_states(
            model, tokenizer, harmless, build_prompt=build_prompt, batch_size=BATCH
        ),
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


def padding_cosine(model, tokenizer, build_prompt, prompts) -> float:
    """P1: batched last-token pooling must match unbatched. Left pads consume position_ids
    in the forward-only path, so this is verified rather than assumed."""
    import torch

    batched = collect_mean_last_token_states(
        model, tokenizer, prompts, build_prompt=build_prompt, batch_size=len(prompts)
    )
    single = collect_mean_last_token_states(
        model, tokenizer, prompts, build_prompt=build_prompt, batch_size=1
    )
    return torch.nn.functional.cosine_similarity(batched, single, dim=-1).min().item()


def split_half_cosines(model, tokenizer, build_prompt, harmful, harmless) -> list[float]:
    """P3: directions from disjoint halves of each side. The corpus is frozen at 128, so a
    low value cannot change it — it is reported, and escalated if very low."""
    import torch

    half = len(harmful) // 2
    first = extract(model, tokenizer, build_prompt, harmful[:half], harmless[:half])
    second = extract(model, tokenizer, build_prompt, harmful[half:], harmless[half:])
    return torch.nn.functional.cosine_similarity(first, second, dim=-1).tolist()


def main(argv: list[str]) -> None:
    if len(argv) != 1:
        raise SystemExit("usage: python scripts/extract_directions.py <model-slug>")
    module = target(argv[0])
    build_prompt = module.build_prompt
    directions_pt = model_dir(module.MODEL_ID) / "refusal_directions.pt"
    directions_json = model_dir(module.MODEL_ID) / "directions.json"

    harmful, harmless = load_extraction_harmful(), load_extraction_harmless()
    model, tokenizer = module.load_model()
    check_inventory(model, module.N_LAYERS)

    directions = extract(model, tokenizer, build_prompt, harmful, harmless)
    check_directions(directions)
    sha256 = save_directions(directions, directions_pt)

    padding = padding_cosine(
        model, tokenizer, build_prompt, harmful[:PADDING_CHECK_PROMPTS]
    )
    split_half = split_half_cosines(model, tokenizer, build_prompt, harmful, harmless)

    write_manifest(directions_json, {
        "model": module.MODEL_ID,
        "revision": module.REVISION,
        "n_layers": module.N_LAYERS,
        "sha256": sha256,
        "padding_cosine_min": padding,
        "split_half_cosine": split_half,
    })
    print(f"wrote {directions_pt}\nwrote {directions_json}")
    print(f"  padding cosine (min over layers): {padding:.6f}")
    print(f"  split-half cosine: min {min(split_half):.3f} at layer {split_half.index(min(split_half))}")


if __name__ == "__main__":
    main(sys.argv[1:])
