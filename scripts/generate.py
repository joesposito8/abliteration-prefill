#!/usr/bin/env python3
"""Generate the selection sweep: the unedited base, then every layer's edit.

Re-running resumes each condition from its own log directory, so a killed run restarts
with the same command.

Run:  python scripts/generate.py results/sweep/generated
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from abliteration import load_directions  # noqa: E402
from abliteration.selection import condition_id  # noqa: E402
from generation.qwen import N_LAYERS, load_model  # noqa: E402
from harness.conditions import Condition  # noqa: E402
from harness.run import run_sweep  # noqa: E402
from study import SEED  # noqa: E402
from study.datasets import DATA_DIR  # noqa: E402

DIRECTIONS_PATH = DATA_DIR / "refusal_directions.pt"


def sweep_conditions() -> list[Condition]:
    """The base row then one condition per layer, named as the freeze step expects."""
    return [
        Condition(id=condition_id(layer), seed=SEED, layer=layer, prompt_set="validation")
        for layer in [None, *range(N_LAYERS)]
    ]


def main(argv: list[str]) -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "no CUDA device; the locked torch replaces the image's, so check the "
            "install before spending GPU time"
        )
    if not DIRECTIONS_PATH.exists():
        raise SystemExit(f"{DIRECTIONS_PATH} is missing; run scripts/extract_directions.py")

    directions = load_directions(DIRECTIONS_PATH)
    conditions = sweep_conditions()
    print(f"{len(conditions)} conditions over the validation set at k=1")

    model, tokenizer = load_model()
    # The validation set is unprefilled, so there is no prefill text to supply.
    logs = run_sweep(
        conditions,
        {},
        module=model,
        tokenizer=tokenizer,
        directions=directions,
        root=Path(argv[0]).resolve(),
    )
    print(f"generated {sum(log.results.total_samples for log in logs)} samples")


if __name__ == "__main__":
    main(sys.argv[1:])
