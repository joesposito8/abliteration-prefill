#!/usr/bin/env python3
"""Generate a run: the two model states over an evaluation set, or the layer sweep.

An evaluation set runs the factorial — the unedited base and the primary's edit, each
crossed with all 14 prefill levels. The validation set is the selection sweep instead: it
ranks every layer on breadth at a single unprefilled draw, and it is what decides the
primary, so no primary is looked up on that path.

Re-running resumes each condition from its own log directory, so a killed run restarts
with the same command.

Run:  python scripts/generate.py --prompt-set strongreject results/main/generated qwen3-4b
      python scripts/generate.py --prompt-set strongreject --num-prompts 30 results/smoke/generated qwen3-4b
      python scripts/generate.py --prompt-set validation results/selection/generated phi-4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from abliteration import load_directions  # noqa: E402
from abliteration.selection import condition_id, sweep_layers  # noqa: E402
from generation import target  # noqa: E402
from harness.conditions import Condition  # noqa: E402
from harness.run import run_sweep  # noqa: E402
from study import SEED  # noqa: E402
from study.datasets import load_prefills, model_dir, model_slug  # noqa: E402

PROMPT_SETS = ("strongreject", "validation")


def primary_layer(module) -> int:
    """The composition condition's layer, read from that target's abliteration freeze."""
    manifest = model_dir(module.MODEL_ID) / "abliteration_manifest.json"
    return json.loads(manifest.read_text())["primary"]["layer"]


def eval_conditions(
    prompt_set: str, module, primary: int | None, num_prompts: int | None = None
) -> list[Condition]:
    """The two model states on an evaluation set, every layer on the selection sweep.

    A resolved primary is what separates them: the sweep is what produces one, so it runs
    before any exists.
    """
    model = model_slug(module.MODEL_ID)
    layers = sweep_layers(module.N_LAYERS) if primary is None else [None, primary]
    return [
        Condition(
            model=model,
            id=condition_id(layer),
            seed=SEED,
            layer=layer,
            prompt_set=prompt_set,
            prefilled=primary is not None,
            num_prompts=num_prompts,
        )
        for layer in layers
    ]


def plan(
    prompt_set: str, module, num_prompts: int | None = None
) -> tuple[list[Condition], dict[tuple[int, str], str]]:
    """The conditions to run and the prefill text they read."""
    if prompt_set == "validation":
        return eval_conditions(prompt_set, module, None, num_prompts), {}
    return (
        eval_conditions(prompt_set, module, primary_layer(module), num_prompts),
        load_prefills(),
    )


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-set", choices=PROMPT_SETS, required=True)
    parser.add_argument("--num-prompts", type=int, default=None)
    parser.add_argument("root")
    parser.add_argument("model")
    args = parser.parse_args(argv)

    # Resolved before torch is imported, so a mistyped target names the enabled ones
    # rather than failing on a dependency that is only installed where it can run.
    module = target(args.model)
    directions_pt = model_dir(module.MODEL_ID) / "refusal_directions.pt"

    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "no CUDA device; the locked torch replaces the image's, so check the "
            "install before spending GPU time"
        )
    if not directions_pt.exists():
        raise SystemExit(f"{directions_pt} is missing; run scripts/extract_directions.py")

    conditions, prefills = plan(args.prompt_set, module, args.num_prompts)
    prefilled = [c.id for c in conditions if c.prefilled]
    print(f"{len(conditions)} conditions over the {args.prompt_set} set, prefilled: {prefilled}")

    directions = load_directions(directions_pt)
    weights, tokenizer = module.load_model()
    logs = run_sweep(
        conditions,
        prefills,
        module=weights,
        tokenizer=tokenizer,
        directions=directions,
        root=Path(args.root).resolve(),
    )
    print(f"generated {sum(log.results.total_samples for log in logs)} samples")


if __name__ == "__main__":
    main(sys.argv[1:])
