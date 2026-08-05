#!/usr/bin/env python3
"""Measure the batch width instead of assuming it, and record what decided it.

Two questions. Under greedy decoding, does a prompt's output depend on what else shared
its forward pass? If it does, the width is a frozen study parameter rather than a cost
knob, because two conditions run at different widths would not be comparable. Then,
under the study's real decoding, what does each width cost.

Parity has to be greedy. Under sampling, a row batched with 31 others draws from a
stream those others also consume, so a difference would say nothing about batching.

Run:  python scripts/batch_sweep.py [artifact-path]
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from generation.qwen import (  # noqa: E402
    DECODING,
    MODEL_ID,
    REVISION,
    build_prompt,
    generate_prompts,
    load_model,
)
from harness_smoke import SYNTHETIC  # noqa: E402
from prefills import STATIC_BASELINE  # noqa: E402
from study.datasets import load_strongreject_prompts  # noqa: E402
from study.manifest import write_manifest  # noqa: E402

ARTIFACT = REPO / "data" / "batch_sweep.json"
SEED = 20260805
WIDTHS = (1, 16, 32, 64, 128)

# Divergence from padding compounds, so it shows up early if it shows up at all, and a
# short cap keeps the reference pass — one prompt at a time — down to a few minutes.
PARITY_PROMPTS = 16
GREEDY = {"do_sample": False, "max_new_tokens": 128}

# Bounds on the choice, stated before any number exists.
VRAM_CEILING_GB = 60
TOLERANCE = 0.10


def workload(tokenizer, rows: int) -> list[str]:
    """Real forbidden prompts under prefills of every shape, so padding waste is real."""
    prefills = (STATIC_BASELINE, *SYNTHETIC)
    forbidden = load_strongreject_prompts()["forbidden_prompt"].head(rows)
    return [
        build_prompt(tokenizer, prompt, prefills[i % len(prefills)])
        for i, prompt in enumerate(forbidden)
    ]


def kv_bytes_per_token(model) -> int:
    config = model.config
    head_dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads
    )
    return 2 * config.num_key_value_heads * head_dim * 2 * config.num_hidden_layers


def measure(model, tokenizer, prompts: list[str], width: int) -> dict:
    """One width, warmed up first so the first call's autotuning stays out of the clock."""
    import torch

    batch = prompts[:width]
    generate_prompts(model, tokenizer, batch, seed=SEED)

    torch.cuda.reset_peak_memory_stats()
    rows, seconds = generate_prompts(model, tokenizer, batch, seed=SEED)
    longest = max(row.prompt_tokens for row in rows) + DECODING["max_new_tokens"]

    return {
        "width": width,
        "seconds": seconds,
        "prompts_per_second": width / seconds,
        "tokens_per_second": sum(row.new_tokens for row in rows) / seconds,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30,
        "predicted_kv_gb": width * longest * kv_bytes_per_token(model) / 2**30,
    }


def parity(model, tokenizer, prompts: list[str]) -> dict:
    """The same prompts batched against themselves generated one at a time."""
    subject = prompts[:PARITY_PROMPTS]
    alone = [
        generate_prompts(model, tokenizer, [prompt], seed=SEED, decoding=GREEDY)[0][
            0
        ].continuation
        for prompt in subject
    ]

    differs = []
    for width in (w for w in WIDTHS if w >= PARITY_PROMPTS):
        rows, _ = generate_prompts(
            model, tokenizer, prompts[:width], seed=SEED, decoding=GREEDY
        )
        if [row.continuation for row in rows[:PARITY_PROMPTS]] != alone:
            differs.append(width)

    return {
        "width_invariant": not differs,
        "differs_at": differs,
        "prompts": PARITY_PROMPTS,
        "max_new_tokens": GREEDY["max_new_tokens"],
    }


def choose_width(measurements: list[dict]) -> tuple[int, str]:
    """The pre-stated rule: fastest that fits, then down while the loss stays small.

    Smaller is preferred at equal speed because a failed batch re-runs that many
    samples, and ``max_samples`` is twice the width.
    """
    eligible = [m for m in measurements if m["peak_vram_gb"] <= VRAM_CEILING_GB]
    if not eligible:
        raise SystemExit(f"every width exceeded {VRAM_CEILING_GB} GB")

    fastest = max(eligible, key=lambda m: m["prompts_per_second"])
    floor = fastest["prompts_per_second"] * (1 - TOLERANCE)

    for measurement in measurements:
        under = 1 - measurement["prompts_per_second"] / fastest["prompts_per_second"]
        measurement["under_fastest"] = round(under, 3)

    chosen, rule = fastest, "fastest within the memory ceiling"
    smaller = sorted(
        (m for m in eligible if m["width"] < fastest["width"]),
        key=lambda m: m["width"],
        reverse=True,
    )
    for candidate in smaller:
        if candidate["prompts_per_second"] < floor:
            break
        chosen, rule = candidate, f"smallest within {TOLERANCE:.0%} of the fastest"
    return chosen["width"], rule


def main(argv: list[str]) -> None:
    import torch
    import transformers

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device")

    artifact = Path(argv[0]).resolve() if argv else ARTIFACT
    model, tokenizer = load_model()
    prompts = workload(tokenizer, max(WIDTHS))

    invariance = parity(model, tokenizer, prompts)
    widths = [measure(model, tokenizer, prompts, width) for width in WIDTHS]
    chosen, rule = choose_width(widths)

    result = {
        "model": MODEL_ID,
        "revision": REVISION,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "seed": SEED,
        "decoding": dict(DECODING),
        "parity": invariance,
        "widths": widths,
        "chosen": chosen,
        "rule": rule,
    }
    write_manifest(artifact, result)
    print(f"parity {invariance}\nchose {chosen} ({rule})\nwrote {artifact}")


if __name__ == "__main__":
    main(sys.argv[1:])
