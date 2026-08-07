#!/usr/bin/env python3
"""Measure the helper's batch width, and what the run costs at the one that wins.

Unlike the target's, this width is not a frozen study parameter: each prefill is produced
once, committed and hashed, and read identically by every downstream condition, so no
comparison depends on the width it was drawn at. Nothing has to be kept invariant, so
there is no parity pass here — only what each width costs.

Run:  python scripts/helper_batch_sweep.py /dev/shm/gemma [artifact-path]
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from generation.batched import generate_prompts  # noqa: E402
from generation.gemma import HELPER_DECODING, load_helper, render_prompt  # noqa: E402
from prefills import (  # noqa: E402
    FAMILIES,
    HELPER_MODEL,
    HELPER_REVISION,
    VARIANTS_PER_FAMILY,
    fill_prompt,
    load_prompt,
)
from study.datasets import load_strongreject_prompts  # noqa: E402
from study.manifest import write_manifest  # noqa: E402

from batch_sweep import choose_width, kv_bytes_per_token  # noqa: E402

ARTIFACT = REPO / "data" / "helper_batch_sweep.json"
SEED = 20260807
WIDTHS = (8, 16, 32)

# 27B in BF16 is ~54 GB of weights before a single KV entry, on an 80 GB card.
VRAM_CEILING_GB = 72
TOLERANCE = 0.10

# Secure-cloud A100-SXM4-80GB.
DOLLARS_PER_HOUR = 1.59


def workload(tokenizer, rows: int) -> list[str]:
    """Real helper prompts, cycling the families so template lengths vary as they will."""
    templates = [load_prompt(family) for family in FAMILIES]
    forbidden = load_strongreject_prompts()["forbidden_prompt"].head(rows)
    return [
        render_prompt(tokenizer, fill_prompt(templates[index % len(templates)], prompt))
        for index, prompt in enumerate(forbidden)
    ]


def measure(model, tokenizer, prompts: list[str], width: int) -> dict:
    """One width, warmed up first so the first call's autotuning stays out of the clock."""
    import torch

    batch = prompts[:width]
    generate_prompts(model, tokenizer, batch, seed=SEED, decoding=HELPER_DECODING)

    torch.cuda.reset_peak_memory_stats()
    rows, seconds = generate_prompts(
        model, tokenizer, batch, seed=SEED, decoding=HELPER_DECODING
    )
    new_tokens = [row.new_tokens for row in rows]
    longest = max(row.prompt_tokens for row in rows) + HELPER_DECODING["max_new_tokens"]

    return {
        "width": width,
        "seconds": seconds,
        "prompts_per_second": width / seconds,
        "tokens_per_second": sum(new_tokens) / seconds,
        # The cost estimate assumed 300; five of the six families emit a sentence or two.
        "mean_new_tokens": sum(new_tokens) / len(new_tokens),
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30,
        "predicted_kv_gb": width * longest * kv_bytes_per_token(model.config.text_config) / 2**30,
    }


def projection(measurement: dict) -> dict:
    """What the opening wave costs at this width, which is what decides whether to run."""
    draws = len(load_strongreject_prompts()) * len(FAMILIES) * VARIANTS_PER_FAMILY
    hours = draws / measurement["prompts_per_second"] / 3600
    return {"draws": draws, "hours": hours, "dollars": hours * DOLLARS_PER_HOUR}


def main(argv: list[str]) -> None:
    import torch
    import transformers

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device")

    snapshot = argv[0]
    artifact = Path(argv[1]).resolve() if len(argv) > 1 else ARTIFACT

    model, tokenizer = load_helper(snapshot)
    prompts = workload(tokenizer, max(WIDTHS))
    widths = [measure(model, tokenizer, prompts, width) for width in WIDTHS]
    chosen, rule = choose_width(widths, ceiling=VRAM_CEILING_GB, tolerance=TOLERANCE)
    winner = next(measurement for measurement in widths if measurement["width"] == chosen)

    result = {
        "model": HELPER_MODEL,
        "revision": HELPER_REVISION,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "seed": SEED,
        "decoding": dict(HELPER_DECODING),
        "widths": widths,
        "chosen": winner,
        "rule": rule,
        "opening_wave": projection(winner),
    }
    write_manifest(artifact, result)

    wave = result["opening_wave"]
    for measurement in widths:
        print(f"  width {measurement['width']:>3}  "
              f"{measurement['prompts_per_second']:.2f} prompts/s  "
              f"{measurement['tokens_per_second']:.0f} tok/s  "
              f"{measurement['peak_vram_gb']:.1f} GiB  "
              f"mean {measurement['mean_new_tokens']:.0f} new tokens")
    print(f"\nchose {chosen} ({rule})")
    print(f"opening wave: {wave['draws']:,} draws, "
          f"{wave['hours']:.2f} h, ${wave['dollars']:.2f}")
    print(f"wrote {artifact}")
    print(f"\nSet HELPER_BATCH = {chosen} in src/harness/helper_provider.py before running "
          "the waves.")


if __name__ == "__main__":
    main(sys.argv[1:])
