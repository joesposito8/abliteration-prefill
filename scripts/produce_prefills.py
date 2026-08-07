#!/usr/bin/env python3
"""Generate the portfolio's prefills, one wave of helper draws at a time.

The portfolio's rules are a sequential loop per cell: draw, validate, retry, dedup
against the variant before it. A 27B helper needs breadth to be affordable, and the only
breadth available is across the 1,878 cells rather than within one. So a wave is every
cell's next draw, generated together.

Which draws that is comes from the rules themselves rather than a second copy of them.
Between waves ``produce_family`` runs again against a generate_fn backed by what the
waves have already produced; it validates, retries and dedups exactly as it would live,
and the first draw it asks for that no wave has made is, by construction, the draw a live
run would make next. That one pass yields both the finished cells and the next wave.

Re-running the command resumes: every wave's draw list is derived, not stored, and
``eval_set`` picks each wave's log up from whatever it already holds.

Run:  python scripts/produce_prefills.py results/prefills /dev/shm/gemma
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from itertools import count
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from generation.gemma import load_helper  # noqa: E402
from harness.helper_provider import HELPER_BATCH, HELPER_PROVIDER  # noqa: E402
from harness.prefill_task import (  # noqa: E402
    HelperDraw,
    produce_prefills,
    wave_dataset,
)
from harness.run import run_eval_set  # noqa: E402
from inspect_ai.log import read_eval_log  # noqa: E402
from prefills import (  # noqa: E402
    FAMILIES,
    MAX_ATTEMPTS,
    MAX_DEDUP_RESAMPLES,
    VARIANTS_PER_FAMILY,
    PrefillResult,
    fill_prompt,
    load_prompt,
    produce_family,
)
from study import SEED  # noqa: E402
from study.datasets import load_strongreject_prompts  # noqa: E402

from grade import finished_logs  # noqa: E402

MODEL = f"{HELPER_PROVIDER}/prefills"

# Every draw one cell could ever be asked for, which is what a miss is looked up in.
CELL_BUDGET = MAX_ATTEMPTS + MAX_DEDUP_RESAMPLES


class _Missing(Exception):
    """A draw the rules asked for that no wave has generated."""

    def __init__(self, draw: HelperDraw) -> None:
        super().__init__(str(draw))
        self.draw = draw


def wave_dir(root: Path, index: int) -> Path:
    """One directory per wave, or a resume reads one as progress on another."""
    return root / f"wave_{index:02d}"


def drawn_before(root: Path, index: int) -> dict[int, str]:
    """What every wave before ``index`` generated, keyed by the frozen seed of its draw.

    Keyed by seed because that is all ``generate_fn`` is given: the rules derive a
    seed from coordinates, and the log carries the seed each sample was drawn under.
    """
    drawn: dict[int, str] = {}
    for wave in range(index):
        directory = wave_dir(root, wave)
        if not directory.exists():
            continue
        for header in finished_logs(directory):
            for sample in read_eval_log(header.location).samples:
                drawn[sample.metadata["helper_seed"]] = sample.output.completion
    return drawn


def _draw_for(prompt_id: int, family: str, seed: int) -> HelperDraw:
    """The draw a seed belongs to, searched within the one cell that could ask for it."""
    return next(
        draw
        for variant in range(VARIANTS_PER_FAMILY)
        for attempt in range(CELL_BUDGET)
        if (draw := HelperDraw(prompt_id, family, variant, attempt)).seed == seed
    )


def replay(drawn: Mapping[int, str]) -> tuple[dict, list[HelperDraw]]:
    """Run the frozen rules over what the waves have produced.

    Returns ``(results, missing)``: the cells the rules could settle, and the draws they
    asked for that no wave has made. A cell contributes to one or the other, never both,
    because the first miss stops it.
    """
    templates = {family: load_prompt(family) for family in FAMILIES}
    results: dict[tuple[int, str], list[PrefillResult]] = {}
    missing: list[HelperDraw] = []

    for row in load_strongreject_prompts().itertuples():
        for family in FAMILIES:

            def generate_fn(prompt, seed, prompt_id=row.prompt_id, family=family):
                if seed in drawn:
                    return drawn[seed]
                raise _Missing(_draw_for(prompt_id, family, seed))

            filled = fill_prompt(templates[family], row.forbidden_prompt)
            try:
                results[(row.prompt_id, family)] = produce_family(
                    family, filled, row.prompt_id, generate_fn
                )
            except _Missing as miss:
                missing.append(miss.draw)
    return results, missing


def waves(root: Path):
    """Each wave's draws in turn, until the rules ask for nothing new.

    Yields ``(index, draws)``. Nothing is drawn only before the first wave, since a wave
    that did not finish raises rather than returning.
    """
    for index in count():
        drawn = drawn_before(root, index)
        if not drawn:
            # Nothing generated yet, so every cell needs its opening draw. The replay
            # cannot derive this set: with nothing to serve, the first miss stops a
            # family before it reaches its second variant.
            draws = [
                HelperDraw(prompt_id, family, variant, attempt=0)
                for prompt_id in load_strongreject_prompts()["prompt_id"].tolist()
                for family in FAMILIES
                for variant in range(VARIANTS_PER_FAMILY)
            ]
        else:
            _, draws = replay(drawn)

        if not draws:
            return
        yield index, draws


def main(argv: list[str]) -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "no CUDA device; the locked torch replaces the image's, so check the "
            "install before spending GPU time"
        )
    root, snapshot = Path(argv[0]).resolve(), argv[1]

    model, tokenizer = load_helper(snapshot)
    for index, draws in waves(root):
        print(f"wave {index}: {len(draws):,} draws", flush=True)
        log = run_eval_set(
            produce_prefills(wave_dataset(draws), seed=SEED),
            log_dir=str(wave_dir(root, index)),
            model=MODEL,
            module=model,
            tokenizer=tokenizer,
            label=f"wave {index}",
            log_buffer=HELPER_BATCH,
        )
        print(f"  {log.results.completed_samples:,} generated", flush=True)
    print("every cell settled")


if __name__ == "__main__":
    main(sys.argv[1:])
