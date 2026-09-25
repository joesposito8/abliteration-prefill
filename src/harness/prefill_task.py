"""One wave of helper draws: which calls to make, and the eval that makes them.

A draw is one seeded helper call. The portfolio's rules decide which draws a cell needs
— up to three attempts for a valid prefill, then up to three more to break a duplicate —
but they decide it one outcome at a time, and an Inspect sample is one-shot. So a wave is
a flat list of draws whose outcomes are all still unknown, and the loop that reads those
outcomes and asks for the next wave lives outside the eval, in
``scripts/produce_prefills.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from inspect_ai import Task, task
from inspect_ai.dataset import Dataset, MemoryDataset, Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate
from prefills import FAMILIES, fill_prompt, helper_seed, load_prompt
from study.datasets import load_strongreject_prompts

from .dataset import sample_id
from .helper_provider import HELPER_FROZEN_CONFIG
from .meta import prefill_metadata


@dataclass(frozen=True)
class HelperDraw:
    """One seeded helper call, named by the coordinates ``helper_seed`` is derived from."""

    prompt_id: int
    family: str
    variant: int
    attempt: int

    @property
    def seed(self) -> int:
        return helper_seed(self.prompt_id, self.family, self.variant, self.attempt)

    @property
    def slot(self) -> str:
        """The portfolio slot this draw is a candidate for."""
        return f"{self.family}:{self.variant}"


def wave_dataset(draws: Sequence[HelperDraw]) -> MemoryDataset:
    """One sample per draw, each carrying the seed the frozen rule gives it.

    ``sample_id`` puts the slot through ``slot_key``, so the colon in a slot name never
    reaches an id: ``eval(sample_id=…)`` splits on the first one and reads the left half
    as a task name, which drops the sample from a resume while the run still reports
    success.
    """
    forbidden = load_strongreject_prompts().set_index("prompt_id")["forbidden_prompt"]
    templates = {family: load_prompt(family) for family in FAMILIES}

    return MemoryDataset(
        [
            Sample(
                id=sample_id(draw.prompt_id, draw.slot, draw.attempt),
                input=fill_prompt(templates[draw.family], forbidden[draw.prompt_id]),
                target="",
                metadata={
                    "prompt_id": draw.prompt_id,
                    "family": draw.family,
                    "variant": draw.variant,
                    "attempt": draw.attempt,
                    "helper_seed": draw.seed,
                },
            )
            for draw in draws
        ],
        name="prefills",
    )


@task
def produce_prefills(dataset: Dataset, seed: int) -> Task:
    """One wave's eval.

    ``seed`` covers the whole wave rather than one draw. ``batching._join`` refuses a
    batch whose members disagree on it, and every draw's ``helper_seed`` differs — so
    the frozen per-draw seeds are carried as sample metadata instead. What they are for
    still holds: an attempt only ever runs in a later wave than the one it retries, and
    so in a different batch, under a different stream.
    """
    if isinstance(dataset, str):
        raise TypeError(f"dataset must be a Dataset, not the name {dataset!r}")
    return Task(
        dataset=dataset,
        solver=generate(),
        config=HELPER_FROZEN_CONFIG.merge(GenerateConfig(seed=seed)),
        metadata=prefill_metadata(),
    )
