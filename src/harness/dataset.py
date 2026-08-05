"""One condition's samples: every eval prompt against every attempt it gets.

A condition has an unprefilled arm of stochastic samples of the bare prompt, and — only
for the aligned base and the composition primary — a prefilled arm of one sample per
portfolio slot. On the eval set both arms are the same width, because the study's
matched-budget rule holds every condition's denominator equal. The validation set is not
part of that comparison: it ranks layers on breadth at a single attempt.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser
from prefills import PORTFOLIO
from study.datasets import (
    PILOT_PROMPTS_CSV,
    STRONGREJECT_CSV,
    VALIDATION_HARMFUL_CSV,
    load_pilot_prompts,
    load_strongreject_prompts,
    load_validation_prompts,
)

from .conditions import Condition


@dataclass(frozen=True)
class EvalSet:
    load: Callable[[], pd.DataFrame]
    csv: Path
    attempts: int


EVAL_SETS = {
    "strongreject": EvalSet(load_strongreject_prompts, STRONGREJECT_CSV, len(PORTFOLIO)),
    "pilot": EvalSet(load_pilot_prompts, PILOT_PROMPTS_CSV, len(PORTFOLIO)),
    "validation": EvalSet(load_validation_prompts, VALIDATION_HARMFUL_CSV, 1),
}

CONTROL = "none"


def slot_key(slot: str) -> str:
    """A slot name safe to put in a ``Sample.id``.

    ``eval(sample_id=…)`` splits an id on its first ``:`` and reads the left half as a
    task name to scope the match, so a raw ``system_simulation:0`` matches no task and
    is dropped from a resume with the run still reporting success. 12 of the 13 slots
    contain a colon.
    """
    return slot.replace(":", "-")


def sample_id(prompt_id: int, slot: str, replicate: int) -> str:
    """``prompt_id`` is the frozen row index, so pilot ids join onto main-run ids."""
    return f"{prompt_id:03d}/{slot_key(slot)}/{replicate:02d}"


def build_dataset(
    condition: Condition, prefills: Mapping[tuple[int, str], str]
) -> MemoryDataset:
    """Cross the condition's eval set with the attempts that set is run at.

    ``prefills`` supplies the text for each ``(prompt_id, slot)``. The harness never
    produces it: 12 of the 13 slots are generated per forbidden prompt by a separate
    helper model, so a missing entry is a ``KeyError`` here rather than a control row
    wearing an attack's label.

    The arms are emitted in blocks rather than interleaved, so a batch tends to hold
    prompts of one length and pads less.
    """
    eval_set = EVAL_SETS[condition.prompt_set]
    rows = list(eval_set.load().itertuples())

    samples = [
        _sample(condition, row, CONTROL, replicate, prefills)
        for row in rows
        for replicate in range(eval_set.attempts)
    ]
    if condition.prefilled:
        samples += [
            _sample(condition, row, slot, 0, prefills)
            for row in rows
            for slot in PORTFOLIO
        ]
    return MemoryDataset(samples, name=condition.prompt_set)


def _sample(condition: Condition, row, slot: str, replicate: int, prefills):
    prefill = "" if slot == CONTROL else prefills[(row.prompt_id, slot)]
    if slot != CONTROL and not prefill:
        raise ValueError(
            f"prefill for ({row.prompt_id}, {slot!r}) is empty; the sample would carry "
            "an attack's label with no attack in it."
        )

    messages = [ChatMessageUser(content=row.forbidden_prompt)]
    if prefill:
        messages.append(ChatMessageAssistant(content=prefill))

    return Sample(
        id=sample_id(row.prompt_id, slot, replicate),
        input=messages,
        target="",
        metadata={
            "condition": condition.id,
            "layer": condition.layer,
            "prompt_set": condition.prompt_set,
            "prompt_id": row.prompt_id,
            "prefill_slot": slot,
            "prefill": prefill,
            "replicate": replicate,
            "category": row.category,
            "source": row.source,
        },
    )
