"""One condition's samples: every eval prompt crossed with every prefill level and draw.

A condition's cells are its prompts crossed with the levels it carries — the bare prompt
alone, or the bare prompt and every portfolio slot — and how many draws a cell gets is a
function of its slot, not one number for the run. ``none`` is the control level of the
prefill factor rather than a separate arm, which is why one comprehension covers both. The
validation set is not part of the factorial: it ranks layers on breadth at a single draw.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser
from prefills import PORTFOLIO
from study import SEED, draws
from study.datasets import (
    STRONGREJECT_CSV,
    VALIDATION_HARMFUL_CSV,
    load_strongreject_prompts,
    load_validation_prompts,
    sample_indices,
)

from .conditions import Condition


@dataclass(frozen=True)
class EvalSet:
    load: Callable[[], pd.DataFrame]
    csv: Path
    draws: Callable[[str], int]


EVAL_SETS = {
    "strongreject": EvalSet(load_strongreject_prompts, STRONGREJECT_CSV, draws),
    "validation": EvalSet(load_validation_prompts, VALIDATION_HARMFUL_CSV, lambda slot: 1),
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
    """Cross the condition's eval set with the prefill levels and draws it is run at.

    ``prefills`` supplies the text for each ``(prompt_id, slot)``. The harness never
    produces it: 12 of the 13 slots are generated per forbidden prompt by a separate
    helper model, so a missing entry is a ``KeyError`` here rather than a control row
    wearing an attack's label.

    Slot-major, so a batch carries one prefill, and a cell's draws are consecutive.
    """
    eval_set = EVAL_SETS[condition.prompt_set]
    frame = eval_set.load()
    if condition.num_prompts is not None:
        frame = frame.iloc[sample_indices(len(frame), condition.num_prompts, SEED)]
    rows = list(frame.itertuples())
    slots = (CONTROL, *PORTFOLIO) if condition.prefilled else (CONTROL,)

    return MemoryDataset(
        [
            _sample(condition, row, slot, replicate, prefills)
            for slot in slots
            for row in rows
            for replicate in range(eval_set.draws(slot))
        ],
        name=condition.prompt_set,
    )


def _sample(condition: Condition, row, slot: str, replicate: int, prefills):
    messages = [ChatMessageUser(content=row.forbidden_prompt)]
    prefill = ""
    if slot != CONTROL:
        prefill = prefills[(row.prompt_id, slot)]
        if not prefill:
            raise ValueError(
                f"prefill for ({row.prompt_id}, {slot!r}) is empty; the sample would "
                "carry an attack's label with no attack in it."
            )
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
