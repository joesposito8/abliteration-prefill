"""One condition's samples: every eval prompt crossed with every prefill slot."""

from __future__ import annotations

from collections.abc import Mapping

from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser
from prefills import PORTFOLIO
from study.datasets import load_pilot_prompts, load_strongreject

from .conditions import Condition

# Only the two sets that are eval sets. The extraction and validation splits are
# deliberately unreachable from here.
EVAL_SETS = {
    "strongreject": load_strongreject,
    "pilot": load_pilot_prompts,
}

CONTROL = "none"

# The unprefilled control alongside the 13 portfolio slots.
SLOTS: tuple[str | None, ...] = (None, *PORTFOLIO)


def slot_key(slot: str | None) -> str:
    """A slot name safe to put in a ``Sample.id``.

    ``eval(sample_id=…)`` splits an id on its first ``:`` and reads the left half as a
    task name to scope the match, so a raw ``system_simulation:0`` matches no task and
    is dropped from a resume with the run still reporting success. 12 of the 13 slots
    contain a colon.
    """
    return (slot or CONTROL).replace(":", "-")


def sample_id(prompt_id: int, slot: str | None) -> str:
    """``prompt_id`` is the frozen row index, so pilot ids join onto main-run ids."""
    return f"{prompt_id:03d}/{slot_key(slot)}"


def build_dataset(
    condition: Condition, prefills: Mapping[tuple[int, str], str]
) -> MemoryDataset:
    """Cross the condition's eval set with every slot.

    ``prefills`` supplies the text for each ``(prompt_id, slot)``. The harness never
    produces it: 12 of the 13 slots are generated per forbidden prompt by a separate
    helper model, so a missing entry is a ``KeyError`` here rather than a control row
    wearing an attack's label.
    """
    rows = EVAL_SETS[condition.prompt_set]()
    return MemoryDataset(
        [
            _sample(condition, row, slot, prefills)
            for row in rows.itertuples()
            for slot in SLOTS
        ],
        name=condition.prompt_set,
    )


def _sample(condition: Condition, row, slot: str | None, prefills) -> Sample:
    prefill = prefills[(row.prompt_id, slot)] if slot is not None else ""
    if slot is not None and not prefill:
        raise ValueError(
            f"prefill for ({row.prompt_id}, {slot!r}) is empty; the sample would carry "
            "an attack's label with no attack in it."
        )

    messages = [ChatMessageUser(content=row.forbidden_prompt)]
    if prefill:
        messages.append(ChatMessageAssistant(content=prefill))

    return Sample(
        id=sample_id(row.prompt_id, slot),
        input=messages,
        target="",
        metadata={
            "condition": condition.id,
            "layer": condition.layer,
            "prompt_set": condition.prompt_set,
            "prompt_id": row.prompt_id,
            # The portfolio's own name, colon intact, unlike the id above.
            "prefill_slot": slot or CONTROL,
            # Grading runs in a later process and cannot re-derive this.
            "prefill": prefill,
            "category": row.category,
            "source": row.source,
        },
    )
