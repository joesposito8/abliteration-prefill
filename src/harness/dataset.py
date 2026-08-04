"""One condition's samples: every eval prompt against every attempt it gets.

A condition has an unprefilled arm of ``ATTEMPTS`` stochastic samples of the bare prompt,
and — only for the aligned base and the composition primary — a prefilled arm of one
sample per portfolio slot. Both arms are the same width because the study's matched-budget
rule holds every condition's denominator equal.
"""

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

# Matched budget: as many plain attempts as there are prefills.
ATTEMPTS = len(PORTFOLIO)


def slot_key(slot: str | None) -> str:
    """A slot name safe to put in a ``Sample.id``.

    ``eval(sample_id=…)`` splits an id on its first ``:`` and reads the left half as a
    task name to scope the match, so a raw ``system_simulation:0`` matches no task and
    is dropped from a resume with the run still reporting success. 12 of the 13 slots
    contain a colon.
    """
    return (slot or CONTROL).replace(":", "-")


def sample_id(prompt_id: int, slot: str | None, replicate: int) -> str:
    """``prompt_id`` is the frozen row index, so pilot ids join onto main-run ids."""
    return f"{prompt_id:03d}/{slot_key(slot)}/{replicate:02d}"


def build_dataset(
    condition: Condition, prefills: Mapping[tuple[int, str], str]
) -> MemoryDataset:
    """Cross the condition's eval set with the attempts that condition makes.

    ``prefills`` supplies the text for each ``(prompt_id, slot)``. The harness never
    produces it: 12 of the 13 slots are generated per forbidden prompt by a separate
    helper model, so a missing entry is a ``KeyError`` here rather than a control row
    wearing an attack's label.

    The arms are emitted in blocks rather than interleaved, so a batch tends to hold
    prompts of one length and pads less.
    """
    rows = list(EVAL_SETS[condition.prompt_set]().itertuples())

    samples = [
        _sample(condition, row, None, replicate, prefills)
        for row in rows
        for replicate in range(ATTEMPTS)
    ]
    if condition.prefilled:
        samples += [
            _sample(condition, row, slot, 0, prefills)
            for row in rows
            for slot in PORTFOLIO
        ]
    return MemoryDataset(samples, name=condition.prompt_set)


def _sample(condition: Condition, row, slot: str | None, replicate: int, prefills):
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
        id=sample_id(row.prompt_id, slot, replicate),
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
            "replicate": replicate,
            "category": row.category,
            "source": row.source,
        },
    )
