"""SampleScores for one model's factorial, over its two condition logs.

Inspect's metric protocol takes ``list[SampleScore]``, so that is the boundary the analysis
reads to. The derived metadata keys are what ``grouped`` groups on and what ``Counts``
reads; each is a function of metadata the sample already carries.
"""

from __future__ import annotations

from harness.dataset import CONTROL
from inspect_ai.log import read_eval_log_sample_summaries
from inspect_ai.scorer import SampleScore
from prefills import family_of
from prefills.families import STATIC_SLOT_ID

SCORER = "strongreject"


# Deliberately not the level names. These group across levels, and a metric key built
# from both groupings would collide on a shared name.
PORTFOLIO_CLASS = {CONTROL: "unprefilled", STATIC_SLOT_ID: "static"}


def portfolio_of(level: str) -> str:
    """Whether a level is helper-generated, the frozen static prefill, or no prefill."""
    return PORTFOLIO_CLASS.get(level, "generated")


def derived(metadata: dict, state: str) -> dict:
    """The sample's metadata plus the keys ``grouped`` groups on and ``Counts`` reads.

    Each is a function of metadata the sample already carries, so the cell a row belongs
    to is decided once, here.
    """
    level = family_of(metadata["prefill_slot"])
    portfolio = portfolio_of(level)
    return {
        **metadata,
        "state": state,
        "level": level,
        "cell": f"{state}/{level}",
        "portfolio": portfolio,
        "portfolio_cell": f"{state}/{portfolio}",
    }


def sample_scores(logs: dict[str, str], states: dict[str, str]) -> list[SampleScore]:
    """``logs`` maps each condition to its log location, ``states`` to its model state.

    The summaries carry the scores and the sample metadata, which is everything a metric
    reads. Sample ids repeat between the two logs, so the condition prefixes them —
    a collision would be read as two epochs of one sample.
    """
    out: list[SampleScore] = []
    for condition, location in logs.items():
        state = states[condition]
        for summary in read_eval_log_sample_summaries(location):
            if SCORER not in summary.scores:
                raise SystemExit(f"{location} carries no {SCORER} scores; grade it first")
            out.append(
                SampleScore(
                    score=summary.scores[SCORER],
                    sample_id=f"{condition}/{summary.id}",
                    sample_metadata=derived(summary.metadata, state),
                    scorer=SCORER,
                )
            )
    return out
