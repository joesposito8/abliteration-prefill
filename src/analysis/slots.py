"""Per-slot reads of a factorial at one draw per slot.

The variant slots are pooled into families everywhere a reported metric is computed, so
this is the one place that keeps them apart. A tree at one draw per slot is not at the
study's draw schedule and ``Counts`` refuses it, which is why these take a frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from harness.dataset import CONTROL


def slot_table(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per (condition, slot): draws, unlocks, the unlock rate and the malformed
    share. Malformed rows stay in the denominator as non-unlocks."""
    return (
        frame.groupby(["condition", "slot"])
        .agg(
            draws=("unlocked", "size"),
            unlocked=("unlocked", "sum"),
            rate=("unlocked", "mean"),
            malformed=("malformed", "mean"),
        )
        .reset_index()
    )


def rates(table: pd.DataFrame) -> pd.DataFrame:
    """The prefilled slots' rates, one column per condition. The unprefilled control is
    dropped: it is the comparator, not a member of the portfolio."""
    return table.pivot(index="slot", columns="condition", values="rate").drop(index=CONTROL)


def spread(table: pd.DataFrame, condition: str) -> dict:
    """How far apart the prefill slots are on one arm -- whether the portfolio's members
    are interchangeable there."""
    held = rates(table)[condition]
    return {
        "condition": condition,
        "slots": int(held.size),
        "min": float(held.min()),
        "max": float(held.max()),
        "spread": float(held.max() - held.min()),
        "sd": float(held.std()),
    }


def against_comparator(table: pd.DataFrame, prefilled: str, composed: str) -> pd.DataFrame:
    """Per slot, what composing wins against the comparator an attacker holding the
    checkpoint actually has.

    The naive gain subtracts the prefill alone, which credits composition for everything
    abliteration was already doing. That attacker can always decline to prefill, so the
    comparator is the better of the slot's own prefill and the composed arm's control.
    """
    comparator = table.set_index(["condition", "slot"]).rate[(composed, CONTROL)]
    held = rates(table)
    return pd.DataFrame(
        {
            "slot": held.index,
            "prefill_only": held[prefilled].to_numpy(),
            "composed": held[composed].to_numpy(),
            "naive_gain": (held[composed] - held[prefilled]).to_numpy(),
            "over_comparator": (held[composed] - np.maximum(held[prefilled], comparator)).to_numpy(),
        }
    )
