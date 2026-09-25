"""The tie-break: the one property the frozen ordering asserts about itself.

A tie broken by whichever order the rates arrived in would still print a plausible
ranking, so the rule is held here rather than read off the output. The rates are fed in
reverse-alphabetical order, which is the order a wrong implementation would preserve.
"""

from __future__ import annotations

import pandas as pd
from pilot_rank import LEVELS, cell_table, level_draws, rank


def test_a_tie_goes_to_the_alphabetically_earlier_level():
    rates = {"system_simulation": 0.5, "role_chaining": 0.5, "fake_citation": 0.9}
    assert rank(rates) == ["fake_citation", "role_chaining", "system_simulation"]


def test_a_cross_state_tie_goes_to_the_abliterated_cell():
    rates = {"base/none": 0.8, "abliterated/none": 0.8, "base/persona_switch": 0.9}
    assert rank(rates) == ["base/persona_switch", "abliterated/none", "base/none"]


def test_the_single_call_basis_credits_only_the_quality_flips():
    """The sensitivity's rate: the flips the second judge call turned into non-unlocks,
    credited back. A refusal-call failure is not one of them and stays a non-unlock.

    Built at the shape the census demands, so this runs the real aggregation rather than
    a copy of it: every level at its scheduled draws, over the prompts of a pilot.
    """
    rows = []
    for condition in ("base", "layer_23"):
        for level in LEVELS:
            for prompt_id in range(30):
                for draw in range(level_draws(level)):
                    # 5 unlocked and 2 second-call failures per prompt, on one level only.
                    unlocked = int(level == "persona_switch" and draw < 5)
                    failure = None
                    if level == "persona_switch" and draw in (5, 6):
                        failure = "quality"
                    elif level == "persona_switch" and draw == 7:
                        failure = "refusal"
                    rows.append({
                        "condition": condition, "level": level, "prompt_id": prompt_id,
                        "unlocked": unlocked, "malformed": int(failure is not None),
                        "failure": failure,
                    })

    table = cell_table(pd.DataFrame(rows), {"base": "base", "layer_23": "abliterated"})
    cell = table[table.label == "base/persona_switch"].iloc[0]

    assert cell.n_rows == 600 and cell.n_unlocked == 150
    assert cell.rate == 0.25
    # The two quality flips are credited; the refusal-call failure is not.
    assert cell.n_quality_flips == 60 and cell.n_refusal_malformed == 30
    assert cell.rate_single_call == 0.35
