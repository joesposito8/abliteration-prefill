"""The tie-break: the one property the frozen ordering asserts about itself.

A tie broken by whichever order the rates arrived in would still print a plausible
ranking, so the rule is held here rather than read off the output. The rates are fed in
reverse-alphabetical order, which is the order a wrong implementation would preserve.
"""

from __future__ import annotations

from pilot_rank import rank


def test_a_tie_goes_to_the_alphabetically_earlier_level():
    rates = {"system_simulation": 0.5, "role_chaining": 0.5, "fake_citation": 0.9}
    assert rank(rates) == ["fake_citation", "role_chaining", "system_simulation"]


def test_a_cross_state_tie_goes_to_the_abliterated_cell():
    rates = {"base/none": 0.8, "abliterated/none": 0.8, "base/persona_switch": 0.9}
    assert rank(rates) == ["base/persona_switch", "abliterated/none", "base/none"]
