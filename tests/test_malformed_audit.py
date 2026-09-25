"""The malformed audit's draw and its unblinding arithmetic, on synthetic frames.

No test reads a log or any generated text.
"""

from __future__ import annotations

import pandas as pd
import pytest

import malformed_audit as ma


def population(sizes: dict[tuple[str, str, str], int]) -> pd.DataFrame:
    """A malformed population with the given rows per (model, cond, failed_call)."""
    rows = []
    for (model, cond, call), n in sizes.items():
        state = "base" if cond in ("B", "P") else "abliterated"
        for i in range(n):
            rows.append({
                "model": model, "condition": "base" if state == "base" else "layer_x",
                "state": state, "cond": cond, "prefilled": cond in ("P", "C"),
                "failed_call": call, "sample_id": f"{i:03d}/{cond}-{call}", "malformed": 1,
            })
    return pd.DataFrame(rows)


SIZES = {
    ("m", "P", "refusal"): 40, ("m", "P", "quality"): 40,
    ("m", "A", "refusal"): 30, ("m", "C", "refusal"): 90, ("m", "C", "quality"): 60,
}


def test_census_shows_every_cell_including_the_empty_ones():
    labels = population(SIZES)
    labels["malformed"] = 1
    table = ma.census(labels)

    assert len(table) == 8  # one model x 4 classes x 2 calls
    assert int(table[(table.cond == "B")].malformed.sum()) == 0
    assert int(table[(table.cond == "A") & (table.failed_call == "quality")].malformed.sum()) == 0
    assert int(table.malformed.sum()) == sum(SIZES.values())


def test_condition_class_is_total_over_state_and_prefill():
    assert [ma.condition_class(s, p) for s in ("base", "abliterated") for p in (False, True)] == ["B", "P", "A", "C"]
    with pytest.raises(KeyError):
        ma.condition_class("merged", True)


def test_the_key_allowlist_carries_no_text_column():
    assert not {"forbidden_prompt", "response", "continuation", "prefill", "raw", "note"} & set(ma.KEY_COLUMNS)


# --- unblinding ---------------------------------------------------------------


def answered(tmp_path, labels, wave1=None):
    """A key and an answers file on disk, the way `bundle` and `serve` leave them."""
    wave1 = wave1 if wave1 is not None else [f"M{i + 1:03d}" for i in range(len(labels))]
    key = pd.DataFrame({
        "item_id": wave1, "wave": 1, "model": "m", "condition": "base", "state": "base",
        "cond": "P", "failed_call": "refusal", "prefill_slot": "s:0", "family": "s",
        "prompt_id": 1, "sample_id": [f"{i}/s" for i in range(len(wave1))], "epoch": 1,
        "reply_digest": "d",
    })
    (tmp_path / "key").mkdir(parents=True, exist_ok=True)
    (tmp_path / "answers").mkdir(parents=True, exist_ok=True)
    key.to_csv(tmp_path / "key" / "key.csv", index=False)
    pd.DataFrame({"item_id": wave1, "label": labels, "forced": 0, "note": ""}).to_csv(
        tmp_path / "answers" / "malformed.csv", index=False)
    return tmp_path


def test_an_unlabelled_item_refuses_the_report(tmp_path):
    answered(tmp_path, ["unlock", "", "refusal"])
    with pytest.raises(SystemExit, match="no label"):
        ma.read_answers(tmp_path)


def test_a_missing_answer_refuses_the_report(tmp_path):
    root = answered(tmp_path, ["unlock", "refusal"])
    pd.DataFrame({"item_id": ["M001"], "label": ["unlock"], "forced": [0], "note": [""]}).to_csv(
        root / "answers" / "malformed.csv", index=False)
    with pytest.raises(SystemExit, match="answered"):
        ma.read_answers(root)


def test_shares_reweight_by_stratum_population():
    """Two strata sampled equally but populated unequally: the big one dominates."""
    frame = pd.DataFrame({
        "cond": "P", "cell": ["a"] * 10 + ["b"] * 10,
        "label": ["unlock"] * 10 + ["refusal"] * 10,
    })
    for label in ma.LABELS:
        frame[label] = (frame.label == label).astype(int)
    populations = pd.Series({"a": 900, "b": 100})

    row = ma.reweighted(frame, ["cond"], populations).iloc[0]
    assert row.unlock_share == pytest.approx(0.9)
    assert row.refusal_share == pytest.approx(0.1)
    assert row.population == 1000
    # The weighted Wilson envelope brackets the point estimate.
    assert row.unlock_lo < row.unlock_share < row.unlock_hi
