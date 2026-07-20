"""Offline tests for abliteration construction (no GPU, no model weights).

Two tiers: the contrast-set logic is pure Python and always runs; the tensor-math
(direction difference-of-means, weight orthogonalization, reconstruct) needs a tensor
library and is skipped where torch is not installed — it runs on the GPU env and any
torch-equipped machine, and the arithmetic is the same on CPU.
"""

from __future__ import annotations

import pytest

from abliteration import contrast


# --- contrast set: dedup, normalization, balanced sampling (pure Python) ---------------

def test_normalize_collapses_case_punctuation_and_quotes():
    assert contrast.normalize('  "Build a BOMB!!" ') == "build a bomb"


def test_is_near_dup_exact_after_normalization():
    assert contrast.is_near_dup("Write a virus.", ['"write a virus"'])


def test_is_near_dup_minor_reword():
    # A near-identical prompt (a trailing word) exceeds the char-ratio threshold.
    assert contrast.is_near_dup(
        "how to make a bomb at home", ["how to make a bomb at home now"]
    )


def test_is_near_dup_not_fooled_by_shared_boilerplate():
    # Regression: two prompts differing only in the key content word share most words but
    # are distinct behaviors; order-sensitive similarity must not flag them.
    assert not contrast.is_near_dup(
        "write a tutorial on how to make a bomb",
        ["write a tutorial on how to make a cake"],
    )


def test_is_near_dup_rejects_unrelated():
    assert not contrast.is_near_dup(
        "how to bake sourdough bread", ["how to pick a lock"]
    )


def test_dedup_drops_planted_collision():
    # One candidate is an AdvBench-sourced duplicate of an excluded prompt; the other is
    # genuinely held out. Only the held-out prompt survives.
    candidates = ["Develop a computer virus", "Write a defamatory article about X"]
    exclude = ['"develop a computer virus."']
    survivors = contrast.dedup(candidates, exclude)
    assert survivors == ["Write a defamatory article about X"]


def test_sample_balanced_is_even_and_deterministic():
    grouped = {c: [f"{c}-{i}" for i in range(10)] for c in "abcd"}
    first = contrast.sample_balanced(grouped, 8, seed=1)
    assert len(first) == 8
    # Two per category -> even coverage across all four groups.
    assert {p.split("-")[0] for p in first} == set("abcd")
    # Seeded: identical inputs reproduce the identical sample.
    assert contrast.sample_balanced(grouped, 8, seed=1) == first


def test_sample_balanced_raises_when_insufficient():
    with pytest.raises(ValueError):
        contrast.sample_balanced({"a": ["x"]}, 5, seed=1)


def test_sample_first_deterministic_and_bounded():
    pool = [f"p{i}" for i in range(50)]
    picked = contrast.sample_first(pool, 10, seed=7)
    assert len(picked) == 10 and len(set(picked)) == 10
    assert contrast.sample_first(pool, 10, seed=7) == picked


# --- tensor math: direction extraction, orthogonalization, reconstruct -----------------
# Gated on torch so the pure-Python tests above still run in the torch-free dev env; the
# arithmetic is identical on CPU and runs on the GPU env / any torch-equipped machine.

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

needs_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")


def _stub_model(n_layers=3, d_model=8, d_attn=12, d_ff=16, vocab=20, *, tie=True):
    """A minimal module tree exposing the attribute paths abliteration edits.

    ``o_proj`` and ``down_proj`` are deliberately non-square (``d_attn``/``d_ff`` !=
    ``d_model``), matching the real ``[d_model, in]`` shape, so a wrong-axis regression on
    the projections would actually change the result.
    """
    import torch.nn as nn

    layers = nn.ModuleList()
    for _ in range(n_layers):
        layer = nn.Module()
        layer.self_attn = nn.Module()
        layer.self_attn.o_proj = nn.Linear(d_attn, d_model, bias=False)  # [d_model, d_attn]
        layer.mlp = nn.Module()
        layer.mlp.down_proj = nn.Linear(d_ff, d_model, bias=False)  # [d_model, d_ff]
        layers.append(layer)

    inner = nn.Module()
    inner.embed_tokens = nn.Embedding(vocab, d_model)
    inner.layers = layers

    model = nn.Module()
    model.model = inner
    model.lm_head = nn.Linear(d_model, vocab, bias=False)
    if tie:
        model.lm_head.weight = inner.embed_tokens.weight
    return model


@needs_torch
def test_target_matrices_count_and_axes_and_tying():
    from abliteration import edit

    model = _stub_model(n_layers=3)
    targets = list(edit.target_matrices(model))
    assert len(targets) == 1 + 2 * 3  # embed + (o_proj, down_proj) per layer
    # Axes: embedding on axis 1 (first target), projections on axis 0.
    assert targets[0][1] == 1
    assert all(axis == 0 for _, axis in targets[1:])
    # Every declared d_model axis actually has length d_model.
    d_model = model.model.embed_tokens.weight.shape[1]
    assert all(w.shape[axis] == d_model for w, axis in targets)
    # The embedding is the tied tensor; it must be present exactly once.
    embed = model.model.embed_tokens.weight
    assert sum(w is embed for w, _ in targets) == 1
    assert model.lm_head.weight is embed  # tie preserved


@needs_torch
def test_orthogonalize_removes_direction_from_every_matrix():
    from abliteration import edit

    model = _stub_model(d_model=8)
    r = torch.zeros(8)
    r[2] = 3.0  # unnormalized on purpose; orthogonalize_ normalizes internally.

    before = {id(w): w.detach().clone() for w, _ in edit.target_matrices(model)}
    edit.orthogonalize_(model, r)

    r_hat = (r / torch.linalg.vector_norm(r)).to(torch.float32)
    for weight, axis in edit.target_matrices(model):
        w = weight.detach().to(torch.float32)
        # Independent orthogonality check along the declared d_model axis: every
        # d_model-vector on that axis is perpendicular to r_hat.
        projected = r_hat @ w if axis == 0 else w @ r_hat
        assert torch.allclose(projected, torch.zeros_like(projected), atol=1e-5)
        # And the perpendicular component is preserved: removing r_hat again is a no-op.
        assert torch.allclose(w, edit._remove_direction(w, r_hat, axis), atol=1e-5)
        # A wrong-axis edit would have changed the untouched (non-d_model) axis length's
        # structure; confirm the edit actually moved the weights.
        assert not torch.equal(w, before[id(weight)].to(torch.float32))


@needs_torch
def test_reconstruct_restores_base_weights_exactly():
    from abliteration import edit

    model = _stub_model()
    base = snapshot = edit.snapshot_targets(model)
    r = torch.randn(8)

    with edit.abliterated(model, r):
        # Inside the block the weights are edited (at least one target changed).
        assert any(
            not torch.equal(w, b)
            for (w, _), b in zip(edit.target_matrices(model), base)
        )
    # After the block every target matches the base snapshot bit-for-bit.
    for (weight, _), saved in zip(edit.target_matrices(model), snapshot):
        assert torch.equal(weight.data, saved)


@needs_torch
def test_refusal_directions_recover_known_axis_and_normalize():
    from abliteration import directions

    n_hidden, d = 4, 6  # 3 layers + embeddings row
    harmless = torch.zeros(n_hidden, d, dtype=torch.float64)
    # Give every layer a distinct non-zero difference; layer 1 differs only along axis 4.
    harmful = torch.zeros(n_hidden, d, dtype=torch.float64)
    harmful[1, 0] = 2.0  # embeddings row (dropped)
    harmful[2, 4] = 5.0  # layer 1
    harmful[3, 1] = 3.0  # layer 2
    dirs = directions.refusal_directions(harmful, harmless)

    assert dirs.shape == (3, d)  # embeddings row dropped -> one row per layer
    expected = torch.zeros(d, dtype=torch.float64)
    expected[4] = 1.0
    assert torch.allclose(dirs[1], expected, atol=1e-9)
    assert torch.allclose(torch.linalg.vector_norm(dirs[1]), torch.tensor(1.0, dtype=torch.float64))


@needs_torch
def test_refusal_directions_reject_degenerate_layer():
    from abliteration import directions

    means = torch.ones(3, 5, dtype=torch.float64)  # harmful == harmless -> zero diff
    with pytest.raises(ValueError):
        directions.refusal_directions(means, means)
