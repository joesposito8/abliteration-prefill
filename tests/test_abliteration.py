"""Offline tests for the abliteration method code (no GPU); gated on torch."""

from __future__ import annotations

import pytest

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

needs_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")


def _stub_model(n_layers=3, d_model=8, d_attn=12, d_ff=16, vocab=20, *, tie=True):
    """Minimal module tree with the edited attribute paths; o_proj/down_proj non-square."""
    import torch.nn as nn

    layers = nn.ModuleList()
    for _ in range(n_layers):
        layer = nn.Module()
        layer.self_attn = nn.Module()
        layer.self_attn.o_proj = nn.Linear(d_attn, d_model, bias=False)
        layer.mlp = nn.Module()
        layer.mlp.down_proj = nn.Linear(d_ff, d_model, bias=False)
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

    targets = list(edit.target_matrices(_stub_model(n_layers=3)))
    assert len(targets) == 1 + 2 * 3
    assert targets[0][1] == 1 and all(axis == 0 for _, axis in targets[1:])
    d_model = targets[0][0].shape[1]
    assert all(w.shape[axis] == d_model for w, axis in targets)
    embed = targets[0][0]
    assert sum(w is embed for w, _ in targets) == 1  # tied embed present once


@needs_torch
def test_orthogonalize_removes_direction_from_every_matrix():
    from abliteration import edit

    model = _stub_model(d_model=8)
    r = torch.zeros(8)
    r[2] = 3.0  # unnormalized; orthogonalize_ normalizes internally
    before = {id(w): w.detach().clone() for w, _ in edit.target_matrices(model)}
    edit.orthogonalize_(model, r)

    r_hat = (r / torch.linalg.vector_norm(r)).to(torch.float32)
    for weight, axis in edit.target_matrices(model):
        w = weight.detach().to(torch.float32)
        projected = r_hat @ w if axis == 0 else w @ r_hat  # every d_model-vector ⟂ r_hat
        assert torch.allclose(projected, torch.zeros_like(projected), atol=1e-5)
        assert torch.allclose(w, edit._remove_direction(w, r_hat, axis), atol=1e-5)  # idempotent
        assert not torch.equal(w, before[id(weight)].to(torch.float32))


@needs_torch
def test_reconstruct_restores_base_weights_exactly():
    from abliteration import edit

    model = _stub_model()
    snapshot = edit.snapshot_targets(model)
    with edit.abliterated(model, torch.randn(8)):
        assert any(not torch.equal(w, b) for (w, _), b in zip(edit.target_matrices(model), snapshot))
    for (weight, _), saved in zip(edit.target_matrices(model), snapshot):
        assert torch.equal(weight.data, saved)


@needs_torch
def test_refusal_directions_recover_known_axis_and_normalize():
    from abliteration import directions

    harmless = torch.zeros(4, 6, dtype=torch.float64)
    harmful = torch.zeros(4, 6, dtype=torch.float64)
    harmful[1, 0], harmful[2, 4], harmful[3, 1] = 2.0, 5.0, 3.0  # every layer nonzero
    dirs = directions.refusal_directions(harmful, harmless)

    assert dirs.shape == (3, 6)  # embeddings row dropped
    expected = torch.zeros(6, dtype=torch.float64)
    expected[4] = 1.0
    assert torch.allclose(dirs[1], expected, atol=1e-9)  # layer 1 = hidden row 2


@needs_torch
def test_refusal_directions_reject_degenerate_layer():
    from abliteration import directions

    means = torch.ones(3, 5, dtype=torch.float64)
    with pytest.raises(ValueError):
        directions.refusal_directions(means, means)
