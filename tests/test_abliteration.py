"""Offline tests for the abliteration method code (no GPU); gated on torch."""

from __future__ import annotations

import pytest

from conftest import needs_torch

try:
    import torch
except ImportError:  # every test that uses it is skipped
    pass


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
@pytest.mark.parametrize("tie", [True, False], ids=["tied", "untied"])
def test_target_matrices_count_and_axes(tie):
    """Both tyings, because the targets differ: Qwen3-4B ties lm_head to embed_tokens
    and phi-4 does not, and neither changes what is edited."""
    from abliteration import edit

    model = _stub_model(n_layers=3, tie=tie)
    targets = list(edit.target_matrices(model))
    assert len(targets) == 1 + 2 * 3
    assert targets[0][1] == 1 and all(axis == 0 for _, axis in targets[1:])
    d_model = targets[0][0].shape[1]
    assert all(w.shape[axis] == d_model for w, axis in targets)
    embed = targets[0][0]
    assert sum(w is embed for w, _ in targets) == 1
    assert all(w is not model.lm_head.weight for w, _ in targets[1:])


@needs_torch
def test_the_declared_layer_count_must_match_what_the_walk_reaches():
    """A short walk edits fewer matrices than the condition claims, and every number
    downstream still reads as a full edit."""
    from abliteration import edit

    edit.check_inventory(_stub_model(n_layers=3), 3)
    with pytest.raises(ValueError, match="expected 73 target matrices, found 7"):
        edit.check_inventory(_stub_model(n_layers=3), 36)


@needs_torch
def test_max_projection_is_near_zero_only_after_the_edit():
    """The number the pod records on real weights."""
    from abliteration import edit

    model = _stub_model(d_model=8)
    r = torch.randn(8)

    before = edit.max_projection(model, r)
    edit.orthogonalize_(model, r)

    assert before > 1e-3
    assert edit.max_projection(model, r) < 1e-5


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
def test_save_load_directions_round_trip_on_cpu(tmp_path):
    from abliteration import directions

    dirs = torch.randn(4, 6, dtype=torch.float64)
    path = tmp_path / "nested" / "directions.pt"
    sha = directions.save_directions(dirs, path)

    loaded = directions.load_directions(path)
    assert loaded.device.type == "cpu"  # must load without CUDA
    assert loaded.dtype == torch.float64
    assert torch.equal(loaded, dirs)
    # torch.save names the zip entries after the file stem, so the hash is stable across
    # directories but not across renames: produce and commit under one basename.
    assert sha == directions.save_directions(dirs, tmp_path / "elsewhere" / "directions.pt")
    assert sha != directions.save_directions(dirs, tmp_path / "renamed.pt")


@needs_torch
def test_orthogonalize_from_loaded_file_matches_in_memory(tmp_path):
    from abliteration import directions, edit

    r = torch.randn(8, dtype=torch.float64)
    path = tmp_path / "directions.pt"
    directions.save_directions(r.unsqueeze(0), path)

    from_memory = _stub_model(d_model=8)
    from_file = _stub_model(d_model=8)
    edit.restore_targets(from_file, edit.snapshot_targets(from_memory))
    edit.orthogonalize_(from_memory, r)
    edit.orthogonalize_(from_file, directions.load_directions(path)[0])

    for (a, _), (b, _) in zip(edit.target_matrices(from_memory), edit.target_matrices(from_file)):
        assert torch.equal(a, b)


@needs_torch
def test_repeated_edit_restore_cycles_do_not_drift():
    from abliteration import edit

    model = _stub_model(d_model=8)
    for weight, _ in edit.target_matrices(model):
        weight.data = weight.data.to(torch.bfloat16)
    snapshot = edit.snapshot_targets(model)

    for _ in range(3):
        edit.orthogonalize_(model, torch.randn(8))
        edit.restore_targets(model, snapshot)
        for (weight, _), saved in zip(edit.target_matrices(model), snapshot):
            assert torch.equal(weight.data, saved)


@needs_torch
def test_refusal_directions_reject_degenerate_layer():
    from abliteration import directions

    means = torch.ones(3, 5, dtype=torch.float64)
    with pytest.raises(ValueError):
        directions.refusal_directions(means, means)
