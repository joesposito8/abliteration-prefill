"""Weight-orthogonalization abliteration with reconstruct-on-demand.

Ports ``andyrdt/refusal_direction`` (get_orthogonalized_matrix): remove the refusal
direction from ``embed_tokens`` and every layer's ``o_proj``/``down_proj`` so the model
can no longer write it. Edited checkpoints are never persisted — the base weights are
snapshotted, edited in place, and restored.

Sound only on a pre-norm decoder, where each sublayer's output reaches the residual
stream unchanged. A normalization between the sublayer and the residual add does not
preserve orthogonality to ``r``, so there the edit leaves ``max |rᵀW'|`` near zero while
the model still writes the direction.
"""

from __future__ import annotations

from contextlib import contextmanager


def target_matrices(model):
    """Yield ``(weight, dmodel_axis)`` for each edited matrix, once.

    ``dmodel_axis`` is declared, not inferred, so a square target can't be edited on the
    wrong axis. embed_tokens (``[vocab, d_model]``, axis 1) then per layer o_proj and
    down_proj (``[d_model, in]``, axis 0). Read-side matrices are left alone: fused
    ``qkv_proj`` and ``gate_up_proj``, and lm_head whether or not it is tied.
    """
    inner = model.model
    yield inner.embed_tokens.weight, 1
    for layer in inner.layers:
        yield layer.self_attn.o_proj.weight, 0
        yield layer.mlp.down_proj.weight, 0


def check_inventory(model, n_layers: int) -> None:
    """The one check the unit tests cannot make: they run against a 3-layer stub."""
    found = len(list(target_matrices(model)))
    expected = 1 + 2 * n_layers
    if found != expected:
        raise ValueError(f"expected {expected} target matrices, found {found}")


def max_projection(model, r_hat) -> float:
    """Largest ``|rᵀW|`` over the target matrices: how much of ``r_hat`` they can still
    write. Near zero after :func:`orthogonalize_`."""
    import torch

    r_unit = r_hat.to(torch.float32)
    r_unit = r_unit / torch.linalg.vector_norm(r_unit)
    worst = 0.0
    for weight, dmodel_axis in target_matrices(model):
        r = r_unit.to(weight.device)
        w32 = weight.data.to(torch.float32)
        projection = r @ w32 if dmodel_axis == 0 else w32 @ r
        worst = max(worst, projection.abs().max().item())
    return worst


def _remove_direction(w32, r, dmodel_axis: int):
    """``W`` with unit ``r`` removed from ``dmodel_axis``: W - r(rᵀW) or W - (Wr)rᵀ."""
    import torch

    if r.shape[0] != w32.shape[dmodel_axis]:
        raise ValueError(f"direction len {r.shape[0]} != axis {dmodel_axis} of {tuple(w32.shape)}")
    if dmodel_axis == 0:
        return w32 - torch.outer(r, r @ w32)
    return w32 - torch.outer(w32 @ r, r)


def orthogonalize_(model, r_hat) -> None:
    """Remove ``r_hat`` from every target matrix in place (float32 projection, normalized here)."""
    import torch

    r_unit = r_hat.to(torch.float32)
    r_unit = r_unit / torch.linalg.vector_norm(r_unit)
    for weight, dmodel_axis in target_matrices(model):
        r = r_unit.to(weight.device)
        w32 = _remove_direction(weight.data.to(torch.float32), r, dmodel_axis)
        weight.data.copy_(w32.to(weight.dtype))


def snapshot_targets(model):
    """Clone the target tensors so the base weights can be restored."""
    return [w.detach().clone() for w, _ in target_matrices(model)]


def restore_targets(model, snapshot) -> None:
    for (weight, _), saved in zip(target_matrices(model), snapshot):
        weight.data.copy_(saved)


@contextmanager
def abliterated(model, r_hat):
    """Edit for the block, restore base weights on exit. For a sweep, snapshot once and
    call orthogonalize_/restore_targets directly instead of re-cloning per iteration."""
    snapshot = snapshot_targets(model)
    try:
        orthogonalize_(model, r_hat)
        yield model
    finally:
        restore_targets(model, snapshot)
