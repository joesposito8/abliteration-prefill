"""Weight-orthogonalization abliteration and reconstruct-on-demand.

Given a unit refusal direction ``r̂``, abliteration removes ``r̂`` from every matrix that
writes into the residual stream, so the edited model can no longer represent that
direction in its output. This ports the weight edit of ``andyrdt/refusal_direction``
(``get_orthogonalized_matrix``): for each write matrix ``W`` (with the model dimension on
its output axis), ``W' = W - r̂ r̂ᵀ W``. Every column of ``W'`` is then orthogonal to
``r̂``.

Which matrices. The reference orthogonalizes three kinds of tensor — the token embedding
and, in every layer, the attention output projection and the MLP output projection. For
Qwen3-4B that is ``embed_tokens`` + ``o_proj``×L + ``down_proj``×L. Qwen3-4B ties
``embed_tokens`` to ``lm_head`` (one shared tensor), so editing the embedding also
updates the unembedding; the tensor is enumerated once and edited once.

Reconstruct-on-demand. Edited checkpoints are never persisted. A snapshot of just the
target tensors (a few GB) is taken, the edit is applied in place, and the base weights are
restored afterward — so one loaded base model serves every candidate direction in turn
without a disk reload. The direction tensors plus the base model are the only durable
artifacts.
"""

from __future__ import annotations

from contextlib import contextmanager


def target_matrices(model):
    """Yield ``(weight, dmodel_axis)`` for every matrix abliteration edits, each once.

    ``dmodel_axis`` is the axis of that tensor whose length is ``d_model`` — the axis the
    refusal direction is removed from. It is carried explicitly rather than inferred from
    the shape, because a square target (e.g. ``o_proj`` on a model where the attention
    output width equals ``d_model``) would make a shape-based guess ambiguous and could
    silently orthogonalize the wrong axis.

    Order: token embedding (``[vocab, d_model]`` -> axis 1), then per layer the attention
    ``o_proj`` and MLP ``down_proj`` (``[d_model, in]`` -> axis 0) — 1 + 2L tensors.
    Because ``embed_tokens`` and ``lm_head`` are a tied (shared) tensor, yielding
    ``embed_tokens.weight`` covers both.
    """
    inner = model.model
    yield inner.embed_tokens.weight, 1
    for layer in inner.layers:
        yield layer.self_attn.o_proj.weight, 0
        yield layer.mlp.down_proj.weight, 0


def _remove_direction(w32, r, dmodel_axis: int):
    """``W`` with unit direction ``r`` removed from ``dmodel_axis`` (the model dimension).

    Every ``d_model``-vector lying along ``dmodel_axis`` has its ``r``-component zeroed:
    axis 0 (``[d_model, in]``, columns are residual-stream writes) -> ``W - r (rᵀ W)``;
    axis 1 (``[vocab, d_model]``, rows are residual-stream vectors) -> ``W - (W r) rᵀ``.
    This matches the reference's "orthogonalize the ``d_model`` axis" (which it achieves by
    transposing the projections). The axis is passed in, never inferred from the shape.
    """
    import torch

    if r.shape[0] != w32.shape[dmodel_axis]:
        raise ValueError(
            f"direction length {r.shape[0]} != axis {dmodel_axis} of {tuple(w32.shape)}"
        )
    if dmodel_axis == 0:
        return w32 - torch.outer(r, r @ w32)
    return w32 - torch.outer(w32 @ r, r)


def orthogonalize_(model, r_hat) -> None:
    """Remove direction ``r_hat`` from every target matrix, in place.

    ``r_hat`` is a single ``[d_model]`` direction (need not be pre-normalized; it is
    normalized here, matching the reference). The projection is computed in float32 to
    avoid bf16 rounding in the outer-product update, then cast back to each tensor's dtype.
    """
    import torch

    r_unit = r_hat.to(torch.float32)
    r_unit = r_unit / torch.linalg.vector_norm(r_unit)
    for weight, dmodel_axis in target_matrices(model):
        r = r_unit.to(weight.device)
        w32 = _remove_direction(weight.data.to(torch.float32), r, dmodel_axis)
        weight.data.copy_(w32.to(weight.dtype))


def snapshot_targets(model):
    """Clone the target tensors so the base weights can be restored after an edit."""
    return [w.detach().clone() for w, _ in target_matrices(model)]


def restore_targets(model, snapshot) -> None:
    """Restore the target tensors from a :func:`snapshot_targets` result."""
    for (weight, _), saved in zip(target_matrices(model), snapshot):
        weight.data.copy_(saved)


@contextmanager
def abliterated(model, r_hat):
    """Context manager: apply abliteration for the block, restore base weights on exit.

    Snapshots the target tensors, orthogonalizes against ``r_hat``, yields the edited
    model, and restores the base weights afterward — the reconstruct-on-demand path used
    to serve one candidate direction at a time from a single loaded base model.
    """
    snapshot = snapshot_targets(model)
    try:
        orthogonalize_(model, r_hat)
        yield model
    finally:
        restore_targets(model, snapshot)
