"""Refusal-direction extraction by difference-of-means.

Ports the direction-extraction math of ``andyrdt/refusal_direction``
(``pipeline/submodules/generate_directions.py``) to plain HuggingFace forward passes;
the reference caches activations through TransformerLens hooks, which this project does
not use. The refusal direction at a layer is the normalized difference between the mean
last-token hidden state over harmful prompts and over harmless prompts (Arditi et al.
2024, arXiv:2406.11717). One forward pass over the contrast set yields a direction for
every layer at once.

Two properties fail silently and so are enforced here:

Last-token position under left padding. The decision to comply or refuse is carried by
the residual state at the final prompt token, so only that position is collected. Batches
are left-padded (matching ``generation.generate_batch``) so index ``-1`` is the last real
token for every row; right padding would average padding positions into the mean.

Accumulation precision. Means are accumulated in float64. The reference notes this
explicitly ("to avoid numerical issues"): summing thousands of bf16 activations in bf16
loses low-order bits and biases the difference.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from generation import build_prompt


def collect_mean_last_token_states(model, tokenizer, prompts, *, batch_size: int = 32):
    """Mean last-token hidden state at every layer over ``prompts``.

    Returns a float64 tensor of shape ``[n_hidden_states, d_model]`` where row ``k`` is
    ``hidden_states[k]`` averaged over all prompts at the last real token. For a model
    with L layers there are L+1 hidden states (embeddings + one per layer), so the
    direction for layer ``l`` is row ``l + 1``.

    Forward-only: no generation, no gradients. Accumulates a running sum and divides once
    at the end rather than storing per-prompt states, so memory is a handful of
    ``[d_model]`` vectors regardless of ``len(prompts)``.
    """
    import torch

    prompts = list(prompts)
    n = len(prompts)
    if n == 0:
        raise ValueError("prompts is empty")

    totals = None
    for start in range(0, n, batch_size):
        batch = prompts[start : start + batch_size]
        rendered = [build_prompt(tokenizer, message) for message in batch]
        encoded = tokenizer(
            rendered, return_tensors="pt", padding=True, padding_side="left"
        ).to(model.device)
        with torch.inference_mode():
            out = model(**encoded, output_hidden_states=True)
        # Left padding puts the last real token at index -1 for every row.
        # hidden_states: tuple of [batch, seq, d_model], one per layer + embeddings.
        per_layer_last = torch.stack(
            [hs[:, -1, :].to(torch.float64) for hs in out.hidden_states], dim=0
        )  # [n_hidden_states, batch, d_model]
        batch_sum = per_layer_last.sum(dim=1)  # [n_hidden_states, d_model]
        totals = batch_sum if totals is None else totals + batch_sum
        del out

    return totals / n


def refusal_directions(harmful_means, harmless_means):
    """Per-layer normalized difference-of-means directions.

    Given the two ``[n_hidden_states, d_model]`` mean tensors, returns ``[n_layers,
    d_model]`` unit vectors, one per transformer layer: row ``l`` is
    ``normalize(harmful_means[l+1] - harmless_means[l+1])`` (the embeddings row 0 is
    dropped, so indexing matches layer number). Kept in float64.
    """
    import torch

    diff = (harmful_means - harmless_means)[1:]  # drop embeddings row -> per layer
    norms = torch.linalg.vector_norm(diff, dim=-1, keepdim=True)
    if (norms == 0).any():
        raise ValueError("a layer produced a zero difference-of-means (degenerate)")
    return diff / norms


def save_directions(directions, path) -> str:
    """Save directions to ``path`` and return the file's SHA-256 (for the manifest)."""
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(directions, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_directions(path):
    """Load a directions tensor saved by :func:`save_directions`."""
    import torch

    return torch.load(path)
