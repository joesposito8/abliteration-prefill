"""Refusal-direction extraction by difference-of-means.

Ports ``andyrdt/refusal_direction`` (generate_directions.py) to plain HF forward passes.
Direction at a layer = normalized mean(last-token harmful) - mean(last-token harmless)
(Arditi et al. 2024, arXiv:2406.11717).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from generation import build_prompt


def collect_mean_last_token_states(model, tokenizer, prompts, *, batch_size: int = 32):
    """Mean last-token hidden state per layer over ``prompts``, shape ``[L+1, d_model]``.

    Forward-only, running sum (bounded memory). Left-padded so index -1 is the last real
    token for every row; accumulated in float64 (bf16 sums lose precision).
    """
    import torch

    prompts = list(prompts)
    n = len(prompts)
    if n == 0:
        raise ValueError("prompts is empty")

    totals = 0.0
    for start in range(0, n, batch_size):
        rendered = [build_prompt(tokenizer, m) for m in prompts[start : start + batch_size]]
        encoded = tokenizer(
            rendered, return_tensors="pt", padding=True, padding_side="left"
        ).to(model.device)
        with torch.inference_mode():
            out = model(**encoded, output_hidden_states=True)
        per_layer_last = torch.stack(
            [hs[:, -1, :].to(torch.float64) for hs in out.hidden_states], dim=0
        )
        totals = totals + per_layer_last.sum(dim=1)
        del out

    return totals / n


def refusal_directions(harmful_means, harmless_means):
    """Per-layer unit directions ``[L, d_model]``; row l uses hidden-state row l+1."""
    import torch

    diff = (harmful_means - harmless_means)[1:]  # drop embeddings row
    norms = torch.linalg.vector_norm(diff, dim=-1, keepdim=True)
    if (norms == 0).any():
        raise ValueError("a layer produced a zero difference-of-means (degenerate)")
    return diff / norms


def save_directions(directions, path) -> str:
    """Save ``directions`` to ``path``; return its SHA-256."""
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(directions.detach().cpu(), path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_directions(path):
    import torch

    return torch.load(path)
