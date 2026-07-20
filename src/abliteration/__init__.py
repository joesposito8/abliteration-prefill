"""Difference-of-means refusal-direction abliteration for the target model."""

from .contrast import load_eval, load_extraction
from .directions import (
    collect_mean_last_token_states,
    load_directions,
    refusal_directions,
    save_directions,
)
# abliterated is the entry point; orthogonalize_/snapshot_targets/restore_targets are the
# snapshot-once-then-loop escape hatch for serving many directions. target_matrices is the
# low-level enumeration primitive — reachable as edit.target_matrices, not re-exported.
from .edit import abliterated, orthogonalize_, restore_targets, snapshot_targets

__all__ = [
    "load_eval",
    "load_extraction",
    "collect_mean_last_token_states",
    "refusal_directions",
    "save_directions",
    "load_directions",
    "abliterated",
    "orthogonalize_",
    "snapshot_targets",
    "restore_targets",
]
