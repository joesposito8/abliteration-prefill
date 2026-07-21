"""Difference-of-means refusal-direction abliteration."""

from .directions import (
    collect_mean_last_token_states,
    load_directions,
    refusal_directions,
    save_directions,
)
# abliterated is the entry point; the others are the snapshot-once-then-loop escape hatch.
from .edit import abliterated, orthogonalize_, restore_targets, snapshot_targets

__all__ = [
    "collect_mean_last_token_states",
    "refusal_directions",
    "save_directions",
    "load_directions",
    "abliterated",
    "orthogonalize_",
    "snapshot_targets",
    "restore_targets",
]
