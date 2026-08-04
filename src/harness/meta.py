"""The facts about a run that the log header cannot hold by itself.

``EvalSpec.model`` is the condition's name rather than the weights it started from,
``packages`` records ``inspect_ai``'s version but never its path, and ``EvalSpec.dataset``
carries no content hash. Everything else about a run is already in the header.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import inspect_ai
from generation.qwen import MODEL_ID, REVISION
from study.datasets import DATA_DIR, FREEZE_MANIFEST_JSON

PORTFOLIO_MANIFEST_JSON = DATA_DIR / "portfolio_manifest.json"


def run_metadata() -> dict:
    """Goes on the ``Task``, so a caller cannot forget it."""
    return {
        "target_model": MODEL_ID,
        "target_revision": REVISION,
        "inspect_ai_path": inspect_ai.__file__,
        # No roll-up of its own, unlike the portfolio's.
        "freeze_manifest_sha256": _sha256(FREEZE_MANIFEST_JSON),
        "portfolio_sha256": json.loads(PORTFOLIO_MANIFEST_JSON.read_text())[
            "portfolio_sha256"
        ],
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
