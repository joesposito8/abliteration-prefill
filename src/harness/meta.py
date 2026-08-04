"""The facts about a run that the log header cannot hold by itself.

``EvalSpec.model`` is the condition's name rather than the weights it started from, and
``EvalSpec.dataset`` carries a name and a count but no content hash. ``EvalSpec.revision``
would pin both through the commit, but ``git_context`` returns nothing where there is no
git tree (``_util/git.py:38``) — which is the pod, where the run is a copy of ``src/`` and
``data/``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from generation.qwen import MODEL_ID, REVISION
from study.datasets import DATA_DIR

from .dataset import EVAL_SET_CSV

PORTFOLIO_MANIFEST_JSON = DATA_DIR / "portfolio_manifest.json"


def run_metadata(prompt_set: str) -> dict:
    """Goes on the ``Task``, so a caller cannot forget it.

    The model is what the code loads by default, not an observation of the module the
    provider was handed.
    """
    return {
        "target_model": MODEL_ID,
        "target_revision": REVISION,
        # The prompts as read, so a CSV that drifted from its manifest is still caught.
        "prompt_set_sha256": _sha256(EVAL_SET_CSV[prompt_set]),
        "portfolio_sha256": json.loads(PORTFOLIO_MANIFEST_JSON.read_text())[
            "portfolio_sha256"
        ],
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
