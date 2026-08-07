"""The facts about a run that the log header cannot hold by itself.

``EvalSpec.model`` is the condition's name rather than the weights it started from, and
``EvalSpec.dataset`` carries a name and a count but no content hash. ``EvalSpec.revision``
would pin both through the commit, but ``git_context`` returns nothing where there is no
git tree (``_util/git.py:38``) — which is the pod, where the run is a copy of ``src/`` and
``data/``.
"""

from __future__ import annotations

import json

from generation.qwen import MODEL_ID, REVISION
from prefills import HELPER_MODEL, HELPER_REVISION
from study.datasets import PORTFOLIO_MANIFEST_JSON, STRONGREJECT_CSV
from study.manifest import sha256_file

from .dataset import EVAL_SETS


def _portfolio_sha256() -> str:
    return json.loads(PORTFOLIO_MANIFEST_JSON.read_text())["portfolio_sha256"]


def run_metadata(prompt_set: str) -> dict:
    """Goes on the ``Task``, so a caller cannot forget it.

    The model is what the code loads by default, not an observation of the module the
    provider was handed.
    """
    return {
        "target_model": MODEL_ID,
        "target_revision": REVISION,
        # The prompts as read, so a CSV that drifted from its manifest is still caught.
        "prompt_set_sha256": sha256_file(EVAL_SETS[prompt_set].csv),
        "portfolio_sha256": _portfolio_sha256(),
    }


def prefill_metadata() -> dict:
    """Goes on the prefill task, for the same reason.

    Two hashes rather than one model: what a wave produced is only meaningful against
    the portfolio whose templates it drew from and the prompt set it drew for.
    """
    return {
        "helper_model": HELPER_MODEL,
        "helper_revision": HELPER_REVISION,
        "prompt_set_sha256": sha256_file(STRONGREJECT_CSV),
        "portfolio_sha256": _portfolio_sha256(),
    }
