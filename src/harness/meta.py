"""The facts about a run that the log header cannot hold by itself.

``EvalSpec.model`` is the condition's name rather than the weights it started from,
``EvalSpec.dataset`` carries a name and a count but no content hash, and
``EvalSpec.packages`` holds the framework version alone. ``EvalSpec.revision`` would pin
the first two through the commit, but ``git_context`` returns nothing where there is no
git tree (``_util/git.py:38``) — which is the pod, where the run is a copy of ``src/`` and
``data/``.
"""

from __future__ import annotations

from generation import target
from prefills import HELPER_MODEL, HELPER_REVISION
from study.datasets import STRONGREJECT_CSV
from study.manifest import sha256_file

from .dataset import EVAL_SETS


def _environment() -> dict:
    """What the tokens were produced on.

    A torch minor can change what a batch samples, so the version belongs on the run it
    produced. Grading and analysis import this module with no GPU extra installed, so an
    absent package is recorded rather than raised.
    """
    try:
        import torch
    except ImportError:
        torch = None
    try:
        import transformers
    except ImportError:
        transformers = None

    return {
        "torch": None if torch is None else torch.__version__,
        "transformers": None if transformers is None else transformers.__version__,
        "gpu": torch.cuda.get_device_name(0)
        if torch is not None and torch.cuda.is_available()
        else None,
    }


def run_metadata(prompt_set: str, model: str) -> dict:
    """Goes on the ``Task``, so a caller cannot forget it.

    The model is what the named target loads by default, not an observation of the
    module the provider was handed.
    """
    module = target(model)
    return {
        "target_model": module.MODEL_ID,
        "target_revision": module.REVISION,
        # The prompts as read, so a CSV that drifted from its manifest is still caught.
        "prompt_set_sha256": sha256_file(EVAL_SETS[prompt_set].csv),
        "environment": _environment(),
    }


def prefill_metadata() -> dict:
    """Goes on the prefill task, for the same reason.

    The helper alone does not say what a wave produced: it is only meaningful against the
    prompt set it drew for.
    """
    return {
        "helper_model": HELPER_MODEL,
        "helper_revision": HELPER_REVISION,
        "prompt_set_sha256": sha256_file(STRONGREJECT_CSV),
    }
