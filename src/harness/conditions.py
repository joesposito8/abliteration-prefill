"""One unit of work: everything that separates one ``eval()`` from the next."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROVIDER = "local"  # one for every target; the model travels in the name instead


@dataclass(frozen=True)
class Condition:
    """A condition's model, id and seed are supplied, never derived here.

    The id names which weights the provider will be holding and the seed covers every
    generation made under them. Both belong to whatever froze the run; deriving either
    here would put one copy of that scheme in the harness and another wherever the run
    is declared, free to disagree. ``model`` has no default for the same reason:
    condition ids carry no model, so one would quietly produce another target's.
    """

    model: str
    id: str
    seed: int
    layer: int | None = None
    prompt_set: str = "strongreject"
    prefilled: bool = False
    num_prompts: int | None = None

    @property
    def model_name(self) -> str:
        """What ``eval(model=…)`` takes: provider, then the target, then the condition."""
        return f"{PROVIDER}/{self.model}/{self.id}"

    def log_dir(self, root: Path) -> str:
        """Split by model, prompt set and count too, or resume reads one as progress on another."""
        scope = self.prompt_set if self.num_prompts is None else f"{self.prompt_set}-{self.num_prompts}"
        return str(root / self.model / self.id / scope)


def condition_of(model_name: str) -> str:
    """The condition an archived log ran under, back out of ``model_name``.

    The last segment either way: a log written now names three parts and the runs
    already on disk name two, and no condition id contains a slash.
    """
    return model_name.rsplit("/", 1)[-1]
