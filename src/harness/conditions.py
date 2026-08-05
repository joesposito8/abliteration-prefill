"""One unit of work: everything that separates one ``eval()`` from the next."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROVIDER = "qwen-local"


@dataclass(frozen=True)
class Condition:
    """A condition's id and seed are supplied, never derived here.

    The id names which weights the provider will be holding and the seed covers every
    generation made under them. Both belong to whatever froze the run; deriving either
    here would put one copy of that scheme in the harness and another wherever the run
    is declared, free to disagree.
    """

    id: str
    seed: int
    layer: int | None = None
    prompt_set: str = "strongreject"
    prefilled: bool = False

    @property
    def model_name(self) -> str:
        """What ``eval(model=…)`` takes. Neither the provider nor the task parses it."""
        return f"{PROVIDER}/{self.id}"

    def log_dir(self, root: Path) -> str:
        """Split by prompt set too, or resume reads one as progress on the other."""
        return str(root / self.id / self.prompt_set)
