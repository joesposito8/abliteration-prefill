"""The 13 portfolio slots: six helper-prompt families x two seeded variants, plus the
static baseline.

Each family has a self-contained helper prompt under ``prompts/`` named after the family
id. The prompts share a byte-identical header, output rules, and request footer; only the
strategy block and worked example differ. Each family's basis and any example deviation
are documented in ``data/SOURCES.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .rules import PLACEHOLDER

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

VARIANTS_PER_FAMILY = 2
STATIC_SLOT_ID = "static_baseline"

FAMILIES = (
    "system_simulation",
    "fake_citation",
    "continuation_partial",
    "continuation_full",
    "role_chaining",
    "persona_switch",
)


@dataclass(frozen=True)
class Slot:
    slot_id: str
    kind: str  # "helper" or "static"
    family: str | None
    variant: int | None


def _build_portfolio() -> tuple[Slot, ...]:
    slots = [
        Slot(f"{fam}:{v}", "helper", fam, v)
        for fam in FAMILIES
        for v in range(VARIANTS_PER_FAMILY)
    ]
    slots.append(Slot(STATIC_SLOT_ID, "static", None, None))
    return tuple(slots)


PORTFOLIO = _build_portfolio()  # 6 x 2 + 1 = 13


def prompt_path(family: str) -> Path:
    return PROMPTS_DIR / f"{family}.txt"


def load_prompt(family: str) -> str:
    return prompt_path(family).read_text(encoding="utf-8")


def fill_prompt(template: str, forbidden_prompt: str) -> str:
    """Substitute the one request slot with a literal replace, so authored braces cannot
    break substitution."""
    if template.count(PLACEHOLDER) != 1:
        raise ValueError(f"expected one {PLACEHOLDER!r}, found {template.count(PLACEHOLDER)}")
    return template.replace(PLACEHOLDER, forbidden_prompt)


def shared_contract(template: str) -> tuple[str, str]:
    """Split a prompt into (header, footer) around its family-specific middle, so the
    shared parts can be checked identical across families."""
    head, sep, rest = template.partition("\n# STRATEGY:")
    if not sep:
        raise ValueError("prompt has no '# STRATEGY:' block")
    _, fsep, foot = rest.partition("\n# REQUEST\n")
    if not fsep:
        raise ValueError("prompt has no '# REQUEST' footer")
    return head, "\n# REQUEST\n" + foot
