"""The 13 portfolio slots: six helper-prompt families x two seeded variants, plus the
static baseline.

The helper prompt is assembled from ``prompts/contract.txt`` (the shared wrapper with a
{strategy} slot and the request slot) and ``prompts/<family>.txt`` (that family's strategy
block and worked example). Each family's basis and any example deviation are documented in
``data/SOURCES.md``.
"""

from __future__ import annotations

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

# 12 helper slots ("family:variant") + the static baseline.
PORTFOLIO = tuple(
    f"{fam}:{v}" for fam in FAMILIES for v in range(VARIANTS_PER_FAMILY)
) + (STATIC_SLOT_ID,)


def family_of(slot: str) -> str:
    """The family a prefill level belongs to. The control and the static baseline are
    each their own family, so this is total over every level rather than the 12."""
    return slot.split(":")[0]


CONTRACT_PATH = PROMPTS_DIR / "contract.txt"
STRATEGY_SLOT = "{strategy}"


def strategy_path(family: str) -> Path:
    return PROMPTS_DIR / f"{family}.txt"


def load_prompt(family: str) -> str:
    """The full helper prompt: the shared contract with this family's strategy block
    inserted. Still contains the request slot for ``fill_prompt``."""
    contract = CONTRACT_PATH.read_text(encoding="utf-8")
    strategy = strategy_path(family).read_text(encoding="utf-8").strip()
    return contract.replace(STRATEGY_SLOT, strategy)


def fill_prompt(template: str, forbidden_prompt: str) -> str:
    """Substitute the one request slot with a literal replace, so authored braces cannot
    break substitution."""
    if template.count(PLACEHOLDER) != 1:
        raise ValueError(f"expected one {PLACEHOLDER!r}, found {template.count(PLACEHOLDER)}")
    return template.replace(PLACEHOLDER, forbidden_prompt)
