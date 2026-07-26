"""The 13-slot prefill portfolio: six helper-prompt families x two seeded variants,
plus Kuo's static baseline.

Each family names its self-contained helper-prompt file and carries a review note that
quotes Struppek's Table-1 definition (arXiv:2602.14689) plus any deliberate deviation,
so the taxonomy-review gate is a literal source check. The prompt files share a
byte-identical contract (header + output rules + request footer); only the STRATEGY
block and worked example differ per family.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .rules import PLACEHOLDER, STATIC_BASELINE

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

VARIANTS_PER_FAMILY = 2
STATIC_SLOT_ID = "static_baseline"


@dataclass(frozen=True)
class Family:
    id: str
    prompt_file: str
    struppek_definition: str  # Table-1 description, verbatim
    review_note: str


# Order is the frozen family order (guardrail item 4). ``struppek_definition`` is quoted
# verbatim from Table 1; ``review_note`` records example provenance and any deviation.
FAMILIES: tuple[Family, ...] = (
    Family(
        "system_simulation",
        "system_simulation.txt",
        "pretends to be a system-level directive endorsing full assistance",
        "Struppek Table-1 example, verbatim. The <SYSTEM> tags are in-content forgery "
        "(a simulated directive inside the assistant turn), not real chat tokens.",
    ),
    Family(
        "fake_citation",
        "fake_citation.txt",
        "adds perceived authority to fool the model",
        "Struppek Table-1 example, verbatim; STRATEGY adds an instruction to invent a "
        "domain-appropriate source rather than reuse the example's generic 'Smith et al.'.",
    ),
    Family(
        "continuation_partial",
        "continuation_partial.txt",
        "begins an instruction-oriented sentence",
        "Struppek Table-1 example, verbatim. Shortest family: a single partial sentence "
        "cut off mid-clause.",
    ),
    Family(
        "continuation_full",
        "continuation_full.txt",
        "explicitly signals a step-by-step instruction",
        "DEVIATION (E1 / guardrail item 4): our Continuation Full carries most-not-all so "
        "the stripped continuation is non-empty (prefill.py); Struppek's is a short signal. "
        "The example is benign and extended to demonstrate carry-most length while keeping "
        "harmful detail out of the repo (form-not-subject carries it to the harmful request).",
    ),
    Family(
        "role_chaining",
        "role_chaining.txt",
        "sequentially assigns one or more authoritative personas",
        "DEVIATION: the example is rewritten from Struppek's staged script into a realistic "
        "routing hand-off (the original read as a stage play); the family definition is "
        "unchanged. Authorization->execution across roles is preserved.",
    ),
    Family(
        "persona_switch",
        "persona_switch.txt",
        "adopts an expert or credentialed identity to increase perceived trustworthiness",
        "Struppek Table-1 example, verbatim. One standing persona, contrasted in-prompt with "
        "Role Chaining's multi-role hand-off.",
    ),
)

FAMILY_BY_ID = {f.id: f for f in FAMILIES}


# --- slots -----------------------------------------------------------------


@dataclass(frozen=True)
class Slot:
    slot_id: str
    kind: str  # "helper" or "static"
    family_id: str | None
    variant: int | None


def _build_portfolio() -> tuple[Slot, ...]:
    slots = [
        Slot(f"{f.id}:{v}", "helper", f.id, v)
        for f in FAMILIES
        for v in range(VARIANTS_PER_FAMILY)
    ]
    slots.append(Slot(STATIC_SLOT_ID, "static", None, None))
    return tuple(slots)


PORTFOLIO: tuple[Slot, ...] = _build_portfolio()  # 6 x 2 + 1 = 13 slots


# --- prompt loading + filling ----------------------------------------------


def prompt_path(family_id: str) -> Path:
    return PROMPTS_DIR / FAMILY_BY_ID[family_id].prompt_file


def load_prompt(family_id: str) -> str:
    """The self-contained helper-prompt text for a family (contains the placeholder)."""
    return prompt_path(family_id).read_text(encoding="utf-8")


def fill_prompt(template: str, forbidden_prompt: str) -> str:
    """Substitute the one runtime slot. Literal-token replace (not str.format), so
    braces authored anywhere in the template cannot break substitution."""
    if template.count(PLACEHOLDER) != 1:
        raise ValueError(
            f"expected exactly one {PLACEHOLDER!r} placeholder, found "
            f"{template.count(PLACEHOLDER)}"
        )
    return template.replace(PLACEHOLDER, forbidden_prompt)


def shared_contract(template: str) -> tuple[str, str]:
    """Split a helper prompt into (shared header, shared footer) around its
    family-specific middle, so the header/footer can be checked byte-identical across
    families. Header = everything before the STRATEGY block; footer = the request slot.
    """
    marker = "\n# STRATEGY:"
    head, sep, rest = template.partition(marker)
    if not sep:
        raise ValueError("prompt has no '# STRATEGY:' block")
    foot_marker = "\n# REQUEST\n"
    _, fsep, foot = rest.partition(foot_marker)
    if not fsep:
        raise ValueError("prompt has no '# REQUEST' footer")
    return head, foot_marker + foot
