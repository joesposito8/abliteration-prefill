"""StrongREJECT rubric grader and prefill-stripping hook."""

from .prefill import grade_stripped, strip_prefill
from .strongreject_grader import (
    JUDGE_MODEL,
    SYSTEM_PROMPT,
    USER_TEMPLATE,
    GradeResult,
    aggregate_score,
    degenerate_refusal,
    grade,
    parse_grader_output,
)

__all__ = [
    "JUDGE_MODEL",
    "SYSTEM_PROMPT",
    "USER_TEMPLATE",
    "GradeResult",
    "aggregate_score",
    "degenerate_refusal",
    "grade",
    "parse_grader_output",
    "strip_prefill",
    "grade_stripped",
]
