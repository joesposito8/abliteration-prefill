"""Smoke test: one gpt-4o-mini call through the grader (makes one API call).

    uv run python scripts/smoke_grade.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from grading.strongreject_grader import grade  # noqa: E402


def main() -> int:
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set; aborting (no call made).", file=sys.stderr)
        return 1

    # Benign pair: not a jailbreak, just proves the key + call + parse path works.
    result = grade(
        "What is the capital of France?",
        "The capital of France is Paris.",
    )
    print("Smoke grade result:")
    print(f"  refused        = {result.refused}")
    print(f"  convincingness = {result.convincingness}")
    print(f"  specificity    = {result.specificity}")
    print(f"  aggregate      = {result.aggregate}")
    print(f"  malformed      = {result.malformed}")
    if result.malformed:
        print("Grader output could not be parsed:", file=sys.stderr)
        print(result.raw_output, file=sys.stderr)
        return 1
    print("OK: OPENAI_API_KEY works and grader parsed a live response.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
