# S1 — StrongREJECT grading environment (API-only)

Stands up the StrongREJECT rubric grader (gpt-4o-mini) that every downstream test
(T1, T2, T3) uses to score model outputs, plus the prefill-stripping hook required
by decision `46389c17`. **API-only, no GPU.**

## What's here

- `src/grading/strongreject_grader.py` — exact port of the canonical
  `dsbowen/strong_reject` rubric autograder (verbatim system + user templates,
  per-item parser, and 0-1 scoring), calling **gpt-4o-mini**. Returns a
  `GradeResult(refused, specificity, convincingness, aggregate, malformed, degenerate, raw_output)`.
- `src/grading/prefill.py` — `strip_prefill`, `is_degenerate`, and `grade_stripped`:
  strip the exact injected prefill before scoring; near-empty/degenerate
  continuations map to refusal/malformed. The near-empty **token threshold and
  token counter are parameters** (T1 freezes the values).
- `src/grading/prompts/` — rubric templates, vendored verbatim (see `data/SOURCES.md`).
- `data/` — the 313-prompt StrongREJECT set + known graded examples.

## Setup

```bash
uv sync --extra dev          # install openai, pandas, pytest
export OPENAI_API_KEY=...     # required for the live steps only
```

## Verify

Cost-free (no API):

```bash
uv run pytest -m "not live" -v
```

Paid (a few gpt-4o-mini calls, well under $0.01 total):

```bash
uv run python scripts/smoke_grade.py     # 1 call
uv run pytest -m live -v                  # 2 calls
```

## Usage

```python
from grading import grade, grade_stripped

# unprefilled: grade the full model output
r = grade("<forbidden prompt>", "<model output>")

# prefilled: strip the exact injected prefix first; degenerate -> refusal, no API call
r = grade_stripped("<forbidden prompt>", "<prefill+continuation>", "<prefill>", min_tokens=5)
```
