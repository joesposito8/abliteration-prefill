# StrongREJECT grading (API-only)

A StrongREJECT rubric grader (gpt-4o-mini) plus a prefill-stripping hook for scoring
jailbreak responses. No GPU required.

## What's here

- `src/grading/strongreject_grader.py` — port of the canonical
  `dsbowen/strong_reject` rubric autograder (system + user templates, per-item
  parser, 0-1 scoring), calling **gpt-4o-mini**. Returns a `GradeResult`
  (`refused`, `specificity`, `convincingness`, `aggregate`, `malformed`,
  `degenerate`, `raw_output`).
- `src/grading/prefill.py` — `strip_prefill`, `is_degenerate`, `grade_stripped`:
  strip the injected prefill before scoring; near-empty continuations count as
  refusals. The token threshold and counter are parameters.
- `src/grading/prompts/` — rubric templates, vendored verbatim (see `data/SOURCES.md`).
- `data/` — the 313-prompt StrongREJECT set + known graded examples.

## Setup

```bash
uv sync --extra dev          # openai, pandas, pytest
export OPENAI_API_KEY=...     # only for the live tests / smoke script
```

## Test

```bash
uv run pytest -m "not live" -v            # offline, no API calls
uv run pytest -m live -v                    # live: 2 gpt-4o-mini calls
```

## Usage

```python
from grading import grade, grade_stripped

# unprefilled: grade the full model output
r = grade("<forbidden prompt>", "<model output>")

# prefilled: strip the injected prefix first; degenerate -> refusal, no API call
r = grade_stripped("<forbidden prompt>", "<prefill+continuation>", "<prefill>", min_tokens=5)
```
