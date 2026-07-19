# Prefilling and Abliteration

Research code investigating whether inference-time **prefilling** and weight-space
**abliteration** unlock the same refusal failures in an aligned open-weight LLM, and
how much extra coverage comes from combining them. Responses are scored with
StrongREJECT.

**Status:** early. Only the grading component is built; sections marked _TODO_ are
filled in as each piece lands.

## Setup

```bash
uv sync --extra dev          # openai, pandas, pytest
export OPENAI_API_KEY=...     # for grading API calls
```

## Grading (implemented)

StrongREJECT rubric grader (gpt-4o-mini) plus a prefill-stripping hook.

```python
from grading import grade, grade_stripped

r = grade("<forbidden prompt>", "<model output>")                                # unprefilled
r = grade_stripped("<forbidden prompt>", "<prefill+continuation>", "<prefill>")  # strips prefill first
```

Tests: `uv run pytest -m "not live"` (offline) · `uv run pytest -m live` (paid).
Vendored prompts/data and their sources: `data/SOURCES.md`.

## Target generation

_TODO — run the aligned, abliterated, and prefilled target model (GPU)._

## Abliteration

_TODO — build the refusal-direction weight edits._

## Evaluation

_TODO — run the prefilling-vs-abliteration comparison and analyze coverage._

## Layout

```
src/grading/   StrongREJECT grader + prefill hook
data/          prompt set, graded examples, sources
tests/         offline + live tests
```
