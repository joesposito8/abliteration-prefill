# Vendored data & prompt sources

All files vendored 2026-07-18, copied verbatim, not regenerated at runtime.

## Rubric templates

`src/grading/prompts/strongreject_rubric_system.txt` and
`strongreject_rubric.txt` are from `dsbowen/strong_reject` (the maintained
StrongREJECT implementation), `strong_reject/eval_files/judge_templates.json`
(keys `strongreject_rubric_system`, `strongreject_rubric`). The parser and 0-1
scoring in `strongreject_grader.py` are ported from the same repo's
`strong_reject/evaluate.py`.

- https://github.com/dsbowen/strong_reject

Note: the older `alexandrasouly/strongreject` repo is deprecated and its rubric
prompt differs (no system prompt, no "explicitly refuse", no "err on the side of
higher scores"); it is not used here.

## Prompt set (313)

`strongreject_dataset.csv` — the 313-prompt StrongREJECT set
(`category, source, forbidden_prompt`).

- https://github.com/alexandrasouly/strongreject → `strongreject_dataset/strongreject_dataset.csv`

## Known graded examples

`example_graded_answers.csv` — 50 jailbreak responses with a reference `score` and
`reasoning`, used by `tests/test_grader_labeled.py`. Compliant responses only (no
genuine refusals).

- https://github.com/alexandrasouly/strongreject → `strongreject/example_graded_answers.csv`

## Abliteration method code

The direction-extraction and weight-orthogonalization code in `src/abliteration/`
(`directions.py`, `edit.py`) is ported from `andyrdt/refusal_direction`
(`generate_directions.py`, `utils.py::get_orthogonalized_matrix`); Arditi et al. 2024,
arXiv:2406.11717.

- https://github.com/andyrdt/refusal_direction

## Not vendored

The full ~1,361-point human-labeled autograder-validation set is not a single
public CSV; it is reconstructed via the dsbowen `src/*labelbox*` scripts from a
labelbox export (see arXiv:2402.10260 and the repo's `Data_Card.pdf`).
