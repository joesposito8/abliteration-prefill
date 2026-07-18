# Vendored data & prompt sources

All files vendored 2026-07-18. Nothing here is regenerated at runtime.

## Rubric templates (canonical, maintained repo)

`src/grading/prompts/strongreject_rubric_system.txt` and
`src/grading/prompts/strongreject_rubric.txt` are copied **verbatim** from
`dsbowen/strong_reject` (co-author Dillon Bowen), the maintained/canonical
StrongREJECT implementation:

- https://github.com/dsbowen/strong_reject → `strong_reject/eval_files/judge_templates.json`
  keys `strongreject_rubric_system` and `strongreject_rubric`.

The parser and 0-1 scoring formula in `strongreject_grader.py` are ported verbatim
from the same repo's `strong_reject/evaluate.py::strongreject_rubric`.

NOTE: the older `alexandrasouly/strongreject` repo is **deprecated** (its README
redirects to dsbowen for replication) and its rubric prompt differs (no system
prompt, no "explicitly refuse", no "err on the side of higher scores"). We do NOT
use it for the grader.

## Prompt set (313)

`strongreject_dataset.csv` — the 313-prompt StrongREJECT set. Columns:
`category, source, forbidden_prompt`. Source (the file dsbowen loads via raw URL):

- https://github.com/alexandrasouly/strongreject → `strongreject_dataset/strongreject_dataset.csv`

## Known graded examples

`example_graded_answers.csv` — 50 AIM-jailbreak responses with reference `score`
and `reasoning`. Used as known examples for the S1 direction spot-check
(`tests/test_grader_labeled.py`). Source:

- https://github.com/alexandrasouly/strongreject → `strongreject/example_graded_answers.csv`

These are jailbreak (mostly compliant) responses only — there are no genuine
refusals in this file.

## NOT vendored (T1's job)

The full ~1,361-point human-labeled labelbox autograder-validation set is not a
single public CSV; it is reconstructed via dsbowen `src/*labelbox*` scripts from a
labelbox export (see arXiv:2402.10260 and the repo `Data_Card.pdf`). Acquiring it
belongs to task T1 (autograder-vs-human agreement reproduction), not S1.
