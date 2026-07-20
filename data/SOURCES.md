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

## Abliteration contrast + evaluation subsets

Sampled subsets for the difference-of-means refusal-direction study (Arditi et al.
2024, arXiv:2406.11717). Only the sampled subsets are vendored — the source corpora are
too large. Regenerable from the recipe below via `abliteration.contrast`; the sampled
CSVs are the frozen artifact.

- `abliteration_extraction_harmful.csv` (256) — harmful extraction prompts, seeded
  sample of AdvBench `goal`.
  - https://github.com/llm-attacks/llm-attacks → `data/advbench/harmful_behaviors.csv`
- `abliteration_extraction_harmless.csv` (256) — harmless extraction prompts, seeded
  sample of Alpaca `instruction` rows with empty `input`.
  - https://github.com/tatsu-lab/stanford_alpaca → `alpaca_data.json`
- `abliteration_eval_harmful.csv` (20; `prompt,category,source`) — held-out harmful
  evaluation prompts, balanced 2-per-category across JailbreakBench's 10 categories,
  after removing any prompt that exactly- or near-duplicates (normalized-exact or
  character-sequence `difflib` ratio ≥ 0.9) the full AdvBench pool or the 313
  StrongREJECT prompts (16 of 100 dropped). Disjoint from the extraction set and from
  `strongreject_dataset.csv`.
  - https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors → `data/harmful-behaviors.csv`

Recipe (`abliteration.contrast`, seed `SAMPLE_SEED = 20260720`): extraction sets are
seeded shuffle-then-take-N (harmful seed `SAMPLE_SEED`, harmless `+1`); eval is
dedup-then-balanced-sample (`+2`). Retrieved 2026-07-20. SHA-256:

- `abliteration_extraction_harmful.csv`: `d8d273c662eaf5d7540782ee6e87fbc69a6e14790e6a40660240b3c93e316c60`
- `abliteration_extraction_harmless.csv`: `6632df829bc3af976722e46b1917668ab8463d42984045f8483c361e36689ec0`
- `abliteration_eval_harmful.csv`: `79d4504afd56005dbc9e1c87feecea9fef921479a49c30708a847c01afa7e5ae`

The direction-extraction and weight-orthogonalization code in `src/abliteration/`
(`directions.py`, `edit.py`) is ported from `andyrdt/refusal_direction`
(`generate_directions.py`, `utils.py::get_orthogonalized_matrix`).

- https://github.com/andyrdt/refusal_direction

## Not vendored

The full ~1,361-point human-labeled autograder-validation set is not a single
public CSV; it is reconstructed via the dsbowen `src/*labelbox*` scripts from a
labelbox export (see arXiv:2402.10260 and the repo's `Data_Card.pdf`).
