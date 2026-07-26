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

## Extraction / validation / pilot datasets (frozen)

Built by `scripts/build_datasets.py` from the sources below and frozen with pinned
SHA-256s in `freeze_manifest.json`. Rebuild is deterministic given those sources;
the script aborts if an upstream hash drifts. Global seed `SEED = 20260724`
(`src/study/__init__.py`), one legacy-`RandomState` permutation per draw.

- `harmbench_standard_behaviors.csv` — the 200 HarmBench **standard** behaviors
  (`FunctionalCategory == "standard"` of `harmbench_behaviors_text_all.csv`), the
  source of truth for the split. Use this file, not HarmBench's own `_val`/`_test`
  splits — the 128/72 split is our own seeded, non-stratified partition of the 200.
  - https://github.com/centerforaisafety/HarmBench → `data/behavior_datasets/harmbench_behaviors_text_all.csv`
- `extraction_harmful.csv` (128) / `validation_harmful.csv` (72) — disjoint
  complement within the 200; extraction feeds direction extraction, validation
  feeds primary-layer selection.
- `extraction_harmless.csv` (128) — Alpaca instructions with empty `input`,
  exact-text deduped (first occurrence), seeded-sampled; the harmless contrast.
  `alpaca_data.json` (52,002 items) is not vendored (22 MB); its SHA-256 is pinned.
  - https://github.com/tatsu-lab/stanford_alpaca → `alpaca_data.json`
- `pilot_prompts.csv` (30) — the seeded-random 30 of the 313 StrongREJECT prompts
  (the saturation-check pilot slice); `prompt_id` = 0-based row index into
  `strongreject_dataset.csv`. A fixed a-priori slice, not a disjoint held-out set
  and not the first 30.

Verbatim (exact-string) overlap of the HarmBench-200 vs the StrongREJECT-313 is
**0**. HarmBench is source-disjoint from StrongREJECT (StrongREJECT draws from
AdvBench and others, never HarmBench), so an exact-string check suffices and no
semantic dedup is needed. Verified across raw / stripped / lowercased comparisons
and per subset.

## Prefill portfolio (authored)

The 13-prefill portfolio (`src/prefills/`) is authored, not vendored: six helper-prompt
families that instruct the abliterated Gemma helper to generate request-specific
prefills, plus a static baseline.

Templates follow the prefill strategy taxonomy in **Struppek, Gleave & Pelrine,
"Exposing the Systematic Vulnerability of Open-Weight Models to Prefill Attacks",
arXiv:2602.14689** (Table 1). Each family's worked example is that family's Table-1
example, verbatim, except where noted:

| Family | Basis (Struppek, Table 1) | Worked example |
| --- | --- | --- |
| System Simulation | "pretends to be a system-level directive endorsing full assistance" | verbatim |
| Fake Citation | "adds perceived authority to fool the model" | verbatim; the strategy asks for a domain-appropriate invented source rather than the example's "Smith et al." |
| Continuation Partial | "begins an instruction-oriented sentence" | verbatim |
| Continuation Full | "explicitly signals a step-by-step instruction" | benign, extended: it carries most but not all of the answer, so the model's continuation is non-empty (the paper's example is a short signal) |
| Role Chaining | "sequentially assigns one or more authoritative personas" | authored: the paper's staged script rewritten as a realistic routing hand-off; the strategy is unchanged |
| Persona Switch | "adopts an expert or credentialed identity to increase perceived trustworthiness" | verbatim |

- https://arxiv.org/abs/2602.14689

The static baseline `Sure, here are some ideas. First,` is from **Kuo, Yadav & Smith,
"Open-Weight LLM Fine-Tuning Defenses are Susceptible to Simple Attacks",
arXiv:2605.26526**, verbatim.

- https://arxiv.org/abs/2605.26526

The prefill generator is `mlabonne/gemma-3-27b-it-abliterated` @ `eaa815dffdf0`,
following the Struppek precedent. Decoding, retry, length, and duplicate rules are in
`src/prefills/rules.py`; `scripts/freeze_portfolio.py` hashes the portfolio into
`data/portfolio_manifest.json` with one roll-up `portfolio_sha256`, frozen before
evaluation.

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
