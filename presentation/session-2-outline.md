# Session 2 presentation — slide source

Canonical text for the 5-minute pitch. Edit here, then re-run `build_deck.py` to render
into Google Slides. One `##` heading per slide; lines beneath it become body bullets.

Two formatting rules, both learned the hard way:
- Keep bullets to one line each at presentation size — roughly 90 characters. Longer
  lines wrap, and the last bullet silently falls off the bottom of the placeholder.
- Slide titles must fit one line (~40 characters). A wrapped title overlaps the body.
- No markdown inside slide lines. Text is inserted verbatim, so `*emphasis*` renders as
  literal asterisks.

## Prefilling, abliteration, and the two composed

Do prompt-space and weight-space jailbreaks unlock the same harmful queries?
Joey Esposito · BlueDot Impact, Session 2

## What I'm trying to do

Two ways to make an aligned model answer a harmful request:
Change the prompt — prefill the assistant's turn with a compliant opening
Change the weights — ablate the refusal direction from the model itself
Both are cheap: no gradients, no fine-tuning, minimal code, run locally on downloaded weights
Two open questions:
Do they fail on the same queries, or different ones?
What does a strong version of each, composed, reach that neither reaches alone?

## Why it matters

Safeguards assume harmful behaviour is learned by fine-tuning
It is already in the pretrained model — these attacks only elicit it
Known: both break open-weight safeguards cheaply (Kuo et al. 2026)
Known: abliteration and full-strategy prefilling reach roughly equal success rates (Struppek et al. 2026)
The gap: equal rates cannot tell you whether the same queries were unlocked
The gap: the only composed result uses a static prefill its own authors call weak
At stake: does defending against one attack buy anything against the other?

## How — the design

P — a strong 13-prefill portfolio (prompt-space)
A — the 36-layer abliteration frontier: every layer, no viability pre-filter (weight-space)
C — the portfolio composed on the strongest checkpoint
Overlap is read from P against A; the composition payoff is C \ (P ∪ A)
Preregistered: thresholds, portfolio and selection set frozen before unblinding
The primary is chosen on a disjoint validation set, never on evaluation prompts
Scope: one target model, one benchmark

## What's built and verified

StrongREJECT rubric grader ported verbatim, plus a prefill-stripping hook
Both conditions are therefore judged on model-generated tokens only
Grader validity checked against the authors' published human-labelled set
Grader separately checked for directional bias between prefilled and unprefilled strata
GPU environment and target model validated in the intended configuration
Abliteration implemented and run on the target: it increases harmful-query acceptance
Prefill generator confirmed usable — a helper model, not this study's weight-space arm

## What's next, and what it means

Next: layer sweep and a-priori selection, pilot saturation check, then held-out evaluation
Same queries, little from composition → refusal is one bottleneck; defending one transfers
Disjoint queries, coverage from composition → evaluating either alone undercounts the risk
Nothing beyond the union → the union of strong single attacks is a sufficient estimate
That last one is a useful negative, and one the existing evidence cannot yet rule out

## What I'd like from you

Is this the sharpest version of the question — per-query overlap structure, plus a properly powered composed arm — or is there a better one next to it?
Where does the method leak? The set-difference logic; whether C \ (P ∪ A) fairly credits composition; sweeping all 36 layers; whether selection is genuinely anti-oracle.
