# BR-2: Genius Weights Mixed Into the Model + Parallel Whole-Answer Drafts Refined by Genius Reasoning

**Date:** 2026-11-24 (UTC+8) | **Parent:** NCM-C1, BR-1 | **Discipline:** WSPL (prereg + amendment frozen pre-eval, no gradients, as-run times, SHA256-tracked)

## The two ideas under test (user's directive)

1. **Mixing genius weights with the model's normal weights** so genius and model "hold context together" and steer every token (M mechanism).
2. **Generate multiple whole ANSWERS together**, then genius **reasons over the drafts and refines them into one answer** (P mechanism) — parallelism at the answer level, not the token level.

## Frozen protocol

- Prereg `8f2bd191...` + amendment_1 `48779fa8...` (both frozen before any eval; amendment triggered by two disclosed diagnostics below).
- Heads: C1 H1..H4 (verbose-format-matched, H1 val-acc 96.4%); model sha re-verified post-recovery.
- Calibration: 12 GSM8K **train-pool** items (zero test data), α ∈ {0, 0.5, 1.0}, frozen tie rule.
- Eval: the same 30 frozen GSM8K items; strict = `#### N` == gold.

## Diagnostic 1 — the mixed mind is a mirror, not an expert (M FAILS)

- Calibration: α 0.0 / 0.5 / 1.0 all tie **4/12 with identical correct-sets** → frozen rule picks α*=0 → M arm skipped.
- Disclosed probe (`br2_divprobe.py`): greedy trajectories **identical to plain at α up to 8.0**.
- Why: the genius head was compiled *from the model's own behavior* and already agrees with the model's argmax ~96% of the time. Mixing a mirror into the original changes nothing. This also retroactively explains C1/BR-1's ~90-94% block acceptance.

## Diagnostic 2 — α cannot diversify drafts (triggers amendment_1)

- Drafts via α-steering would be identical → P switched (disclosed) to **1 greedy anchor + 3 temperature-0.8 sampled drafts** (seeds 1001/1002/1003). Genius reasoning/refinement rule unchanged: teacher-force each draft, H1 agreement on in-U positions, pick max, tie → shorter.

## Results (30 frozen items)

| Arm | strict acc | tok/s or wall | note |
|---|---|---|---|
| A2 (re-baseline) | **50.0%** (15/30) | 8.39 tok/s | **bit-identical regeneration of C1 arm A** (8,065 tokens exactly) — environment validated post-recycle |
| M (mixed mind) | — | — | **FAIL at calibration** per frozen rule (α*=0) |
| P (4 drafts + genius refine) | **40.0%** (12/30) | 3.49× wall of A2 | **FAIL** per frozen band (−10.0pp < −6.67pp) |
| Oracle best-of-4 (diagnostic) | **60.0%** (18/30) | — | the pool DOES contain better answers (+10pp headroom) |

## Where P lost (selection analysis)

- 18/30 pools contain a correct draft; anchor correct on 15 of them.
- **Kept** the correct anchor on 12; **lost 3** — the agreement score ranked a *wrong* draft above the correct anchor (e.g. item 265: wrong draft 0.9898 > correct anchor 0.9677).
- **Recovered 0/3** salvageable items (anchor wrong, correct draft present).
- Root cause: **agreement saturates** — top-score ties on 12/30 items; short "safe" answers score ~1.0 agreement while correct ones take digit-risks that create disagreements. Raw agreement rewards caution, not correctness.

## Verdict (frozen bands)

**MIXED MIND FAILS AS A STEERER; THE WHOLE-ANSWER POOL HAS REAL HEADROOM (oracle 60% vs 50%) BUT THE GENIUS AGREEMENT SCORE CANNOT HARVEST IT — P FAILS (40.0%, −10pp, 3.49× wall).**
Per the user's own rule ("if accuracy drops, that's not a win"), P as implemented is not a win. The experiment localizes the wall precisely: **the pool has the answers, the selector is the bottleneck**. Fix direction: a *discriminating* score (genius-model disagreement density, confidence-profile matching, or H1..H4 full-profile scoring), not more drafts or stronger mixing.

## Honest notes

- Bug disclosure: the first P run scored 0.0 everywhere (argmax *index* compared to *token id*); caught by a position-level debug (true agreement 5/5 in-window), fixed, poisoned state wiped, re-run from scratch. No number in this report comes from the buggy run.
- All 4 drafts unique on 30/30 items — temperature sampling delivered genuine diversity where α could not.
- Winner distribution: anchor 15, sampled 15 — the selector genuinely exercised its choice, it just chose wrong 3 times and never salvaged.
- The 3.49× wall is the honest price of 4 answers + a scoring pass; no speed claim is made for P.

## Artifacts (SHA256 in results JSON)

wspl_br2_prereg.json, wspl_br2_calibration.json, wspl_br2_results.json, br2_common.py, br2_calibrate.py, br2_eval.py, br2_assemble.py, br2_divprobe.py, br2_debug_score.py; states in `realbench_cache/br2_{cal,eval}/`.