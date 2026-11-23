# KGAT Ladder — WSPL progress record (2026-11-23)

## What WSPL is

WSPL (Weight-Space Programming & Learning) is the continuation of the NSC-1 line with a stricter
thesis: **capabilities are compiled directly into weights** — no gradient training, no backprop,
no neural-network training loop anywhere. Every update is closed-form linear algebra
(ridge, IRLS Newton, weighted counts/quantiles, rank-1 Gram downdates). Rules of the program:

- every rung is **preregistered**: success criteria frozen before any test-side evaluation;
- only real measured numbers are reported (honest negatives allowed);
- every model/reports include **build times** (all rungs fit inside 460s foreground slices);
- **bit-level reproducibility** is gated (v4/v5 live-regeneration checks); leave-one-out (LOO)
  discipline prevents train-side self-leakage.

## The ladder

| rung | what was compiled | GSM8K-full strict (n=1319) | oracle | verdict |
|---|---|---|---|---|
| v0 | data→weights: solved-problem embedding memory (ridge + prototype + kNN consensus) | 0.0% (best readout kNN-median 1.06%) | — | memory baseline; ARC-C 52.45%, MMLU-1k 44.9%, MMLU-full 45.08% |
| v1 | genius thinking → weights: mined <<expr=res>> chains → skeleton classifier + word→constant table + order prior | 2.35% strict / 4.02% loose | — | PASS (>2× recall baseline 1.06%) |
| v2 | Genius-RAM: 4 weight artifacts (classifier, word→constant v2, op-transition matrix, per-skeleton bands) + memory pull | 1.97% strict / 3.34% loose | — | honest negative (grammar gate FAILED 0.3868 < 0.43; mem-only 2.12% > fused 1.97% > genius-only 1.67%) |
| v3 | NCM-PBCS: one-pass coarse-to-fine parallel generation + selection | 1.44% strict / 2.58% loose | **39.95%** | **SELECTION WALL FOUND** (oracle 27.7× above realized) |
| v4 | fine-tuning → weights: 13-feature IRLS logistic head fitted on LOO train candidates | **3.18%** (2.21× v3) | 39.95%, realized 8.0% | PARTIAL; HLE full-head transfer failed (1.05% < f1-only 2.10%) |
| v5 | genius weight artifacts fine-tuned themselves (FT-A/B/C closed-form, leak-free LOO delivery signal) | **3.26%** (best arm S1) | 39.50% (Δ −0.45pp = unchanged), realized 8.25% | PARTIAL; selection upgrade NOT CONFIRMED |

HLE transfer: T2-full (n=572) v3 1.75% / v4 f1-only **2.10%** / v5 S2 2.10% (not improved);
HLE-v1 T2 (n=56): v5 S1 **1.79%** — the first nonzero strict result in the program (v0–v4 all 0.0%).

## What test are we running

The preregistered suite, frozen in each rung's prereg before any test eval:

| suite | n | metric |
|---|---|---|
| GSM8K full test, numeric | 1319 | strict = the single selected answer equals gold (abstain counts wrong); loose = best-ranked valid draft (diagnostic only) |
| GSM8K calc suite | 3825 | strict (v0 only — excluded from making: expressions embedded in text would be an oracle shortcut) |
| HLE T2-full | 572 of 2370 | same strict rule (gold parses as int) |
| HLE v1 T2 | 56 | same strict rule |
| ARC-Challenge full | 1144 | MC consensus accuracy (v0 recipe) |
| MMLU | 1000 (v1 slice) / 14042 (full pool) | MC consensus accuracy |

Integrity gates (v4/v5): GATE-1 signal pass reproduces v4 LOO records bit-exactly (25/25 items);
GATE-2 recomputed v3-control equals v3's reported strict/loose exactly
(0.014404852160727824 / 0.02577710386656558). Train-side artifacts are LOO-adjusted so no item's
own solution can leak into its candidate generation or head fitting.

## What the oracle score means

`oracle_full_inline` (v3/v4/v5 preregistered diagnostic) = the fraction of problems where **at least
one valid candidate chain in the one-pass generated pool contains the gold answer**.

- It is the **ceiling of any selector** working on that pool: a perfect selector could score at most
  the oracle. It measures GENERATION coverage, independent of selection.
- `realized_ratio = acc_when_gold_present` = how much of that ceiling the compiled selector actually
  converts. The oracle-vs-selected gap is the **selection wall**.
- Numbers: v3 1.44% strict vs 39.95% oracle (realized 3.6% of ceiling); v4 head 3.18% vs 39.95%
  (realized 8.0%); v5 3.26% vs 39.50% (realized 8.25%). Top-5 / top-10 rank quality: v4 10.9% / 18.0%.
- The oracle was preregistered in v3 precisely to decompose failure: it showed generation was NOT
  the bottleneck (40% of problems have the right answer somewhere in the pool) — selection is.
  That single diagnostic redirected the ladder (v4 head, v5 genius-FT).
- HLE oracle is 17.13% (T2-full) / 12.5% (v1-T2): a much harder pool — fewer problems are even
  reachable by the compiled grammar, consistent with the story-specific-literal ceiling measured in v1.

## v5 verdict and where the wall actually is

Frozen verdicts: primary PARTIAL (best arm S1 3.26% > v4 3.18% but below the 4.0% PASS bar);
selection upgrade NOT CONFIRMED (realized 8.25% < 10% target); generation unchanged; HLE transfer
not improved. Diagnostics that matter for the next rung:

- FT-C word→constant table changed **0/60** entries — that arm was inert.
- A-only ablation (skeleton-reliability fine-tuning alone) was the strongest measured configuration:
  **3.71% strict, oracle 40.18%, top-5 12.96%** — it motivates a preregistered v6 with skeleton
  reliability as the main arm.
- The v5 head's `frac_inband` (0.048) and `mag_soft` (0.028) dominate the fitted weights — the
  selector leans on memory-band priors, not on chain provenance. Richer, more discriminative
  selection features are the attack surface, not more generation.

## Repo map for this record

```
results/kgat_v0_prereg.json … kgat_v5_prereg.json   frozen protocols (one per rung)
results/kgat_v0_results.json, kgat_v01_rule_study   v0 results + fallback-rule study
results/kgat_v{1..5}_results_digest.json            per-rung results (per-item dumps omitted)
results/kgat_v4_head.json, kgat_v5_head{,_A,_B,_C}.json  compiled selector weights (IRLS, no backprop)
results/kgat_v5_times.json                          per-stage build times
results/kgat_mmlu_full_results.json                 MMLU full-pool (n=14042) v0-recipe result
```

## Local-only artifacts (sandbox), SHA256-pinned

The full per-item state dumps, the frozen index prereg and the rung scripts are preserved in the
local sandbox (`/home/z/my-project/...`); fingerprints so they can be verified or re-pushed later:

| artifact | sha256 (first 16) |
|---|---|
| scripts/realbench/push_data_min/prereg.json (frozen indices, 116,244 B) | 8d9af280356d821b |
| scripts/realbench/push_data_min/prereg_v2_amendment.json | 9205233882711b39 |
| scripts/realbench/heads.py | f2bb55cf4675c006 |
| scripts/kgat_v0.py | e4322d9f9583d9a9 |
| scripts/kgat_v1_making.py | 1ded7e78ea76f7f8 |
| scripts/kgat_v2_genius_ram.py | 378d312045060c23 |
| scripts/kgat_v3_ncm_pbcs.py | 22f0e69ac38364a7 |
| scripts/kgat_v4_ft_weights.py | 61812a7f504dfbfe |
| scripts/kgat_v5_genius_ft.py | d0e45c40ee7b772f |
| scripts/v4_wrapup_note.py | 81114f9809c29a3d |
| scripts/v5_wrapup_note.py | e623ddd3aac1c629 |

State files with full per-item candidate pools: `download/wspl_kgat_v{2..5}_state_*.json`
(train/test/hle/ablation slices, 0.5–6 MB each).
