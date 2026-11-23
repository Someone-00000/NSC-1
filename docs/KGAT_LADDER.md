# KGAT Ladder — WSPL progress record (2026-11-24)

## What WSPL is

WSPL (Weight-Space Programming & Learning) is the continuation of the NSC-1 line with a stricter
thesis: **capabilities are compiled directly into weights** — no gradient training, no backprop,
no neural-network training loop anywhere. Every update is closed-form linear algebra
(ridge, IRLS Newton, weighted counts/quantiles, rank-1 Gram downdates). Rules of the program:

- every rung is **preregistered**: success criteria frozen before any test-side evaluation;
- only real measured numbers are reported (honest negatives allowed);
- every model/reports include **build times** (all rungs fit inside 460s foreground slices);
- **bit-level reproducibility** is gated (v4/v5 live-regeneration checks; v6 artifact replay);
- leave-one-out (LOO) discipline prevents train-side self-leakage.

## The ladder

| rung | what was compiled | GSM8K-full strict (n=1319) | oracle | verdict |
|---|---|---|---|---|
| v0 | data→weights: solved-problem embedding memory (ridge + prototype + kNN consensus) | 0.0% (best readout kNN-median 1.06%) | — | memory baseline; ARC-C 52.45%, MMLU-1k 44.9%, MMLU-full 45.08% |
| v1 | genius thinking → weights: mined <<expr=res>> chains → skeleton classifier + word→constant table + order prior | 2.35% strict / 4.02% loose | — | PASS (>2× recall baseline 1.06%) |
| v2 | Genius-RAM: 4 weight artifacts (classifier, word→constant v2, op-transition matrix, per-skeleton bands) + memory pull | 1.97% strict / 3.34% loose | — | honest negative (grammar gate FAILED 0.3868 < 0.43; mem-only 2.12% > fused 1.97% > genius-only 1.67%) |
| v3 | NCM-PBCS: one-pass coarse-to-fine parallel generation + selection | 1.44% strict / 2.58% loose | **39.95%** | **SELECTION WALL FOUND** (oracle 27.7× above realized) |
| v4 | fine-tuning → weights: 13-feature IRLS logistic head fitted on LOO train candidates | **3.18%** (2.21× v3) | 39.95%, realized 8.0% | PARTIAL; HLE full-head transfer failed (1.05% < f1-only 2.10%) |
| v5 | genius weight artifacts fine-tuned themselves (FT-A/B/C closed-form, leak-free LOO delivery signal) | **3.26%** (best arm S1) | 39.50% (Δ −0.45pp = unchanged), realized 8.25% | PARTIAL; selection upgrade NOT CONFIRMED |
| v6 | DVA: direct vector access given to the compiled selector (ridge vector readout, neighbor-exact profile, 921-d interaction) — **zero new data** | 3.49% (best arm A1 ≡ C0-refit 46/1319; v4 artifact anchor replayed bit-exact 42/1319) | 39.95% (unchanged pool) | **H1 REJECTED, H2 SUPPORTED**: direct vector access changed ZERO selections; compression is a sufficient statistic at this data scale |

HLE transfer: T2-full (n=572) v3 1.75% / v4 f1-only **2.10%** / v5 S2 2.10% / v6 arms 1.05–1.92% (not improved);
HLE-v1 T2 (n=56): v5 S1 **1.79%** and v6 C0/A1 1.79% (1/56) — the only nonzero strict results in the program (v0–v4 all 0.0%).

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
  (realized 8.0%); v5 3.26% vs 39.50% (realized 8.25%); v6 C0-refit 3.49% vs 39.95% (realized 8.7%).
  Top-5 / top-10 rank quality: v4 10.9% / 18.0%; v6 C0-refit 12.3% / 19.3%.
- The oracle was preregistered in v3 precisely to decompose failure: it showed generation was NOT
  the bottleneck (40% of problems have the right answer somewhere in the pool) — selection is.
- HLE oracle is 17.13% (T2-full) / 12.5% (v1-T2): a much harder pool.

## v6 — DVA (Direct Vector Access): the vector-access question, answered

User question (2026-11-24): *we used the model's vectors to build the data (memory library). Instead
of compiling MORE weights from more data, what happens when the genius weights we already made get
DIRECT access to the vectors?*

Design (prereg frozen before eval, `results/kgat_v6_dva_prereg.json`): run on the FROZEN v4 state
files (44,243 candidate rows / 373 positives — bit-identical to v4, nothing regenerated). Keep every
v4 component byte-identical; add closed-form channels that read the frozen 896-d vectors directly:

- **A1 ridge readout**: 896-weight linear map question-vector→answer-magnitude compiled from the
  1000-item memory in ONE closed-form solve (0.44s) — the fastest data→weight conversion measured
  in the program; exact LOO via the hat-matrix identity for train rows.
- **A2 neighbor profile**: element-wise exact-match profile against the 10 pulled memory golds
  (v4's f8 was the aggregated band check of the same pull).
- **A3 full DVA**: 921 features including per-dimension q_std ⊙ log10(v+1) interaction.
- **C0 control**: exact refit of the v4 13-feature head under the frozen protocol.

Reconstruction gates (sandbox recycle destroyed the rung scripts; all science state survived):
GATE-0 data identity 1000/1000; GATE-1 recomputed kNN-10 reproduces stored v3/v4 band features at
**100.000%** (61,449/61,449 rows); GATE-2 refit convention recovered (P1: mean-form Hessian, L2 not
divided by weight sum), v4 artifact weights replay **all four v4-reported numbers bit-exactly**
(42/1319 = 3.1842%, realized 7.9696%, top5 10.9174%, top10 18.044%); GATE-3 v3 control bit-exact.

Results (single frozen eval per arm):

| arm | features | GSM8K-full strict | realized | top5 | HLE T2-full |
|---|---|---|---|---|---|
| C0 refit control | 13 | **3.49%** (46/1319) | 8.73% | 12.28% | 1.57% |
| A1 ridge readout | 14 | 3.49% (46/1319) — **≡ C0, zero selections changed** | 8.73% | 12.21% | 1.57% |
| A2 neighbor profile | 24 | 3.26% (43/1319) | 8.16% | 10.99% | 1.05% |
| A3 full DVA | 921 | 2.81% (37/1319) | 7.02% | 10.39% | 1.92% |
| v4 artifact anchor | 13 | 3.18% (42/1319) | 7.97% | 10.92% | — |

Verdict (frozen criterion met as PARTIAL, with the honest attribution recorded in the results JSON):
the direct vector channel compiled to w=0.064 and flipped **zero** top-1 selections; wider raw access
monotonically degrades (46 → 46 → 43 → 37). **H1 (vector access adds signal) REJECTED; H2 (the
13-feature compression is a sufficient statistic of the memory pull at this data scale) SUPPORTED.**
With 373 positives and zero gradients, more access = more noise, not more capability.

Stability: every head recompiled twice — bit-identical (max weight diff 0.0) in all arms.
Build times: ridge readout 0.44s (896 weights), 13/14-feature heads 0.5–0.8s, 921-feature head 89.1s,
total run 100.5s — all inside a 460s slice.

**Implication for the big run**: the fastest data→weight conversion is already here (seconds, closed
form, bit-stable). The binding constraint is DATA VOLUME into weights, not feature access: 373
positives cannot support 921-dim direct readouts, and the compressed summary of the pull already
carries everything the vectors offer the selector. The big run's lever: compile MORE solved problems
into the memory and the head (data→weights at scale), then revisit direct vector access.

## Repo map for this record

```
results/kgat_v0_prereg.json … kgat_v6_dva_prereg.json  frozen protocols (one per rung)
results/kgat_v0_results.json, kgat_v01_rule_study       v0 results + fallback-rule study
results/kgat_v{1..6}_results(_digest).json              per-rung results (per-item dumps omitted)
results/kgat_v4_head.json, kgat_v5_head{,_A,_B,_C}.json compiled selector weights (IRLS, no backprop)
results/kgat_v6_dva_head_{C0,A1,A2}.json                v6 compiled heads (A3 pinned locally, see below)
results/kgat_v5_times.json, kgat_v6_dva_times.json      per-stage build times
results/kgat_mmlu_full_results.json                     MMLU full-pool (n=14042) v0-recipe result
```

## Local-only artifacts (sandbox), SHA256-pinned

The full per-item state dumps, the frozen index prereg and the rung scripts are preserved in the
local sandbox (`/home/z/my-project/...`); fingerprints so they can be verified or re-pushed later:

| artifact | sha256 (first 16) |
|---|---|
| scripts/realbench/push_data_min/prereg.json (frozen indices, 116,244 B) | 8d9af280356d821b |
| scripts/realbench/push_data_min/prereg_v2_amendment.json | 9205233882711b39 |
| scripts/realbench/heads.py | f2bb55cf4675c006 |
| scripts/kgat_v0.py … kgat_v5_genius_ft.py (LOST in sandbox recycle; protocol fully recoverable from preregs) | e4322d9f9583d9a9 … d0e45c40ee7b772f |
| scripts/kgat_v6_dva.py (v6 rung script) | e857eb921c435f7c |
| download/wspl_kgat_v6_dva_head_A3.json (921-weight head, 21,284 B) | 84124d7c8bb4cc3a |
| download/wspl_kgat_v6_dva_prereg.json | 6c6ace1bd85a5651 |
| download/wspl_kgat_v4_state_train.json | 42eb4784dd01033e |
| download/wspl_kgat_v4_state_test.json | e28cb4676a7a0719 |
| download/wspl_kgat_v4_state_hle.json | 4c05c79eb2450f07 |
| realbench_cache/embeddings/emb_gsm_train.npy | 7fcb3a59281fac35 |
| realbench_cache/embeddings/emb_gsm_test_full.npy | 9b0da6379b5bd3d4 |

State files with full per-item candidate pools: `download/wspl_kgat_v{2..5}_state_*.json`
(train/test/hle/ablation slices, 0.5–6 MB each).