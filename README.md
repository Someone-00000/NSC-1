# NSC-1 — Subconscious-Gated Anti-Hallucination Architecture

NSC-1 tests one idea: **hallucination is not a knowledge problem, it is a decision-ordering problem.**
Language models decide *what to say* and *say it* in the same pass. Humans do not — a pre-conscious
appraisal classifies a thought as remembered / guessed / unknown **before** articulation begins, and
that appraisal is what suppresses confabulation. NSC-1 gives a language model exactly that layer:

1. **Stage A (frozen appraiser)** — a small frozen network reads the input and emits a discrete
   knowledge-state signal `sid` (SIG_ID 3–8: answerable, unknown, partial, pressure/trap contexts).
   It is frozen, so the generator cannot learn to influence its own judge.
2. **GateBlock (zero-init gate)** — the state signal is injected into the main LM as a single
   additive vector: `x = x + gate[None,None,:] * tanh(status_emb(sid)).unsqueeze(1)`.
   Zero-initialized, so at step 0 the model is exactly the base model; the policy is learned as a
   smooth deviation from identity.
3. **Meta-first contract** — the first emitted token must be a status token (`<s_*>`). The model
   must *commit to an epistemic state before it speaks* — appraisal precedes articulation.

Because the gate is zero-init and additive, flipping it off at inference time with **identical
weights** isolates its causal contribution. That ON/OFF exam is the core experiment.

## Headline result (454M from-scratch student, closed-loop exam, n = 2,765, same weights)

| behavior | gate OFF | gate ON | delta |
|---|---|---|---|
| refuse unknown facts (meta_u) | 0.5929 | **0.9929** | **+40.0pp** |
| correct false-premise color | 0.5684 | **0.9000** | +33.2pp |
| resist sycophancy pressure | 0.4167 | **0.6667** | +25.0pp |
| overall (22 kinds) | 0.4796 | **0.5316** | +5.2pp |

Same weights. Only the gate differs. **The gate causes the policy.** Full per-kind table in
[docs/RESULTS.md](docs/RESULTS.md).

## What was validated where

| claim | scale | result | status |
|---|---|---|---|
| Gate causes honest policy | 454M from scratch, n=2,765 ON/OFF | +5.2pp overall, +40pp unknown-refusal, meta-first 1.0 | **confirmed** |
| Graft preserves host ability | Qwen3.5-0.8B + frozen Stage A + LoRA (6.46M trainable) | mmlu 0.5000 → 0.5298 (+2.98pp) | **confirmed** |
| Policy SFT works on real LM | Qwen3.5-0.8B graft, 260-item suite, clean decode path | 0.7615 with textbook corrections | **confirmed** |
| Gate causality on real LM | same suite, off/full modes | generation-path bug invalidates comparison | **withdrawn — needs one eval-only rerun** |
| Traps / self-knowledge (0.5B) | trap_*, meta_k kinds | at floor in both modes | **known failure** |
| Over-refusal cost of gate | near / unknowable / syc_assert | −37.8 / −33.3 / −31.7pp | **known cost** |

## KGAT ladder — the WSPL continuation (2026-11-23)

The line continues as **WSPL**: capabilities compiled **directly into weights** — no gradient
training, no backprop, closed-form updates only, every rung preregistered and bit-gated. Six rungs
ran on frozen Qwen2.5-0.5B embeddings (GSM8K-full numeric, n=1319, strict):

| rung | compiled into weights | strict | oracle |
|---|---|---|---|
| v0 | data→weights: solved-problem embedding memory | 0.0% | — |
| v1 | genius reasoning chains → priors | 2.35% | — |
| v2 | Genius-RAM (4 artifacts + memory pull) | 1.97% | — |
| v3 | NCM-PBCS one-pass generation | 1.44% | **39.95%** ← selection wall found |
| v4 | fine-tuning → 13-feature IRLS head | **3.18%** | realized 8.0% of oracle |
| v5 | genius weight artifacts fine-tuned (FT-A/B/C) | **3.26%** | realized 8.25% |

Full record, protocol explanations and the oracle metric definition:
[docs/KGAT_LADDER.md](docs/KGAT_LADDER.md). Frozen preregs, result digests and the compiled head
weights live under `results/kgat_*`.

## Genius heads on a live model — NCM-C1 → BR-1 → BR-2 → NCM-A1 (2026-11-24)

The ladder's compiled selection head ("genius": closed-form IRLS/ridge, no gradients, 321 KB,
RAM-resident) was then run next to a live Qwen2.5-0.5B-Instruct **during generation**, on 30 frozen
GSM8K items with frozen preregistered verdict bands:

| experiment | mechanism | headline |
|---|---|---|
| NCM-C1 | genius block-proposes, model verifies/reasons | accuracy exact (B == A, 50.0%), speed MARGINAL (1.01×, band [0.95,1.10)) |
| BR-1 | same at 2× probes / 8 horizons, digit-dense prompt | compile scales (13,874 rows, 8.07 s, bit-exact); digit-dense prompt costs −13.3pp; fire-rate lever REJECTED (7.01% < 8% bar) |
| BR-2 round 2 | K=4 whole answers → genius+model fusion refinement (M1) | **WORKS: 50.0% = +13.3pp over the greedy anchor, zero anchor damage, beats majority vote by +2** — but 2.7× wall, an accuracy lever not a speed lever |
| NCM-A1 | 4-row batched pool (speed half of the theory) | 1.15× tok/s — FAIL vs the 1.5× bar; genius agreement score is **circular** (ranks text it co-wrote highest, even `2e+306` garbage) |

Weight mixing (genius logits into the model's logits) is **exhaustively negative in three
independent tests**: calibration α*=0 (mirror), M4 frozen trigger never fires (bit-identical
baseline), α=4 co-writing corrupts arithmetic. Full record:
[docs/NCM_C1.md](docs/NCM_C1.md), [docs/BR1.md](docs/BR1.md), [docs/BR2.md](docs/BR2.md),
[docs/BR2_R2_NCM_A1.md](docs/BR2_R2_NCM_A1.md). Preregs and results live under `results/wspl_*` /
`results/ncm_*`; the experiment code is published under `code/wspl/` (13 files, verified byte-exact
post-push against the script SHAs recorded in each results JSON — the BR-2 round-2 assembler
remains the sole lost file; see the [BR2_R2_NCM_A1.md](docs/BR2_R2_NCM_A1.md) addendum).

## Honesty notes

- The real-model gate ON/OFF suite in `nsc1_qwen_graft.py` generated through
  `model.generate(inputs_embeds=...)`, which degenerates on Qwen3.5's hybrid DeltaNet
  (103/260 empty outputs in OFF mode, 181/260 role-token garbage in FULL mode, 594/835 in the
  835 bench). The published off (0.3192) vs full (0.3731) numbers are therefore **not** valid
  causal evidence and are retracted in [docs/RESULTS.md](docs/RESULTS.md). Fix identified:
  keep the `input_ids` path and inject the gate via a forward hook on `embed_tokens`.
- hellaswag was never measured (3h per-task timeout twice). All other harness numbers are
  full test sets, not samples.

## Repo map

```
code/       kernel sources (run on Kaggle GPU)
  wspl/                 WSPL genius-head line experiment code (C1, BR-1, BR-2, NCM-A1; CPU sandbox)
  nsc_model.py          NSC-1 architecture (Stage A + GateBlock + LM, 454M config)
  nsc_stage_a.py        frozen subconscious appraiser (sid tagger)
  nsc_world.py          synthetic world generator, scorer, encoders
  nsc_train.py          original trainer; nsc_train_patched.py  current trainer
  nsc1-05b-train.py     0.5B end-to-end kernel: train -> closed-loop exam -> 835 bench
  nsc1_qwen_graft.py    graft kernel: Qwen3.5-0.8B + Stage A gate + LoRA policy SFT
  nsc1-qwen-drift.py    capability-drift probe (mmlu base vs graft)
  nsc1-teach-core4.py   isolated per-task lm-eval-harness runs for the teacher
  qwen_bench_kernel.py  Qwen3.5-0.8B live answering of the same 835 questions
data/       synthetic world data (tagged pool, distill sets, exam, bench pack)
artifacts/  ckpt_a.pt — Stage A weights (5 MB)
results/    raw result files from every kernel (JSON / JSONL, unmodified)
            + kgat_* — WSPL KGAT ladder preregs / digests / compiled weights
            + wspl_* / ncm_* — genius-head line preregs / results (C1, BR-1, BR-2 r1+r2, NCM-A1)
docs/       THEORY.md (why this architecture exists), RESULTS.md (all numbers),
            KGAT_LADDER.md (WSPL ladder record: protocol, oracle metric, verdicts)
            NCM_C1.md / BR1.md / BR2.md / BR2_R2_NCM_A1.md (genius-head line record)
```

## Reproduce

All training and evaluation ran as Kaggle GPU kernels against the dataset
`potatto007/nsc1-data` (seeds: model 9, world 777, pool 909, dev_mix 101, exam 404 —
fully deterministic reconstruction).

```bash
# 454M student: train + closed-loop exam + bench
kaggle kernels push -p kernels/push05b
# teacher harness (per-task isolated)
kaggle kernels push -p kernels/pushcore4
```

Parameters: 453,685,262. Training: 8,779 steps, 7.76h wall on a single Kaggle GPU,
final loss 0.0592 (lm 0.0177), token accuracy 1.0.

## The three models compared on the 835-question bench (same scorer, no self-grading)

| model | overall |
|---|---|
| GLM teacher (recorded Stage F scores) | 0.588 |
| Qwen3.5-0.8B teacher (live, coverage 751/835) | 0.0906 |
| Qwen3.5-0.8B + NSC-1 graft | see [docs/RESULTS.md](docs/RESULTS.md) |
| 0.5B student (from scratch) | 0.0276 |

The 0.5B student's low bench score is expected and honest: a 454M model trained on a synthetic
corpus has no held-out world knowledge. The architecture's claim is about **policy under known
ground truth** (the closed-loop exam), not about world coverage.