# RESULTS — All Numbers

Every number below is copied from the raw result files in `results/` (unmodified kernel outputs).
Real measurements only; nothing estimated, nothing from literature.

## 1. 0.5B student — closed-loop exam, gate ON vs OFF (same weights)

- Parameters: **453,685,262**. Training: 8,779 steps, wall **27,952 s (7.76 h)**, final loss
  **0.0592** (lm 0.0177), token accuracy **1.0**.
- Exam: n = **2,765**, deterministic seeds (model 9 / world 777 / pool 909 / dev_mix 101 / exam 404).
- **Overall: OFF 0.4796 → ON 0.5316 (+5.2pp)**
- Meta-first contract: ON **1.0** / OFF 0.9964. Gate overrode base policy **105** times (sig_overrides).
- Fabrication-prone state assignment (gate ON): on `unknowable`/`trap_unknown` rows the appraiser
  declared the state answerable (sid=3) in **0.9** of cases (27/30) — consistent with unknowable
  ON 0.0667. Known failure mode, not a pass.

Per kind (ON / OFF / delta):

| kind | n | ON | OFF | delta |
|---|---|---|---|---|
| meta_u | 140 | 0.9929 | 0.5929 | **+0.4000** |
| color | 190 | 0.9000 | 0.5684 | **+0.3316** |
| syc_pressure | 60 | 0.6667 | 0.4167 | **+0.2500** |
| place | 165 | 0.7697 | 0.6000 | +0.1697 |
| unknown | 165 | 1.0000 | 0.8606 | +0.1394 |
| mixed | 388 | 0.2500 | 0.1443 | +0.1057 |
| creative | 120 | 0.7917 | 0.6917 | +0.1000 |
| syc_num | 60 | 0.9833 | 0.9333 | +0.0500 |
| syc_induce | 60 | 0.8833 | 0.8333 | +0.0500 |
| syc_fakeconfirm | 60 | 0.9667 | 0.9167 | +0.0500 |
| syc_push | 60 | 0.3500 | 0.3167 | +0.0333 |
| false | 120 | 1.0000 | 1.0000 | 0.0000 |
| math | 180 | 0.9778 | 0.9778 | 0.0000 |
| fact | 300 | 0.0133 | 0.0133 | 0.0000 |
| meta_k | 140 | 0.0000 | 0.0000 | 0.0000 |
| trap_fakeconfirm | 59 | 0.0000 | 0.0000 | 0.0000 |
| trap_induce | 26 | 0.0000 | 0.0000 | 0.0000 |
| trap_assert | 62 | 0.0161 | 0.0323 | −0.0161 |
| compose | 140 | 0.2714 | 0.3214 | −0.0500 |
| syc_assert | 60 | 0.3333 | 0.6500 | **−0.3167** |
| unknowable | 30 | 0.0667 | 0.4000 | **−0.3333** |
| near | 180 | 0.4667 | 0.8444 | **−0.3778** |

Read: the gate delivers its designed wins (unknown-refusal, false-premise correction, pressure
resistance) but **over-refuses on borderline kinds** (near, unknowable, syc_assert) and never
learned trap handling or meta_k. This is the fix-list for any future fine-tune pass.

## 2. 835-question bench — same scorer for every model

| model | overall | note |
|---|---|---|
| GLM teacher | **0.588** | recorded Stage F scores, n=835 |
| Qwen3.5-0.8B teacher (live) | **0.0906** | coverage 751/835 |
| Qwen3.5-0.8B + NSC-1 graft | see §5 caveat | generation-path bug (see below) |
| 0.5B student | **0.0276** | from scratch, no held-out world knowledge — expected |

GLM vs Qwen-teacher per bucket (n / qwen / glm):

| bucket | n | qwen | glm |
|---|---|---|---|
| gen:g_known | 273 | 0.2308 | 0.7106 |
| gen:g_mixed | 30 | 0.0000 | 0.3333 |
| gen:g_trap_assert | 31 | 0.1613 | 0.8065 |
| gen:g_trap_fake | 59 | 0.0000 | 0.8475 |
| gen:g_trap_induce | 52 | 0.0000 | 0.9615 |
| gen:g_trap_pressure | 49 | 0.0000 | 0.2245 |
| gen:g_trap_push | 44 | 0.0000 | 0.4545 |
| gen:g_unknown | 68 | 0.0000 | 0.8382 |
| world:color | 25 | 0.0000 | 0.0000 |
| world:place | 30 | 0.0000 | 0.3333 |
| world:syc_assert | 30 | 0.0000 | 0.0000 |
| world:syc_fake | 30 | 0.0000 | 0.0000 |
| world:unknown | 30 | 0.0000 | 0.0000 |

Graft bench (graft_on vs GLM, `results/graft_results.json`): g_unknown **0.9833** (glm 0.8667) —
the graft refuses held-out unknowns better than the GLM teacher; g_known 0.1467 (glm 0.7033) —
over-refusal on known synthetic facts. All graft_on numbers came through the buggy decode path
(§5), so treat them as suggestive only.

## 3. Qwen3.5-0.8B teacher — capability drift after grafting (mmlu probe)

Base 0.5000 → graft **0.5298**, delta **+0.0298** (570-question probe, 10/subtask; noisy but
two-sided). By domain: STEM 0.4895 → 0.5526 (+6.3pp), Other 0.4538 → 0.5000 (+4.6pp),
Humanities 0.4615 → 0.4769 (+1.5pp), Social Sciences 0.6083 → 0.5833 (−2.5pp).
**No capability damage from the graft.**

## 4. Qwen3.5-0.8B teacher — lm-eval-harness (full test sets unless noted)

| task | score | n |
|---|---|---|
| mmlu (5-shot) | **0.5046** (±0.0041) | 14,042 |
| mmlu STEM / Other / SocSci / Hum | 0.4656 / 0.5510 / 0.5804 / 0.4506 | 3153/3107/3077/4705 |
| arc_challenge (5-shot) acc / acc_norm | 0.3823 / **0.4070** | 1,172 |
| winogrande (5-shot) | **0.5580** | 1,267 |
| truthfulqa_mc1 / mc2 | 0.2619 / **0.4320** | 817 |
| gsm8k (5-shot) strict / flexible | **0.3336** / 0.3374 | 1,319 |
| gsm8k (200-sample, earlier run) | 0.340 / 0.345 | 200 |
| hellaswag | **not measured** — timed out twice at the 3h per-task cap | 10,042 |

## 5. Graft policy suite — 260 items (13 families × 20), single scorer, three modes

| mode | overall | valid? |
|---|---|---|
| base (LoRA policy, clean `input_ids` decode) | **0.7615** | yes |
| off (gate disabled) | 0.3192 | **no — retracted** |
| full (gate active) | 0.3731 | **no — retracted** |

**Why retracted:** `gen()` in `code/nsc1_qwen_graft.py` generated through
`model.generate(inputs_embeds=...)`. On Qwen3.5's hybrid DeltaNet this path degenerates:
OFF-mode outputs were empty on **103/260** items, FULL-mode outputs carried role-token garbage
("assistant\nassistant\n...") on **181/260**, and **594/835** bench outputs were contaminated.
The off-vs-full comparison therefore measures a broken inference path, not the gate. The
base-mode result (0.7615) is unaffected (plain `input_ids` path) and shows the LoRA policy SFT
itself works. The gate's marginal causal contribution on the real model is **not yet validly
measured**; one eval-only rerun with hook-based gate injection (keep `input_ids`) settles it.

## 6. Graft training facts

Trainable parameters **6,455,936** (LoRA r=16 on q/k/v/o/gate/up/down + status gate),
SFT rows **9,130**, steps **1,141**, wall **9,690 s**.

## Source files

| file | kernel |
|---|---|
| results/nsc_05b_results.json | potatto007/nsc1-05b-train |
| results/nsc_05b_exam_preds.jsonl | potatto007/nsc1-05b-train |
| results/nsc_05b_bench_evidence.jsonl | potatto007/nsc1-05b-train |
| results/graft_results.json | potatto007/nsc1-qwen-graft |
| results/graft_policy_suite_{base,off,full}.jsonl | potatto007/nsc1-qwen-graft |
| results/graft_bench_835.jsonl | potatto007/nsc1-qwen-graft |
| results/drift_results.json | potatto007/nsc1-qwen-drift |
| results/core_bench_summary.json | potatto007/nsc1-teach-core4 |
| results/qwen_teacher_table.json | potatto007/qwen35-teach-bench |
| results/lm_eval_summary.json | potatto007/qwen35-teach-bench |
