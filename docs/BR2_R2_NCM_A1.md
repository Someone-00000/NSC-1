# BR-2 Round 2 + NCM-A1: Whole-Answer Pools, Genius Refinement, and the Mixing Question — Final Verdict

**Date:** 2026-11-24 (UTC+8) | **Parents:** BR-2 round 1, BR-1, NCM-C1 | **Discipline:** WSPL (prereg + amendments frozen pre-eval, no gradients, as-run times, SHA256-tracked, sandbox-recycle disclosures inline)

## The user's two-part theory, operationalized

1. **Mix genius weights with the model's normal weights** so both "hold equal context together" (steering/co-writing).
2. **Generate multiple WHOLE ANSWERS together**, then the genius **reasons over the drafts and refines them into ONE answer** — parallelism at the answer level.

Two preregistered experiments tested it:

| Experiment | Mechanism | Prompt | Pool | Refinement rule (frozen) |
|---|---|---|---|---|
| **BR-2 round 2** (`wspl_br2_prereg.json`, amended 09:04) | serial K=4 drafts: greedy + 3 sampled T=0.7 | digit-dense (BR-1) | MGEN state, 120 candidates | **M1** = min-max(genius)+min-max(model) 50/50, **grouped by predicted value**, winner = argmax group weight; **M2** = majority vote control; **M3** = genius-only |
| **NCM-A1** (`wspl_ncm_a1_prereg.json`, frozen 11:06) | **batched** K=4 rows in one forward/step: anchor α=0, genius co-write α=4 (z-hat rule), sample T=0.8, sample T=1.0 | digit-dense (BR-1) | pool state, 120 rollouts | **sel_genius** = argmax mean(agreement×conf); **sel_logprob** control; **sel_fusion** = z(genius)+z(logprob) |

## GATES (all re-verified after two mid-run sandbox recycles)

- GATE-3 model identity: `fdf756fa…` PASS (weights re-fetched + re-hashed after each recycle)
- GATE-H heads identity: `d2885e6b…` PASS
- GATE-0 gold alignment: 30/30 PASS (both experiments)
- GATE-alpha0 / anchor validity: **30/30 bit-identical** to BR-1 arm A token ids in BOTH pools (batched-forward fp differences flipped nothing)

## Results (30 frozen GSM8K items, indices 7+43k)

### BR-2 round 2 — serial pool + genius refinement

| Arm | strict | acc | vs anchor 11/30 (36.67%) |
|---|---|---|---|
| M0 anchor (greedy, = A_br) | 11 | 36.67% | — |
| **M1 fusion refinement (primary)** | **15** | **50.0%** | **+4 items = +13.3pp → SUCCESS band** |
| M2 majority vote (control) | 13 | 43.33% | +2 items |
| M3 genius-only | 13 | 43.33% | +2 items |
| M4 mixing-as-written | 11 | 36.67% | 0 (trigger never fired — see disclosure) |
| Oracle best-of-4 | 17 | 56.67% | ceiling |
| **H_B (genius adds): YES** (+2 over majority) | | | **H_D user theory: WORKS** |

- M1 recovered 4 anchor-wrong items (566, 695, 910, 996) and destroyed **zero** anchor-correct items.
- Cost (honest): 87.7 s serial per item (~2.7× single-answer) — accuracy win, not a speed win (frozen H_C had no speed gate).
- Tie to prior work: M1's 50.0% exactly ties the C1 verbose baseline (50.0%) **on the digit-dense prompt**, where single-answer greedy only gets 36.67%.

### NCM-A1 — batched pool (speed half of the theory)

| Metric | Value | Frozen band | Verdict |
|---|---|---|---|
| Aggregate tok/s | 9.96 vs 8.69 = **1.15×** | SUCCESS ≥1.5×, PARTIAL ≥1.2× | **FAIL** |
| Best selection arm (all three) | 11/30 = 36.67% | PARTIAL ≥41.67% | **FAIL** |
| Diversity (≥2 distinct preds) | 29/30 = 96.7% | target ≥50% | CONFIRMED |
| Genius signal (sel_genius ≥ sel_logprob) | 11 vs 11 | YES/NO | YES (tie — disclosed) |
| Tokens per forward | 1.0 → **2.61** | (mechanism diagnostic) | batching amortizes weight reads |
| Oracle best-of-4 | 16/30 = 53.33% | diagnostic | pool has headroom |

- Speed decomposition (honest): the 4× token-per-forward gain is real (2.61× measured), but 4× prefill per item plus per-row Python overhead dilute end-to-end to 1.15× — just under the 1.2× PARTIAL bar.
- **The killer finding — the genius score is circular:** `sel_genius`/`sel_fusion` picked the α=4 genius-co-written row on 22/30 items. The genius rates text it co-wrote highest even when that text is numeric garbage (item 7: genius score 0.635 on a `2e+306` prediction vs 0.244 on the sane anchor). A mirror scoring its own reflection. Selection based on the genius's agreement with genius-written text cannot recover the pool's headroom.

## Weight mixing: exhaustively negative (three independent tests)

1. **Calibration (round 1):** α 0/0.5/1.0 tie on identical correct-sets → α*=0, arm skipped; trajectories identical up to α=8.
2. **M4 as-written (round 2):** frozen trigger "softmax max-conf ≥ 0.938" never fires in 5,419 steps → bit-identical baseline replica (30/30 ids verified).
3. **α=4 co-writing (NCM-A1):** same 36.67% as anchor, corrupts arithmetic into `2e+306`/`inf` on 5+ items while the genius scores it *highest*.

**Why:** the H1 genius head is a closed-form compression of the model's own teacher-forced behavior — it agrees with the model's argmax ~96% of the time. Mixing a mirror into the mirror adds nothing; forcing it to lead corrupts arithmetic.

## Disclosures (full, per WSPL)

- **Prereg scale bug (found at run time):** prereg M4 says "softmax max-conf ≥ tau_1 (0.938)", but BR-1's tau[0]=0.9378 was calibrated on the **raw max-logit** scale (`br1_build.py`: `conf=Zva.max(1)`). Softmax ≥ 0.938 is effectively unreachable → M4 ≡ baseline. Frozen text wins as-written; reported as a no-op result. The earlier in-session mixing run (13 flips, 43.3%, verbose prompt) lost its state in a sandbox recycle and is superseded by this verifiable re-run.
- **Two sandbox recycles** (between items 222→265, and post-completion): torch/transformers reinstalled, model re-fetched + SHA re-verified, gsm8k cache rebuilt; both pool states re-run/recovered with bit-identical counters (MGEN 25707/25696, M4 5419/5419, pool 9609/25034) — an unplanned determinism verification.
- **Voided slices:** amendment_1 (raw-z α grid, 7 items) and amendment_2 (α=8/16 digit soup, 4 items) disclosed in the NCM-A1 prereg — no results reported from them.
- Answer-region interpretation and tie-break conventions disclosed inside the results JSONs.
- BR-2 round-1 artifacts (prereg 48779fa8→31faeb32, calibration 38baa372, results with A2/P arms) were lost in the final hard reset; their SHA256 identities are recorded in `results/wspl_br2_round2_results.json` and the numbers survive in this doc's round-1 references (A2 rebaseline 15/30 = 50.0% bit-identical to C1; P arm 12/30 = 40.0% FAIL vs oracle 60%).

### Addendum (2026-11-25, post-publication recovery and verification)

- **Recovery:** the snapshot restore behind the losses above was partial — `results/wspl_br2_results.json` (round-1, mtime 08:47:51Z) and the round-1 doc `docs/BR2.md` **survived** and are now pushed verbatim; the "lost" statement above is thereby narrowed to the round-1 *state* files (`state_A2`, `state_P`) and the original (pre-amendment) prereg bytes, which remain lost.
- **Chain re-verified on disk (2026-11-25, before the second rollback):** amended prereg `31faeb32…` PASS, calibration `38baa372…` PASS, genius heads `d2885e6b…` PASS, `br2_eval.py` = round-2 as-run `04f82e14…` PASS, `ncm_a1_eval.py` = `c398374f…` PASS, `br1_build.py` = `c7312b23…` PASS, `br1_eval.py` = `ec6c26e7…` PASS, `br2_common.py` = `b49adae0…` PASS, `br2_calibrate.py` = `813be463…` PASS.
- **Two provenance gaps disclosed (verdict unaffected):**
  1. The round-2 `assemble_script` SHA `f345ecfa…` does not match the surviving `br2_assemble.py` (`26db928a…`, mtime 08:47 = the round-1 assembler). The snapshot restore rolled the round-2 assembler back; its exact bytes are lost and only the SHA survives in the results JSON. Mitigation: the assembled per-item records were re-verified against the surviving MGEN state — **27/27 overlapping items match pred-for-pred** — and the three items missing from the rolled-back state (1168, 1211, 1254) are complete in the results JSON.
  2. `state_MGEN.json` on disk now holds a 27/30 partial (tokens 23,807, SHA `dc048381…`), not the complete 30/30 state (`007c15f7…`, tokens 25,696) recorded in the results JSON. Same rollback mechanism; the complete state exists only as that recorded SHA.
- **Code publication attempted and pre-empted (2026-11-25):** a further sandbox snapshot rollback destroyed the WSPL script bytes (all of `kgat_v*.py`, `ncm_c1_*.py`, `br1_*.py`, `br2_*.py`, `ncm_a1_eval.py`) before they could be committed to this repo; no byte-exact recovery path exists from the rolled-back snapshot. The experiment code is therefore NOT in this repo. The script SHAs recorded inside each results JSON remain the only identity anchors for the as-run code, and the pre-push hash verification logged in this addendum (2026-11-25) is the last independent record that those identities held on disk immediately before the rollback. KGAT-ladder scripts were likewise never published — no pushed digest records their SHAs.

## Bottom line

> **Generate whole answers together, then let genius+model jointly reason them into one — that WORKS (50.0%, +13.3pp, zero anchor damage, beats majority by 2 items). But it is an accuracy lever that costs compute, not a speed lever: batched generation only reaches 1.15× end-to-end, and weight-mixing the genius into the model is a dead end in every form we tested.** The next lever for speed is making the *refinement* free (the genius head already lives in RAM next to the weights) rather than making generation wider.

## SHA table

| Artifact | SHA256 (first 16) |
|---|---|
| wspl_ncm_a1_prereg.json | see results JSON `sha256.prereg` |
| wspl_ncm_a1_pool_state.json | `10540cae2e6f3cad…` (1.4MB, kept sandbox-side) |
| wspl_ncm_a1_results.json | assembled from state, script SHA logged |
| wspl_br2_round2_results.json | assembled from state, script SHA logged |
| model.safetensors | `fdf756fa7fcbe740…` (frozen) |
| wspl_br1_heads.npz | `d2885e6bf7a73106…` (frozen) |

Full 64-hex digests are inside each results JSON (`sha256` block) — self-verifying.