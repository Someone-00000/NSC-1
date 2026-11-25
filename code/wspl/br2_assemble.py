"""Assemble wspl_br2_results.json: A2 re-baseline, M calibration verdict, P parallel-refine arm."""
import json, hashlib

ROOT = "/home/z/my-project"
ev = f"{ROOT}/realbench_cache/br2_eval"

def sha(p):
    h = hashlib.sha256(); h.update(open(p, "rb").read()); return h.hexdigest()

a2 = json.load(open(f"{ev}/state_A2.json"))
p = json.load(open(f"{ev}/state_P.json"))
cal = json.load(open(f"{ROOT}/download/wspl_br2_calibration.json"))

# A2 summary
a2_items = a2["items"]
n = len(a2_items)
a2_strict = sum(v["strict"] for v in a2_items.values())
a2_toks = sum(v["n_tokens"] for v in a2_items.values())
a2_walls = sum(v["wall_s"] for v in a2_items.values())
A2 = {"n": n, "strict": a2_strict, "strict_acc": a2_strict / n, "tokens": a2_toks,
      "wall_gen_s": round(a2_walls, 1), "tok_per_s": round(a2_toks / a2_walls, 2)}

# P summary
P_items = p["items"]
np_ = len(P_items)
win_strict = sum(v["winner_strict"] for v in P_items.values())
oracle = sum(v["oracle_bestof4"] for v in P_items.values())
walls = sum(v["wall_s"] for v in P_items.values())
uniq_frac = sum(v["n_unique"] for v in P_items.values()) / (4 * np_)
anchor_strict = sum(a2_items[str(i)]["strict"] for i in map(int, P_items.keys()))

# selection analysis (clean counters)
sel = {"oracle_true_items": 0, "anchor_right_of_those": 0, "anchor_wrong_but_pool_has_correct": 0,
       "kept_correct_anchor": 0, "lost_correct_anchor_to_wrong_draft": 0,
       "recovered_salvageable": 0, "top_score_ties": 0, "tie_break_picked_shorter_sampled": 0}
for i, v in P_items.items():
    drafts = v["drafts"]
    top = max(d["score"] for d in drafts)
    ties = [d for d in drafts if d["score"] >= top - 1e-9]
    anchor_d = drafts[0]
    any_correct = any(d["strict"] for d in drafts)
    if any_correct:
        sel["oracle_true_items"] += 1
        if anchor_d["strict"]:
            sel["anchor_right_of_those"] += 1
            if v["winner_strict"]:
                sel["kept_correct_anchor"] += 1
            else:
                sel["lost_correct_anchor_to_wrong_draft"] += 1
                if len(ties) > 1 and ties[0]["kind"] != "greedy_anchor":
                    sel["tie_break_picked_shorter_sampled"] += 1
        else:
            sel["anchor_wrong_but_pool_has_correct"] += 1
            if v["winner_strict"]:
                sel["recovered_salvageable"] += 1
    if len(ties) > 1:
        sel["top_score_ties"] += 1

# winner kind distribution
wk = {}
for v in P_items.values():
    wk[v["winner_kind"]] = wk.get(v["winner_kind"], 0) + 1

# C1 comparison for oracle framing
c1_oracle = 0.3995451099317665

res = {
 "version": "br2",
 "date": "2026-11-24",
 "prereg": "download/wspl_br2_prereg.json (with amendment_1, frozen pre-eval)",
 "sha256": {
  "prereg": sha(f"{ROOT}/download/wspl_br2_prereg.json"),
  "calibration": sha(f"{ROOT}/download/wspl_br2_calibration.json"),
  "heads_c1": sha(f"{ROOT}/download/wspl_ncm_c1_heads.npz"),
  "model_safetensors": sha(f"{ROOT}/models/qwen25-05b/model.safetensors"),
  "state_A2": sha(f"{ev}/state_A2.json"),
  "state_P": sha(f"{ev}/state_P.json"),
  "br2_common_py": sha(f"{ROOT}/scripts/br2_common.py"),
  "br2_calibrate_py": sha(f"{ROOT}/scripts/br2_calibrate.py"),
  "br2_eval_py": sha(f"{ROOT}/scripts/br2_eval.py")
 },
 "amendments": json.load(open(f"{ROOT}/download/wspl_br2_prereg.json"))["amendments"],
 "environment": {"torch": "2.15.0+cpu", "transformers": "5.24.0", "threads": 2, "dtype": "float32",
                 "heads": "C1 H1..H4 npz (verbose-format-matched); H1 used for mixing/scoring"},
 "calibration": {
  "summary": cal["summary"], "alpha_star": cal["alpha_star"],
  "mechanism_M_verdict": cal["mechanism_M_verdict_at_calibration"],
  "note": "all three alphas tie 4/12 with identical correct-sets; divprobe (disclosed) shows greedy trajectories identical to plain at alpha up to 8.0 - genius-H1 agrees with model argmax ~96%, mixing cannot steer greedy decode"
 },
 "A2_rebaseline": {**A2, "vs_C1_armA": "strict 0.50 == 0.50, tokens 8065 == 8065 (bit-identical regeneration post-recovery), tok/s 8.39 vs 8.35"},
 "M_arm": {"status": "SKIPPED per frozen calibration rule (alpha*=0)"},
 "P_arm": {
  "n": np_, "winner_strict": win_strict, "winner_strict_acc": win_strict / np_,
  "oracle_bestof4": oracle / np_, "anchor_strict_in_pool": anchor_strict,
  "wall_total_s": round(walls, 1), "avg_wall_per_item_s": round(walls / np_, 1),
  "wall_multiplier_vs_A2": round(walls / A2["wall_gen_s"], 2),
  "unique_draft_fraction": round(uniq_frac, 3),
  "winner_kind_dist": wk,
  "selection_analysis": sel
 },
 "bug_disclosure": {
  "scoring_index_bug": "first P run scored 0.0 everywhere: argmax index (0..11) was compared to token id (15..24/198/151645); caught by position-level debug on item 7 (actual agreement 5/5 in window), fixed to U_IDS[argmax]==next, poisoned P state wiped and rerun from scratch - no results were reported from the buggy run"
 }
}

# frozen-band verdicts
d_m = 0.0  # M skipped -> mechanism failed at calibration
dP = (win_strict / np_ - A2["strict_acc"]) * 100
res["verdict_bands"] = {
 "M_vs_A2": "FAIL AT CALIBRATION (alpha*=0.0 per frozen tie rule; arm skipped)",
 "P_vs_A2": ("WIN" if dP >= 6.67 else ("NEUTRAL" if dP > -6.67 else "FAIL")) + f" (delta {dP:+.1f}pp, band +/-6.67pp)",
 "speed_M": "n/a (M skipped)",
 "speed_P": f"wall multiplier {res['P_arm']['wall_multiplier_vs_A2']}x (P costs more wall by design: 3 extra drafts + scoring pass; no speed claim)"
}
res["verdict"] = (
 f"MIXED MIND FAILS AS A STEERER; WHOLE-ANSWER POOL HAS REAL HEADROOM BUT THE GENIUS AGREEMENT SCORE CANNOT HARVEST IT - P FAILS THE FROZEN BAND. "
 f"(1) Mixing genius weights into the model's own output weights is a NO-OP on greedy decode: calibration tied 4/12 across alpha 0/0.5/1.0 (identical correct-sets), and the disclosed probe shows trajectories identical at alpha up to 8.0. The genius head was compiled FROM the model's own behavior and agrees with its argmax ~96% of the time - it is a compressed mirror, not an independent expert, so 'both minds holding context together' adds nothing the model does not already think. M arm skipped per frozen rule. "
 f"(2) Whole-answer parallelism itself is validated: 4/4 unique drafts on every item, and the best-of-4 oracle ceiling is {oracle}/{np_} = {oracle/np_:.1%} vs single-answer {A2['strict_acc']:.1%} - the pool DOES contain better answers, the user's intuition is right. "
 f"(3) But the genius refinement step cannot find them: raw agreement saturates (top-score ties on {sel['top_score_ties']}/30 items), recovered {sel['recovered_salvageable']}/{sel['anchor_wrong_but_pool_has_correct']} salvageable items, and its shorter-tie-break threw away {sel['lost_correct_anchor_to_wrong_draft']} correct anchors (tie-break picked a wrong, shorter sampled draft). Winner strict {win_strict}/{np_} = {win_strict/np_:.1%} vs anchor {A2['strict_acc']:.1%} = {dP:+.1f}pp -> FAIL per frozen band (and {res['P_arm']['wall_multiplier_vs_A2']}x wall cost). "
 f"HONEST BOTTOM LINE: per the user's own rule, P as implemented is NOT a win - accuracy dropped and wall time grew 3.5x. The experiment localizes the failure precisely: the pool has the answers (oracle 60%), the selector is the wall (agreement saturates at 1.0). The fix is a discriminating score - e.g. genius-model disagreement density or confidence-profile matching - not more drafts."
)
json.dump(res, open(f"{ROOT}/download/wspl_br2_results.json", "w"), indent=1)
print(json.dumps({"A2": A2, "P": res["P_arm"], "verdict_bands": res["verdict_bands"], "selection_analysis": sel}, indent=1))
