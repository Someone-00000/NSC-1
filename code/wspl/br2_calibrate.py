"""BR-2 calibration: pick alpha* for the mixed mind on 12 TRAIN-pool items (zero test data).
Prereg rules frozen: alpha grid {0.0, 0.5, 1.0}; max strict count wins; tie -> smaller alpha;
alpha*=0 -> mechanism-M declared FAIL at calibration (M eval arm skipped), P proceeds regardless.
Greedy, max_new=256 (disclosed in prereg), threads=2 fp32. Checkpoint/resume per (alpha, item).
Usage: br2_calibrate.py [--wall 420]
"""
import os, sys, json, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from br2_common import (ROOT, OUT, U_IDS, register_mixed_mind, build_fewshot,
                        prompt_ids, extract_ans)

ap = argparse.ArgumentParser()
ap.add_argument("--wall", type=float, default=420.0)
args = ap.parse_args()
T0 = time.time()
torch.set_num_threads(2)

MD = f"{ROOT}/models/qwen25-05b"
CAL_IDX = [11 + 41 * k for k in range(12)]
ALPHAS = [0.0, 0.5, 1.0]
MAX_NEW = 256
CDIR = f"{ROOT}/realbench_cache/br2_cal"
os.makedirs(CDIR, exist_ok=True)

from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from datasets import load_from_disk
tok = AutoTokenizer.from_pretrained(MD)
model = AutoModelForCausalLM.from_pretrained(MD, dtype=torch.float32, low_cpu_mem_usage=True)
model.eval()
gsm = load_from_disk(f"{ROOT}/realbench_cache/gsm8k")
FEWSHOT = build_fewshot(tok, gsm)


def gen_alpha(pids, alpha, max_new):
    """Greedy generation with mixed-mind hook at given alpha. Returns token list."""
    handle = register_mixed_mind(model, [alpha]) if alpha > 0 else None
    cache = DynamicCache()
    try:
        with torch.no_grad():
            hb = model.model(input_ids=torch.tensor([pids], dtype=torch.long),
                             past_key_values=cache, use_cache=True)
            h_last = hb.last_hidden_state[:, -1:]           # (1,1,896)
            logits = model.lm_head(h_last)[0, -1].float()   # hook fires here
            gen = []
            for _ in range(max_new):
                nt = int(logits.argmax())
                gen.append(nt)
                if nt == tok.eos_token_id:
                    break
                hb = model.model(input_ids=torch.tensor([[nt]], dtype=torch.long),
                                 past_key_values=cache, use_cache=True)
                logits = model.lm_head(hb.last_hidden_state[:, -1:])[0, -1].float()
        return gen
    finally:
        if handle is not None:
            handle.remove()


state_f = f"{CDIR}/cal_state.json"
state = json.load(open(state_f)) if os.path.exists(state_f) else {}
for a in ALPHAS:
    key = f"alpha_{a}"
    if key not in state:
        state[key] = {"items": {}}

for a in ALPHAS:
    key = f"alpha_{a}"
    done = set(int(k) for k in state[key]["items"])
    for i in CAL_IDX:
        if i in done:
            continue
        if time.time() - T0 > args.wall:
            json.dump(state, open(state_f, "w"))
            print(f"[cal a={a}] wall guard; saved", flush=True)
            sys.exit(2)
        row = gsm["train"][i]
        ans = row["answer"]
        import re as _re
        m = _re.search(r"####\s*(-?[0-9][0-9,\.]*)", ans)
        gold = float(m.group(1).replace(",", "")) if m else None
        t0 = time.time()
        pids = prompt_ids(tok, FEWSHOT, row["question"])
        gen = gen_alpha(pids, a, MAX_NEW)
        text = tok.decode(gen, skip_special_tokens=False)
        pred = extract_ans(text)
        strict = bool(pred is not None and gold is not None and abs(pred - gold) < 1e-6)
        state[key]["items"][str(i)] = {"strict": strict, "pred": pred, "gold": gold,
                                       "n_tokens": len(gen), "wall_s": round(time.time() - t0, 2)}
        json.dump(state, open(state_f, "w"))
        print(f"[cal a={a}] item {i}: strict={strict} pred={pred} gold={gold} tok={len(gen)}", flush=True)

summary = {}
for a in ALPHAS:
    key = f"alpha_{a}"
    items = state[key]["items"]
    n = len(items)
    sc = sum(1 for v in items.values() if v["strict"])
    toks = sum(v["n_tokens"] for v in items.values())
    walls = sum(v["wall_s"] for v in items.values())
    summary[key] = {"n": n, "strict": sc, "strict_acc": sc / n if n else 0.0,
                    "tok_per_s": round(toks / walls, 2) if walls else None}

# frozen selection rule: max strict; tie -> smaller alpha
best_sc = max(summary[f"alpha_{a}"]["strict"] for a in ALPHAS)
chosen = min(a for a in ALPHAS if summary[f"alpha_{a}"]["strict"] == best_sc)
cal = {
    "version": "br2_calibration",
    "date": "2026-11-24",
    "prereg_sha_frozen": "8f2bd19155004a4904107862cdd4b8cf5f68828c117940f79367a1b8551a4483",
    "cal_items": "train pool indices 11+41k, k=0..11 (zero test data)",
    "max_new": MAX_NEW,
    "summary": summary,
    "alpha_star": chosen,
    "mechanism_M_verdict_at_calibration": "PASS" if chosen > 0 else "FAIL (alpha*=0; M eval arm skipped per frozen rule)",
    "heads_sha256": "3823c4e76519f836fdc36c30ca633df2b3c85c573af99658e7c407f6fad9c25c"
}
json.dump(cal, open(f"{OUT}/wspl_br2_calibration.json", "w"), indent=1)
print(json.dumps(cal, indent=1), flush=True)
print("[cal] DONE", flush=True)
