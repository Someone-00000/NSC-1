"""WSPL BR-2 eval: multiple whole answers generated together, genius reasons over them.

Modes:
  --mode MGEN : per item generate K=4 whole answers (cand0 greedy, cand1..3 sampled T=0.7,
                frozen seeds), score each candidate inline (genius H1 prob of chosen tokens
                over U; model mean logprob), per-candidate checkpoint/resume.
  --mode M4   : literal weight-mixing generation - at confident steps add standardized genius
                H1 logits into model logits restricted to U (gamma=2.0, tau_1 from BR-1 heads).

Prereg: download/wspl_br2_prereg.json (frozen SHA 31faeb32...). No gradients.
Prompt identical to br1_eval.py (digit-dense exemplars train 0,1).
"""
import os, sys, json, time, re, argparse
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True, choices=["MGEN", "M4"])
ap.add_argument("--wall", type=float, default=420.0)
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()
MODE = args.mode
T0 = time.time()

ROOT = "/home/z/my-project"
MD = f"{ROOT}/models/qwen25-05b"
OUT = f"{ROOT}/download"
EVAL_DIR = f"{ROOT}/realbench_cache/br2_eval"
os.makedirs(EVAL_DIR, exist_ok=True)
torch.set_num_threads(2)

MAX_NEW = 16 if args.smoke else 384
N_EVAL = 2 if args.smoke else 30
EVAL_IDX = [7 + 43 * k for k in range(30)][:N_EVAL]
K_CAND = 4
TEMP = 0.7
GAMMA = 2.0

heads = np.load(f"{OUT}/wspl_br1_heads.npz")
W, B, mu, sd, TAU, U_IDS = heads["W"], heads["B"], heads["mu"], heads["sd"], heads["tau"], heads["u_ids"]
W0, B0 = W[0], B[0]           # H1 head
TAU1 = float(TAU[0])
U_LIST = [int(u) for u in U_IDS]
U_INDEX = {u: i for i, u in enumerate(U_LIST)}
U_TENSOR = torch.tensor(U_LIST, dtype=torch.long)

from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
tok = AutoTokenizer.from_pretrained(MD)
model = AutoModelForCausalLM.from_pretrained(MD, dtype=torch.float32, low_cpu_mem_usage=True)
model.eval()

from datasets import load_from_disk
gsm = load_from_disk(f"{ROOT}/realbench_cache/gsm8k")

def digit_dense(answer_raw):
    lines = answer_raw.strip().split("\n")
    out = []
    for ln in lines:
        if "####" in ln:
            out.append(ln.strip())
            continue
        for span in re.findall(r"<<([^>]+)>>", ln):
            out.append(span.strip())
    return "\n".join(out)

_r0, _r1 = gsm["train"][0], gsm["train"][1]
FEWSHOT = [(_r0["question"], digit_dense(_r0["answer"])),
           (_r1["question"], digit_dense(_r1["answer"]))]

def user_content(q):
    e1, e2 = FEWSHOT
    return (f"Here are two worked examples.\n\nExample 1:\nQuestion: {e1[0].strip()}\nAnswer: {e1[1]}\n\n"
            f"Example 2:\nQuestion: {e2[0].strip()}\nAnswer: {e2[1]}\n\n"
            f"Now solve.\nQuestion: {q.strip()}\nAnswer: Let's think step by step. "
            f"Do short calculations, one per line, then end with \"#### <final numeric answer>\".")

state_test = json.load(open(f"{OUT}/wspl_kgat_v4_state_test.json"))["records"]

def prompt_ids(q):
    msgs = [{"role": "user", "content": user_content(q)}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    ids = enc["input_ids"] if isinstance(enc, dict) else enc.input_ids
    if isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return [int(t) for t in ids]

def extract_ans(text):
    m = re.search(r"####\s*(-?[0-9][0-9,\.]*)", text)
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            pass
    nums = re.findall(r"-?[0-9][0-9,\.]*", text)
    if nums:
        try:
            return float(nums[-1].replace(",", ""))
        except ValueError:
            return None
    return None

def genius_z1(h_vec):
    xs = (h_vec - mu) / sd
    return W0 @ xs + B0    # (12,) H1 logits over U

STATE_F = f"{EVAL_DIR}/state_{MODE}.json"
state = json.load(open(STATE_F)) if os.path.exists(STATE_F) else {"items": {}, "counters": {"forwards": 0, "tokens": 0, "mix_steps": 0, "steps": 0}}

def fwd(ids, cache):
    t = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids=t, past_key_values=cache, use_cache=True, output_hidden_states=True)
    state["counters"]["forwards"] += 1
    return out.logits[0, -1].float(), out.hidden_states[-1][0, -1].float().numpy()

def gen_candidate(q, cand, idx):
    """Generate one whole answer; return rec with inline scores."""
    g = torch.Generator(); g.manual_seed(1000003 + idx * 7919 + cand)
    pids = prompt_ids(q)
    cache = DynamicCache()
    cur_logits, cur_h = fwd(pids, cache)
    gen, gp, lp = [], [], []
    wall0 = time.time()
    sampled = cand > 0
    while len(gen) < MAX_NEW:
        state["counters"]["steps"] += 1
        z1 = genius_z1(cur_h)
        p1 = torch.softmax(torch.tensor(z1), dim=0).numpy()
        if MODE == "M4":
            if float(p1.max()) >= TAU1:
                zstd = (z1 - z1.mean()) / (z1.std() + 1e-8)
                z_model_U = cur_logits[U_TENSOR].numpy()
                y = int(U_LIST[int(np.argmax(z_model_U + GAMMA * zstd))])
                state["counters"]["mix_steps"] += 1
            else:
                y = int(cur_logits.argmax())
        elif sampled:
            probs = torch.softmax(cur_logits / TEMP, dim=0)
            y = int(torch.multinomial(probs, 1, generator=g).item())
        else:
            y = int(cur_logits.argmax())
        if y in U_INDEX:
            gp.append(float(p1[U_INDEX[y]]))
        lp.append(float(torch.log_softmax(cur_logits, dim=0)[y]))
        gen.append(y)
        state["counters"]["tokens"] += 1
        if y == tok.eos_token_id:
            break
        cur_logits, cur_h = fwd([y], cache)
    text = tok.decode(gen, skip_special_tokens=False)
    rec = {"cand": cand, "pred": extract_ans(text), "n_tokens": len(gen),
           "wall_s": round(time.time() - wall0, 2),
           "genius_score": round(float(np.mean(gp)), 6) if gp else 0.0,
           "n_U": len(gp),
           "model_score": round(float(np.mean(lp)), 6) if lp else -1e9,
           "text": text, "ids": gen}
    return rec

if MODE == "MGEN":
    items = state["items"]
    for idx in EVAL_IDX:
        rec_item = items.get(str(idx), {"cands": []})
        done_c = {c["cand"] for c in rec_item["cands"]}
        if len(done_c) >= K_CAND:
            continue
        row = gsm["test"][idx]
        gold = float(state_test[idx]["gold"])
        for cand in range(K_CAND):
            if cand in done_c:
                continue
            if time.time() - T0 > args.wall:
                print(f"[MGEN] wall guard mid-item {idx} at cand {cand}; checkpointed", flush=True)
                json.dump(state, open(STATE_F, "w")); sys.exit(0)
            rec = gen_candidate(row["question"], cand, idx)
            rec["gold"] = gold
            rec_item["cands"].append(rec)
            state["counters"]["tokens"] = state["counters"]["tokens"]  # already counted inline
            json.dump(state, open(STATE_F, "w"))
            print(f"[MGEN] item {idx} cand {cand}: tok={rec['n_tokens']} wall={rec['wall_s']}s "
                  f"pred={rec['pred']} gn={rec['genius_score']:.3f} lp={rec['model_score']:.3f} gold={gold}", flush=True)
        items[str(idx)] = rec_item
        json.dump(state, open(STATE_F, "w"))
else:  # M4
    items = state["items"]
    for idx in EVAL_IDX:
        if str(idx) in items:
            continue
        if time.time() - T0 > args.wall:
            print(f"[M4] wall guard; {len(items)} done", flush=True)
            break
        row = gsm["test"][idx]
        gold = float(state_test[idx]["gold"])
        rec = gen_candidate(row["question"], 0, idx)
        rec["gold"] = gold
        rec["strict"] = bool(rec["pred"] is not None and abs(rec["pred"] - gold) < 1e-6)
        items[str(idx)] = rec
        json.dump(state, open(STATE_F, "w"))
        print(f"[M4] item {idx}: tok={rec['n_tokens']} wall={rec['wall_s']}s strict={rec['strict']} "
              f"pred={rec['pred']} gold={gold} mix_steps={state['counters']['mix_steps']}", flush=True)

# summary
json.dump(state, open(STATE_F, "w"))
c = state["counters"]
print(json.dumps({"mode": MODE, "counters": c,
                  "tok_per_s": round(c["tokens"] / max(1e-9, T0 and (time.time() - T0)), 2)}), flush=True)
