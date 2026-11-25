"""WSPL BR-1 eval: arms A (AR baseline), B (genius block H1..H8 + verify/reason), C (genius fire, no verify).
Prereg: download/wspl_br1_prereg.json. Greedy fp32 threads=2 KV cache max_new=384.
Checkpoint/resume per item. Usage: --arm A|B|C --wall 420 [--smoke]
Prompt MUST match br1_build.py user_content exactly (digit-dense exemplars).
"""
import os, sys, json, time, re, argparse
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True)
ap.add_argument("--wall", type=float, default=420.0)
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()
ARM = args.arm
T0 = time.time()

ROOT = "/home/z/my-project"
MD = f"{ROOT}/models/qwen25-05b"
OUT = f"{ROOT}/download"
EVAL_DIR = f"{ROOT}/realbench_cache/br1_eval"
os.makedirs(EVAL_DIR, exist_ok=True)
torch.set_num_threads(2)

MAX_NEW = 16 if args.smoke else 384
N_EVAL = 2 if args.smoke else 30
EVAL_IDX = [7 + 43 * k for k in range(30)][:N_EVAL]

heads = np.load(f"{OUT}/wspl_br1_heads.npz")
W, B, mu, sd, TAU, U_IDS = heads["W"], heads["B"], heads["mu"], heads["sd"], heads["tau"], heads["u_ids"]
NH = W.shape[0]  # 8 horizons

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

STATE_F = f"{EVAL_DIR}/state_{ARM}.json"
state = json.load(open(STATE_F)) if os.path.exists(STATE_F) else {"items": {}, "counters": {"forwards": 0, "fires": 0, "proposals": 0, "accepted": 0, "fallbacks": 0, "fire_steps": 0, "steps": 0}}

def load_A_ids():
    f = f"{EVAL_DIR}/state_A.json"
    d = json.load(open(f)) if os.path.exists(f) else {}
    return {int(k): v["ids"] for k, v in d.get("items", {}).items()}
A_ids = load_A_ids() if ARM in ("B", "C") else {}

def genius_propose(h_vec):
    xs = (h_vec - mu) / sd
    prop, conf = [], []
    for hi in range(NH):
        z = xs @ W[hi].T + B[hi]
        conf.append(float(z.max()))
        prop.append(int(U_IDS[int(z.argmax())]))
    m = 0
    for hi in range(NH):
        if conf[hi] >= float(TAU[hi]):
            m = hi + 1
        else:
            break
    return prop, conf, m

def seq_len(cache):
    try:
        return int(cache.get_seq_length())
    except Exception:
        try:
            return int(cache.layers[0].keys.shape[-2])
        except Exception:
            return int(cache.key_cache[0].shape[-2])

def fwd(ids, cache, want_hidden=False):
    t = torch.tensor([ids], dtype=torch.long)
    kw = dict(input_ids=t, past_key_values=cache, use_cache=True)
    if want_hidden:
        kw["output_hidden_states"] = True
    with torch.no_grad():
        out = model(**kw)
    state["counters"]["forwards"] += 1
    logits = out.logits[0, -1].float()
    h = None
    if want_hidden:
        h = out.hidden_states[-1][0, -1].float().numpy()
    return logits, h

def crop_cache(cache, remove_n):
    cache.crop(-int(remove_n))
    return cache

def run_item(q):
    pids = prompt_ids(q)
    cache = DynamicCache()
    kw = dict(input_ids=torch.tensor([pids], dtype=torch.long), past_key_values=cache, use_cache=True, output_hidden_states=True)
    with torch.no_grad():
        out = model(**kw)
    state["counters"]["forwards"] += 1
    cur_logits = out.logits[0, -1].float()
    cur_h = out.hidden_states[-1][0, -1].float().numpy()
    gen, events = [], {"fires": 0, "proposals": 0, "accepted": 0, "fallbacks": 0, "fire_steps": 0, "steps": 0}
    wall0 = time.time()
    while len(gen) < MAX_NEW:
        events["steps"] += 1
        base_next = int(cur_logits.argmax())
        if ARM == "A":
            if base_next == tok.eos_token_id:
                gen.append(base_next)
                break
            lg, _ = fwd([base_next], cache)
            gen.append(base_next)
            cur_logits = lg
            continue
        prop, conf, m = genius_propose(cur_h)
        if ARM == "B":
            if m == 0:
                if base_next == tok.eos_token_id:
                    gen.append(base_next); break
                lg, hh = fwd([base_next], cache, want_hidden=True)
                gen.append(base_next); cur_logits, cur_h = lg, hh
                events["fallbacks"] += 1
                continue
            events["fire_steps"] += 1; events["fires"] += m; events["proposals"] += m
            block = prop[:m]
            with torch.no_grad():
                vf = model(input_ids=torch.tensor([block], dtype=torch.long), past_key_values=cache, use_cache=True)
            state["counters"]["forwards"] += 1
            vlog = vf.logits[0].float()  # (m, V); vlog[i] predicts token after block[i]
            a = 0
            if int(cur_logits.argmax()) == block[0]:
                a = 1
                for i in range(m - 1):
                    if int(vlog[i].argmax()) == block[i + 1]:
                        a += 1
                    else:
                        break
            if a == 0:
                crop_cache(cache, m)
                lg, hh = fwd([base_next], cache, want_hidden=True)
                gen.append(base_next); cur_logits, cur_h = lg, hh
                events["fallbacks"] += 1
            else:
                if a < m:
                    crop_cache(cache, m - a)
                gen.extend(block[:a])
                events["accepted"] += a
                bonus = int(vlog[a - 1].argmax())
                lg, hh = fwd([bonus], cache, want_hidden=True)
                gen.append(bonus); cur_logits, cur_h = lg, hh
                if bonus == tok.eos_token_id:
                    break
        elif ARM == "C":
            if m == 0:
                if base_next == tok.eos_token_id:
                    gen.append(base_next); break
                lg, hh = fwd([base_next], cache, want_hidden=True)
                gen.append(base_next); cur_logits, cur_h = lg, hh
                events["fallbacks"] += 1
                continue
            events["fire_steps"] += 1; events["fires"] += m; events["proposals"] += m
            block = prop[:m]
            lg, hh = fwd(block, cache, want_hidden=True)
            gen.extend(block)
            cur_logits, cur_h = lg, hh
            if block[-1] == tok.eos_token_id:
                break
    wall = time.time() - wall0
    return gen, wall, events

items_done = set(int(k) for k in state["items"])
for n, i in enumerate(EVAL_IDX):
    if i in items_done:
        continue
    if time.time() - T0 > args.wall:
        print(f"[eval {ARM}] wall guard; {len(state['items'])} done", flush=True)
        break
    row = gsm["test"][i]
    gold = float(state_test[i]["gold"])
    t_item0 = time.time()
    fw_before = state["counters"]["forwards"]
    gen, wall_gen, events = run_item(row["question"])
    gen = gen[:MAX_NEW]
    fw_item = state["counters"]["forwards"] - fw_before
    text = tok.decode(gen, skip_special_tokens=False)
    pred = extract_ans(text)
    strict = bool(pred is not None and abs(pred - gold) < 1e-6)
    ids = gen
    first_div = None
    if ARM in ("B", "C") and i in A_ids:
        a_ids = A_ids[i]
        first_div = next((p for p, (x, y) in enumerate(zip(ids, a_ids)) if x != y), min(len(ids), len(a_ids)) if len(ids) != len(a_ids) else None)
        if first_div is None and len(ids) == len(a_ids):
            first_div = -1  # identical
    rec = {"idx": i, "gold": gold, "pred": pred, "strict": strict, "n_tokens": len(gen),
           "wall_gen_s": round(wall_gen, 2), "wall_item_s": round(time.time() - t_item0, 2),
           "forwards": fw_item, "events": events,
           "ids": ids, "text": text, "first_divergence_vs_A": first_div}
    state["items"][str(i)] = rec
    for kk in ("fires", "proposals", "accepted", "fallbacks", "fire_steps", "steps"):
        state["counters"][kk] += events[kk]
    json.dump(state, open(STATE_F, "w"))
    print(f"[eval {ARM}] item {i}: tok={len(gen)} wall={wall_gen:.1f}s strict={strict} pred={pred} gold={gold}", flush=True)

done = state["items"]
n = len(done)
if n:
    stricts = [v["strict"] for v in done.values()]
    toks = sum(v["n_tokens"] for v in done.values())
    walls = sum(v["wall_gen_s"] for v in done.values())
    summ = {"arm": ARM, "n_done": n, "n_total": len(EVAL_IDX), "strict_acc": sum(stricts) / n,
            "tokens": toks, "wall_gen_s": round(walls, 1), "tok_per_s": round(toks / walls, 2),
            "counters": {k: state["counters"][k] for k in state["counters"]}}
    print(json.dumps(summ), flush=True)
    json.dump(summ, open(f"{EVAL_DIR}/summary_{ARM}.json", "w"))
