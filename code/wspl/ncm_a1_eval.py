"""WSPL NCM-A1 eval: ANSWER-level parallel generation.
K=4 whole-answer rollouts batched in ONE forward per step; per-rollout genius-logit
mixing alpha in {0.0, 0.5, 1.0, 1.5} (H1 head standardized scores added to the 12
frozen universe logits); per-step capture of chosen-token logprob (unmixed model
logits) + H1 confidence/agreement for genius selection.
Prereg: download/wspl_ncm_a1_prereg.json (frozen, SHA 510c6b79...).
Checkpoint/resume per item. Usage: --wall 420 [--smoke] [--limit N]
"""
import os, sys, json, time, re, hashlib, argparse
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--wall", type=float, default=420.0)
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()
T0 = time.time()

ROOT = "/home/z/my-project"
MD = f"{ROOT}/models/qwen25-05b"
OUT = f"{ROOT}/download"
EVAL_DIR = f"{ROOT}/realbench_cache/ncm_a1_eval"
os.makedirs(EVAL_DIR, exist_ok=True)
torch.set_num_threads(2)

MAX_NEW = 16 if args.smoke else 384
N_EVAL = 2 if args.smoke else 30
K = 4
# frozen per amendment_2: pool row spec
ROW_SPEC = [
    {"mode": "greedy_mix", "alpha": 0.0},    # anchor (no mixing)
    {"mode": "greedy_mix", "alpha": 4.0},    # genius co-writing (z_hat rule, amendment_1)
    {"mode": "sample", "T": 0.8},             # pure-model self-consistency
    {"mode": "sample", "T": 1.0},             # pure-model self-consistency
]
EVAL_IDX = [7 + 43 * k for k in range(30)][:N_EVAL]

def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

# ---- gates ----
model_sha = sha256(f"{MD}/model.safetensors")
assert model_sha == "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe", f"GATE-3 FAIL {model_sha}"
heads_sha = sha256(f"{OUT}/wspl_br1_heads.npz")
assert heads_sha == "d2885e6bf7a73106ff60aaa844c2004cb3415a4d12739f741167e419a205176f", f"GATE-H FAIL {heads_sha}"
print("[GATE] model+heads identity PASS", flush=True)

heads = np.load(f"{OUT}/wspl_br1_heads.npz")
W, B, mu, sd = heads["W"], heads["B"], heads["mu"], heads["sd"]
U_IDS = [int(x) for x in heads["u_ids"]]
U_IDS_A = np.array(U_IDS)
W1, B1 = W[0], B[0]                     # H1 head (next-token digit predictor)

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

# GATE-0 gold alignment
gate0 = []
for i in EVAL_IDX:
    ans = gsm["test"][i]["answer"]
    m = re.search(r"####\s*(-?[0-9][0-9,\.]*)", ans)
    gold_ds = float(m.group(1).replace(",", "")) if m else None
    gate0.append(gold_ds is not None and abs(gold_ds - float(state_test[i]["gold"])) < 1e-6)
assert all(gate0), "GATE-0 FAIL"
print(f"[GATE-0] gold alignment {sum(gate0)}/{N_EVAL} PASS", flush=True)

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

STATE_F = f"{EVAL_DIR}/pool_state.json"
state = json.load(open(STATE_F)) if os.path.exists(STATE_F) else {"items": {}, "counters": {"forwards": 0, "steps": 0, "tokens": 0}}

# alpha=0 anchor ids from BR-1 arm A
A_state = json.load(open(f"{ROOT}/realbench_cache/br1_eval/state_A.json")) if os.path.exists(f"{ROOT}/realbench_cache/br1_eval/state_A.json") else {}
A_ids = {int(k): v["ids"] for k, v in A_state.get("items", {}).items()}

def genius_z(h_vec):
    xs = (h_vec - mu) / sd
    return xs @ W1.T + B1                    # (12,) H1 scores

def run_item(q, item_idx):
    pids = prompt_ids(q)
    B = len(ROW_SPEC)
    L = len(pids)
    # batched prefill: identical prompt rows (disclosed: 4x prefill compute, kept for API simplicity)
    input_ids = torch.tensor([pids] * B, dtype=torch.long)
    attn = torch.ones((B, L), dtype=torch.long)
    cache = DynamicCache()
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn, past_key_values=cache, use_cache=True, output_hidden_states=True)
    state["counters"]["forwards"] += 1
    rows = {"gen": [[] for _ in range(B)], "logprobs": [[] for _ in range(B)],
            "confs": [[] for _ in range(B)], "agreements": [[] for _ in range(B)],
            "done": [False] * B, "eos_at": [None] * B}
    wall0 = time.time()
    cur_h = out.hidden_states[-1][:, -1].float().numpy()          # (B, 896)
    cur_logits = out.logits[:, -1].float().numpy()                # (B, V)
    gens = [torch.Generator().manual_seed(1000003 * int(item_idx) + r) for r in range(B)]  # frozen seeds
    for step in range(MAX_NEW):
        state["counters"]["steps"] += 1
        next_tokens = np.zeros(B, dtype=np.int64)
        for r in range(B):
            spec = ROW_SPEC[r]
            z = genius_z(cur_h[r])                                # (12,)
            conf = float(z.max())
            if rows["done"][r]:
                next_tokens[r] = tok.eos_token_id
                continue
            m = cur_logits[r].max()
            tid = None
            if spec["mode"] == "greedy_mix":
                lg = cur_logits[r].copy()
                if spec["alpha"] > 0:
                    zabs = float(np.abs(z).max())
                    z_hat = (z / zabs) if zabs > 0 else np.zeros_like(z)
                    lg[U_IDS_A] += spec["alpha"] * z_hat          # amendment_1 rule
                tid = int(lg.argmax())
            else:  # sample, pure model
                lgT = torch.tensor(cur_logits[r] / spec["T"], dtype=torch.float32)
                tid = int(torch.multinomial(torch.softmax(lgT, 0), 1, generator=gens[r]).item())
            # capture chosen-token logprob under UNMIXED model logits
            lp = float(cur_logits[r][tid] - (m + np.log(np.exp(cur_logits[r] - m).sum())))
            rows["logprobs"][r].append(lp)
            rows["confs"][r].append(conf)
            if tid in U_IDS:
                rows["agreements"][r].append(1.0 if int(z.argmax()) == U_IDS.index(tid) else 0.0)
            else:
                rows["agreements"][r].append(0.0)
            rows["gen"][r].append(tid)
            if tid == tok.eos_token_id:
                rows["done"][r] = True
                rows["eos_at"][r] = len(rows["gen"][r])
            next_tokens[r] = tid
        # done rows keep emitting EOS (excluded from counts), still fed to the batch
        with torch.no_grad():
            o2 = model(input_ids=torch.tensor(next_tokens, dtype=torch.long).unsqueeze(1),
                       past_key_values=cache, use_cache=True, output_hidden_states=True)
        state["counters"]["forwards"] += 1
        cur_h = o2.hidden_states[-1][:, -1].float().numpy()
        cur_logits = o2.logits[:, -1].float().numpy()
        if all(rows["done"][r] for r in range(B)):
            break
    wall = time.time() - wall0
    for r in range(B):
        n_real = rows["eos_at"][r] if rows["eos_at"][r] is not None else len(rows["gen"][r])
        state["counters"]["tokens"] += n_real
    return rows, wall

items_done = set(int(k) for k in state["items"])
for n, i in enumerate(EVAL_IDX):
    if i in items_done:
        continue
    if time.time() - T0 > args.wall:
        print(f"[a1] wall guard; {len(state['items'])} done", flush=True)
        break
    row = gsm["test"][i]
    gold = float(state_test[i]["gold"])
    t0 = time.time()
    rows, wall_gen = run_item(row["question"], i)
    rec_rows = []
    for r in range(len(ROW_SPEC)):
        eos_at = rows["eos_at"][r]
        n_real = eos_at if eos_at is not None else len(rows["gen"][r])
        gen = rows["gen"][r][:n_real]
        text = tok.decode(gen, skip_special_tokens=False)
        pred = extract_ans(text)
        strict = bool(pred is not None and abs(pred - gold) < 1e-6)
        rec_rows.append({
            "row_spec": ROW_SPEC[r], "n_tokens": n_real, "text": text, "pred": pred, "strict": strict,
            "ids": gen,
            "logprobs": rows["logprobs"][r][:n_real],
            "confs": rows["confs"][r][:n_real],
            "agreements": rows["agreements"][r][:n_real],
        })
    # anchor: alpha=0 row ids vs BR-1 arm A ids
    anchor_match = None
    if i in A_ids:
        anchor_match = (rec_rows[0]["ids"] == A_ids[i])
    state["items"][str(i)] = {"idx": i, "gold": gold, "rows": rec_rows,
                              "wall_gen_s": round(wall_gen, 2),
                              "forwards_item": 0, "anchor_match": anchor_match}
    json.dump(state, open(STATE_F, "w"))
    preds = [rr["pred"] for rr in rec_rows]
    stricts = [rr["strict"] for rr in rec_rows]
    print(f"[a1] item {i}: wall={wall_gen:.1f}s strict_per_alpha={stricts} preds={preds} anchor={anchor_match}", flush=True)

n = len(state["items"])
if n:
    print(f"[a1] done {n}/{len(EVAL_IDX)} tokens={state['counters']['tokens']} forwards={state['counters']['forwards']}", flush=True)
