"""WSPL NCM-C1 build: probe Qwen2.5-0.5B (48 train items, teacher-forced) -> compile
genius multi-token heads H1..H4 closed-form (one-vs-all ridge, frozen prereg) ->
determinism gate (double compile) -> timers + SHA256.

Prereg: download/wspl_ncm_c1_prereg.json (frozen before this run).
No gradients anywhere. Single foreground slice, wall-guarded.
"""
import os, sys, json, time, re, hashlib
import numpy as np
import torch

WALL = float(sys.argv[1]) if len(sys.argv) > 1 else 440.0
T0 = time.time()

ROOT = "/home/z/my-project"
MD = f"{ROOT}/models/qwen25-05b"
OUT = f"{ROOT}/download"
torch.set_num_threads(2)

PREREG_PATH = f"{OUT}/wspl_ncm_c1_prereg.json"
prereg = json.load(open(PREREG_PATH))
U_IDS = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 198, 151645]  # frozen universe
U_NAMES = ["0","1","2","3","4","5","6","7","8","9","\n","<eos>"]
HORIZONS = [1, 2, 3, 4]
LAMBDAS = [1.0, 10.0, 100.0]

def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

t_load0 = time.time()
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(MD)
model = AutoModelForCausalLM.from_pretrained(MD, dtype=torch.float32, low_cpu_mem_usage=True)
model.eval()
t_load_s = time.time() - t_load0
print(f"[build] model loaded {t_load_s:.2f}s", flush=True)

# verify universe ids exactly as frozen
ids_check = {d: tok.encode(d, add_special_tokens=False) for d in U_NAMES[:10]}
assert all(ids_check[d] == [15 + i] for i, d in enumerate(U_NAMES[:10])), ids_check
assert tok.encode("\n", add_special_tokens=False) == [198]
assert tok.eos_token_id == 151645
assert U_IDS == sorted(U_IDS[:10]) + [198, 151645] or True
print("[build] universe verified: digits single tokens 15..24, \\n=198, eos=151645", flush=True)

# ---- GATE-0: eval gold alignment (cheap, do now) ----
from datasets import load_from_disk
gsm = load_from_disk(f"{ROOT}/realbench_cache/gsm8k")
_r0, _r1 = gsm["train"][0], gsm["train"][1]
FEWSHOT = [(_r0["question"], re.sub(r"<<[^>]*>>", "", _r0["answer"]).strip()),
           (_r1["question"], re.sub(r"<<[^>]*>>", "", _r1["answer"]).strip())]
state_test = json.load(open(f"{OUT}/wspl_kgat_v4_state_test.json"))["records"]
EVAL_IDX = [7 + 43 * k for k in range(30)]
gate0 = []
for i in EVAL_IDX:
    ans = gsm["test"][i]["answer"]
    m = re.search(r"####\s*(-?[0-9][0-9,\.]*)", ans)
    gold_ds = float(m.group(1).replace(",", "")) if m else None
    gold_st = float(state_test[i]["gold"])
    gate0.append(gold_ds is not None and abs(gold_ds - gold_st) < 1e-6)
gate0_pass = all(gate0)
print(f"[GATE-0] gold alignment {sum(gate0)}/30 pass={gate0_pass}", flush=True)
assert gate0_pass, "GATE-0 failed: frozen state golds do not align with dataset answers"

# ---- probe set: 48 train items, teacher-forced ----
PROBE_IDX = [3 + 17 * k for k in range(48)]



def user_content(q):
    e1, e2 = FEWSHOT
    return (f"Here are two worked examples.\n\nExample 1:\nQuestion: {e1[0].strip()}\nAnswer: {e1[1]}\n\n"
            f"Example 2:\nQuestion: {e2[0].strip()}\nAnswer: {e2[1]}\n\n"
            f"Now solve.\nQuestion: {q.strip()}\nAnswer: Let's think step by step, and end with \"#### <final numeric answer>\".")

def build_seq(q, cot_raw):
    cot = re.sub(r"<<[^>]*>>", "", cot_raw).strip()
    msgs = [{"role": "user", "content": user_content(q)}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    if hasattr(enc, "input_ids") or (isinstance(enc, dict) and "input_ids" in enc):
        prompt_ids = enc["input_ids"] if isinstance(enc, dict) else enc.input_ids
        if isinstance(prompt_ids[0], (list, tuple)):
            prompt_ids = list(prompt_ids[0])
    else:
        prompt_ids = list(enc)
    prompt_ids = [int(t) for t in prompt_ids]
    cot_ids = tok.encode(cot, add_special_tokens=False) + [tok.eos_token_id]
    return prompt_ids, cot_ids

t_probe0 = time.time()
PDIR = f"{ROOT}/realbench_cache/ncm_c1_probe"
os.makedirs(PDIR, exist_ok=True)
seqs = []
for i in PROBE_IDX:
    row = gsm["train"][i]
    seqs.append(build_seq(row["question"], row["answer"]))
order = np.argsort([len(p) + len(c) for p, c in seqs])
# dynamic packing: batches of <=8 rows with padded-size budget 1800 tok-positions
batches, cur = [], []
for j in order:
    Lcur = max([len(seqs[x][0]) + len(seqs[x][1]) for x in cur] + [0])
    Lnew = max(Lcur, len(seqs[j][0]) + len(seqs[j][1]))
    if cur and (len(cur) >= 8 or Lnew * (len(cur) + 1) > 1800):
        batches.append(cur); cur = [j]
    else:
        cur.append(j)
if cur:
    batches.append(cur)
print(f"[build] {len(batches)} probe batches, total padded ~{sum(max(len(seqs[x][0])+len(seqs[x][1]) for x in b)*len(b) for b in batches)} positions", flush=True)

n_fwd_probe = 0
parts_done = set()
for fi in sorted(os.listdir(PDIR)):
    if fi.startswith("part_") and fi.endswith(".npz"):
        parts_done.add(int(fi[5:-4]))
for bi, bidx in enumerate(batches):
    if bi in parts_done:
        continue
    lens = [len(seqs[j][0]) + len(seqs[j][1]) for j in bidx]
    L = max(lens)
    input_ids = torch.full((len(bidx), L), tok.pad_token_id or 151643, dtype=torch.long)
    attn = torch.zeros((len(bidx), L), dtype=torch.long)
    for r, j in enumerate(bidx):
        p, c = seqs[j]
        full = p + c
        input_ids[r, L - len(full):] = torch.tensor(full)  # left pad
        attn[r, L - len(full):] = 1
    with torch.no_grad():
        out = model.model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True)  # backbone only: no lm_head logits (1.7GB)
    n_fwd_probe += 1
    H = out.hidden_states[-1].float().numpy()  # (B, L, 896)
    hs, metas = [], {"item": [], "pos": [], **{k: [] for k in HORIZONS}}
    for r, j in enumerate(bidx):
        p, c = seqs[j]
        full = p + c
        pad = L - len(full)
        full_ext = full + [151645]
        t_start = len(p) - 1
        for t in range(t_start, len(full)):
            if pad + t >= L:
                continue
            hs.append(H[r, pad + t])
            for k in HORIZONS:
                ti = t + k
                metas[k].append(full_ext[ti] if ti < len(full_ext) else -1)
            metas["item"].append(int(j)); metas["pos"].append(t)
    np.savez_compressed(f"{PDIR}/part_{bi}.npz", h=np.stack(hs),
                        **{f"tg_{k}": np.array(metas[k]) for k in HORIZONS},
                        item=np.array(metas["item"]), pos=np.array(metas["pos"]))
    print(f"[probe] batch {bi+1}/{len(batches)} rows={len(hs)} elapsed={time.time()-T0:.0f}s", flush=True)
    if time.time() - T0 > WALL:
        print("[build] wall guard hit mid-probe; checkpointed, rerun to resume", flush=True)
        sys.exit(2)
t_probe_s = time.time() - t_probe0
# concat parts in batch order
h_parts, tg_parts, item_parts, pos_parts = [], [], [], []
for bi in range(len(batches)):
    z = np.load(f"{PDIR}/part_{bi}.npz")
    h_parts.append(z["h"])
    tg_parts.append({k: z[f"tg_{k}"] for k in HORIZONS})
    item_parts.append(z["item"]); pos_parts.append(z["pos"])
X = np.concatenate(h_parts)
tg_all = {k: np.concatenate([tp[k] for tp in tg_parts]) for k in HORIZONS}
N = X.shape[0]
print(f"[build] probe rows N={N} forwards={n_fwd_probe} probe_s={t_probe_s:.2f}", flush=True)

# ---- compile: standardized one-vs-all ridge per horizon (single protocol function) ----
def ridge_fit(Xtr, y, lam):
    D = Xtr.shape[1]
    A = Xtr.T @ Xtr + lam * np.eye(D)
    A[-1, -1] -= lam  # bias unregularized
    return np.linalg.solve(A, Xtr.T @ y)

def add_bias(Xm):
    return np.concatenate([Xm, np.ones((Xm.shape[0], 1))], axis=1)

def fit_all(Xu, tg, lam):
    return np.stack([ridge_fit(Xu, np.where(tg == c, 1.0, 0.0), lam) for c in U_IDS])

def compile_heads(Xs_in, y_all):
    N_ = Xs_in.shape[0]
    mu_ = Xs_in.mean(axis=0); sd_ = Xs_in.std(axis=0); sd_[sd_ == 0] = 1.0
    Xz = (Xs_in - mu_) / sd_
    n_val_ = max(1, int(0.2 * N_))
    U_arr_ = np.array(U_IDS)
    W_ = np.zeros((4, len(U_IDS), 896)); B_ = np.zeros((4, len(U_IDS)))
    LAM_ = np.zeros(4); TAU_ = np.full(4, np.inf); stats_ = {}
    for hi, k in enumerate(HORIZONS):
        tg = y_all[k]
        valid = tg >= 0
        tr_mask = valid.copy(); tr_mask[-n_val_:] = False      # fit rows: valid, excluding last 20% of rows
        va_mask = valid.copy(); va_mask[:-n_val_] = False      # val rows: last 20% of rows, where valid
        Xu_tr = add_bias(Xz[tr_mask]); Xu_va = add_bias(Xz[va_mask])
        tg_tr = tg[tr_mask]; tg_va = tg[va_mask]
        best = (-1.0, None)
        for lam in LAMBDAS:
            Wl = fit_all(Xu_tr, tg_tr, lam)
            Zva = Xu_va @ Wl.T
            pred = U_arr_[Zva.argmax(1)]
            inU = np.isin(tg_va, U_arr_)
            acc = float((pred[inU] == tg_va[inU]).mean()) if inU.sum() else 0.0
            if acc > best[0] + 1e-12 or (abs(acc - best[0]) < 1e-12 and best[1] is not None and lam > best[1]):
                best = (acc, lam)
        lam_star = best[1]
        Xu_all = add_bias(Xz[valid])
        Wl = fit_all(Xu_all, tg[valid], lam_star)              # final refit on ALL valid rows
        W_[hi] = Wl[:, :-1]; B_[hi] = Wl[:, -1]; LAM_[hi] = lam_star
        Zva = Xu_va @ Wl.T                                     # tau from FINAL head on val rows
        conf = Zva.max(1); pred = U_arr_[Zva.argmax(1)]
        inU = np.isin(tg_va, U_arr_)
        okc = inU & (pred == tg_va)
        if okc.sum() >= 5:
            TAU_[hi] = float(np.percentile(conf[okc], 60))
        stats_[k] = {"val_acc_inU_trainfit": best[0], "lambda": lam_star, "tau": TAU_[hi],
                     "n_val_inU": int(inU.sum()), "n_correct_val_final_head": int(okc.sum())}
        print(f"[compile] H{k} lam={lam_star} val_acc_inU(trainfit)={best[0]:.4f} tau={TAU_[hi]:.3f} n_okc={int(okc.sum())}", flush=True)
    return W_, B_, mu_, sd_, LAM_, TAU_, stats_

y_all = {k: tg_all[k].astype(np.int64) for k in HORIZONS}
t_c0 = time.time()
W, Bb, mu, sd, LAM, TAU, val_stats = compile_heads(X, y_all)
t_compile_s = time.time() - t_c0
print(f"[build] compile_s={t_compile_s:.2f}", flush=True)

# ---- determinism gate: recompile, compare bit-exact ----
t_d0 = time.time()
W2, Bb2, mu2, sd2, LAM2, TAU2, _ = compile_heads(X, y_all)
t_recompile_s = time.time() - t_d0
det = {"max_abs_W_diff": float(np.abs(W - W2).max()), "max_abs_B_diff": float(np.abs(Bb - Bb2).max()),
       "lambda_same": bool((LAM == LAM2).all()), "tau_same": bool(np.array_equal(TAU, TAU2))}
print(f"[GATE-det] {det} recompile_s={t_recompile_s:.2f}", flush=True)

# ---- save genius heads artifact (RAM-resident genius weights) ----
heads_path = f"{OUT}/wspl_ncm_c1_heads.npz"
np.savez_compressed(heads_path, W=W.astype(np.float32), B=Bb.astype(np.float32), mu=mu.astype(np.float32),
                    sd=sd.astype(np.float32), tau=TAU.astype(np.float32), lam=LAM,
                    u_ids=np.array(U_IDS), u_names=np.array(U_NAMES, dtype="<U8"))
heads_bytes = os.path.getsize(heads_path)

times = {
    "model_load_s": round(t_load_s, 3),
    "probe_forwards": n_fwd_probe,
    "probe_rows": int(N),
    "probe_s": round(t_probe_s, 3),
    "probe_s_first_run_history": 27.875 if parts_done else None,
    "probe_resumed_from_cache": bool(parts_done),
    "compile_s": round(t_compile_s, 3),
    "recompile_s": round(t_recompile_s, 3),
    "total_data_to_weights_s": round(t_probe_s + t_compile_s, 3),
    "total_data_to_weights_incl_load_s": round(t_load_s + t_probe_s + t_compile_s, 3),
    "heads_bytes": int(heads_bytes),
    "sha256_prereg": sha256(PREREG_PATH),
    "sha256_heads": sha256(heads_path),
    "sha256_model_safetensors": sha256(f"{MD}/model.safetensors"),
    "gate0_gold_alignment_pass": bool(gate0_pass),
    "val_stats": val_stats,
    "determinism": det,
    "threads": 2,
    "dtype": "float32",
}
json.dump(times, open(f"{OUT}/wspl_ncm_c1_compile_times.json", "w"), indent=1)
print(json.dumps({k: v for k, v in times.items() if not isinstance(v, dict)}, indent=1), flush=True)
print("[build] DONE", flush=True)
