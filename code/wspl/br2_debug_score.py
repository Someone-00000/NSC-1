"""Debug BR-2 scoring: compare genius-H1 predictions vs actual draft tokens position by position."""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from br2_common import ROOT, OUT, U_IDS, build_fewshot, prompt_ids

torch.set_num_threads(2)
MD = f"{ROOT}/models/qwen25-05b"
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(MD)
model = AutoModelForCausalLM.from_pretrained(MD, dtype=torch.float32, low_cpu_mem_usage=True)
model.eval()
gsm = load_ds = None
from datasets import load_from_disk
gsm = load_from_disk(f"{ROOT}/realbench_cache/gsm8k")
FEWSHOT = build_fewshot(tok, gsm)

heads = np.load(f"{OUT}/wspl_ncm_c1_heads.npz")
W1 = torch.tensor(heads["W"][0].astype(np.float32))
b1 = torch.tensor(heads["B"][0].astype(np.float32))
mu = torch.tensor(heads["mu"].astype(np.float32))
sd = torch.tensor(heads["sd"].astype(np.float32))
print("W1 shape", W1.shape, "tau(H1)=", float(heads["tau"][0]))

U_SET = set(U_IDS)
a2 = json.load(open(f"{ROOT}/realbench_cache/br2_eval/state_A2.json"))
anchor_ids = a2["items"]["7"]["ids"][:120]
q = gsm["test"][7]["question"]
pids = prompt_ids(tok, FEWSHOT, q)
full = pids + anchor_ids
print("gen tokens:", len(anchor_ids), "first ids:", anchor_ids[:20])

with torch.no_grad():
    hb = model.model(input_ids=torch.tensor([full], dtype=torch.long), output_hidden_states=True)
H = hb.last_hidden_state[0].float()
print("H shape:", H.shape)

t_start = len(pids) - 1
n_checked = 0
for t in range(t_start, t_start + 80):
    nxt = full[t + 1]
    if nxt not in U_SET:
        continue
    z1 = ((H[t] - mu) / sd) @ W1.T + b1
    top = torch.topk(z1, 3)
    am = int(top.indices[0]); conf = float(top.values[0])
    n_checked += 1
    if n_checked <= 15:
        print(f"t={t}: next={nxt} ({tok.decode([nxt])!r}) | H1 argmax={am} ({tok.decode([U_IDS[am]])!r}) conf={conf:.2f} top3={[int(i) for i in top.indices]}")
print("in-U positions in window:", n_checked)
