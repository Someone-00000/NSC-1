"""BR-2 diversity probe (diagnostic, disclosed): does alpha steering actually diverge
greedy trajectories? Test one eval-family prompt with alpha in {0,1,2,4,8} and diff streams."""
import os, sys, json, time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from br2_common import ROOT, register_mixed_mind, build_fewshot, prompt_ids

torch.set_num_threads(2)
MD = f"{ROOT}/models/qwen25-05b"
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from datasets import load_from_disk
tok = AutoTokenizer.from_pretrained(MD)
model = AutoModelForCausalLM.from_pretrained(MD, dtype=torch.float32, low_cpu_mem_usage=True)
model.eval()
gsm = load_from_disk(f"{ROOT}/realbench_cache/gsm8k")
FEWSHOT = build_fewshot(tok, gsm)
i = 7  # first frozen eval item
pids = prompt_ids(tok, FEWSHOT, gsm["test"][i]["question"])

def gen(alpha, max_new=120):
    handle = register_mixed_mind(model, [alpha]) if alpha > 0 else None
    cache = DynamicCache()
    try:
        with torch.no_grad():
            hb = model.model(input_ids=torch.tensor([pids], dtype=torch.long), past_key_values=cache, use_cache=True)
            logits = model.lm_head(hb.last_hidden_state[:, -1:])[0, -1].float()
            gen = []
            for _ in range(max_new):
                nt = int(logits.argmax())
                gen.append(nt)
                if nt == tok.eos_token_id:
                    break
                hb = model.model(input_ids=torch.tensor([[nt]], dtype=torch.long), past_key_values=cache, use_cache=True)
                logits = model.lm_head(hb.last_hidden_state[:, -1:])[0, -1].float()
        return gen
    finally:
        if handle is not None:
            handle.remove()

t0 = time.time()
streams = {}
for a in [0.0, 1.0, 2.0, 4.0, 8.0]:
    streams[a] = gen(a)
base = streams[0.0]
print(f"elapsed {time.time()-t0:.0f}s", flush=True)
for a, s in streams.items():
    first_div = next((p for p, (x, y) in enumerate(zip(s, base)) if x != y), None)
    same = (s == base)
    print(f"alpha={a}: len={len(s)} identical_to_plain={same} first_div={first_div}")
print("decoded plain:", tok.decode(base, skip_special_tokens=False)[:220])
print("decoded a=4:", tok.decode(streams[4.0], skip_special_tokens=False)[:220])
print("decoded a=8:", tok.decode(streams[8.0], skip_special_tokens=False)[:220])
