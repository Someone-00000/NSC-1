"""BR-2 shared: mixed-mind hook (genius H1 mixed into model output weights) + prompt utils.
Prereg: download/wspl_br2_prereg.json (frozen, SHA 8f2bd191...).
The hook adds alpha * z1 for the 12 universe tokens to lm_head output every call,
mathematically identical to patching lm_head rows W_t += alpha*W1_t/sd (tie_word_embeddings=True
forbids in-place row patching without corrupting input digit embeddings; disclosed in prereg).
"""
import re
import numpy as np
import torch

ROOT = "/home/z/my-project"
OUT = f"{ROOT}/download"
U_IDS = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 198, 151645]


def load_genius_h1():
    heads = np.load(f"{OUT}/wspl_ncm_c1_heads.npz")
    W1 = heads["W"][0].astype(np.float32)   # (12, 896)
    b1 = heads["B"][0].astype(np.float32)   # (12,)
    mu = heads["mu"].astype(np.float32)     # (896,)
    sd = heads["sd"].astype(np.float32)     # (896,)
    return W1, b1, mu, sd


def register_mixed_mind(model, alpha_vec):
    """Register forward hook on model.lm_head. alpha_vec: torch tensor (B,) per-sequence alpha.
    Returns the hook handle (keep to remove)."""
    W1, b1, mu, sd = load_genius_h1()
    dev = model.lm_head.weight.device
    W1_t = torch.tensor(W1, device=dev); b1_t = torch.tensor(b1, device=dev)
    mu_t = torch.tensor(mu, device=dev); sd_t = torch.tensor(sd, device=dev)
    U_t = torch.tensor(U_IDS, device=dev, dtype=torch.long)
    av = torch.tensor(alpha_vec, dtype=torch.float32, device=dev).view(-1, 1)  # (B,1)

    def hook(module, inputs, output):
        h = inputs[0]
        h2 = h.reshape(-1, h.shape[-1]).float()          # (N, 896)
        z1 = ((h2 - mu_t) / sd_t) @ W1_t.T + b1_t        # (N, 12)
        n = h2.shape[0]
        rep = n // av.shape[0]
        a = av.repeat(rep, 1) if rep > 1 else av         # (N, 1)
        out2 = output.reshape(n, -1)
        out2[:, U_t] += a * z1
        return output

    return model.lm_head.register_forward_hook(hook)


def build_fewshot(tok, gsm):
    _r0, _r1 = gsm["train"][0], gsm["train"][1]
    FEWSHOT = [( _r0["question"], re.sub(r"<<[^>]*>>", "", _r0["answer"]).strip()),
               (_r1["question"], re.sub(r"<<[^>]*>>", "", _r1["answer"]).strip())]
    return FEWSHOT


def user_content(FEWSHOT, q):
    e1, e2 = FEWSHOT
    return (f"Here are two worked examples.\n\nExample 1:\nQuestion: {e1[0].strip()}\nAnswer: {e1[1]}\n\n"
            f"Example 2:\nQuestion: {e2[0].strip()}\nAnswer: {e2[1]}\n\n"
            f"Now solve.\nQuestion: {q.strip()}\nAnswer: Let's think step by step, and end with \"#### <final numeric answer>\".")


def prompt_ids(tok, FEWSHOT, q):
    msgs = [{"role": "user", "content": user_content(FEWSHOT, q)}]
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
