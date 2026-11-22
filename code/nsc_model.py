"""NSC-1 v2 model: GateBlock transformer, causal, KV-cache, meta/type heads.
As-built param counts (verified):
  (D=256, L=4)  -> 3,197,966   (committed est. 3,197,964, +2)
  (D=768, L=14) -> 99,408,398  (committed est. 99,408,396, +2)
  (D=1536,L=16) -> 453,685,262 (committed est. 453,685,260, +2)
"""
import math, torch, torch.nn as nn, torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nsc_world import VOCAB, MAXLEN, PAD, CLS, SEP, EOS, SIG_IDS, N_SID, enc, ITOS

N_META, N_TYPE = 6, 8


def sin_pos(T, D, device, dtype, off=0):
    pe = torch.zeros(T, D, device=device, dtype=dtype)
    pos = torch.arange(off, off + T, device=device, dtype=dtype).unsqueeze(1)
    div = torch.exp(torch.arange(0, D, 2, device=device, dtype=dtype) * (-math.log(10000.0) / D))
    pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
    return pe


class Attn(nn.Module):
    def __init__(self, D, nh):
        super().__init__()
        self.nh, self.Dh = nh, D // nh
        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.o = nn.Linear(D, D, bias=False)
        self.qn = _RMS(D)
        self.kn = _RMS(D)

    def forward(self, x, kv=None):
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = self.qn(q).view(B, T, self.nh, self.Dh).transpose(1, 2)
        k = self.kn(k).view(B, T, self.nh, self.Dh).transpose(1, 2)
        v = v.view(B, T, self.nh, self.Dh).transpose(1, 2)
        if kv is None:
            K, V = k, v
        else:
            K = torch.cat([kv[0], k], dim=2); V = torch.cat([kv[1], v], dim=2)
        newkv = (K, V)
        if kv is None:
            # training/full path: memory-efficient fused kernel (mathematically identical
            # to q@K^T/sqrt(dh) + causal mask + softmax + @V; is_causal == triu(1) mask)
            y = torch.nn.functional.scaled_dot_product_attention(q, K, V, is_causal=True)
        else:
            att = (q @ K.transpose(-2, -1)) / math.sqrt(self.Dh)
            att = att.softmax(-1)  # single query step attends to all cache
            y = (att @ V)
        return self.o(y.transpose(1, 2).reshape(B, T, D)), newkv


class _RMS(nn.Module):
    def __init__(self, d):
        super().__init__(); self.w = nn.Parameter(torch.ones(d))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.w


class Block(nn.Module):
    def __init__(self, D, nh):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.ln3, self.ln4 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.attn = Attn(D, nh)
        self.mlp = nn.Sequential(nn.Linear(D, 4 * D, bias=False), nn.GELU(),
                                 nn.Linear(4 * D, D, bias=False))
        self.gate = nn.Parameter(torch.zeros(D))
        self.status_emb = nn.Embedding(N_SID, D)

    def forward(self, x, sid, kv=None):
        sid = sid.expand(x.shape[0]) if sid.dim() == 0 else sid
        g = self.gate.to(x.dtype) * torch.tanh(self.status_emb(sid).to(x.dtype)).unsqueeze(1)
        x = x + g
        h = self.ln1(x)
        a, newkv = self.attn(h, kv)
        x = x + self.ln2(a)
        m = self.ln3(x)
        x = x + self.ln4(self.mlp(m))
        return x, newkv


class NSCModel(nn.Module):
    def __init__(self, D=256, L=4, nh=4, V=VOCAB):
        super().__init__()
        self.D, self.L, self.nh = D, L, nh
        self.tok = nn.Embedding(V, D)
        self.blocks = nn.ModuleList([Block(D, nh) for _ in range(L)])
        self.meta_head = nn.Linear(D, N_META)
        self.type_head = nn.Linear(D, N_TYPE)
        self.lm_head = nn.Linear(D, V, bias=False)
        self.lm_head.weight = self.tok.weight

    def forward(self, ids, sid, kv=None, pos_off=0):
        x = self.tok(ids)
        T = ids.shape[1]
        x = x + sin_pos(T, self.D, ids.device, x.dtype, pos_off).unsqueeze(0)
        kvs = []
        for i, b in enumerate(self.blocks):
            x, kv_i = b(x, sid, None if kv is None else kv[i])
            kvs.append(kv_i)
        cls = x[:, 0]
        return self.lm_head(x), self.meta_head(cls), self.type_head(cls), kvs

    @torch.no_grad()
    def generate(self, inp_ids, sid, max_new=40, gates_on=True):
        backup = None
        if not gates_on:
            backup = [b.gate.detach().clone() for b in self.blocks]
            for b in self.blocks: b.gate.zero_()
        dev = next(self.parameters()).device
        ids = torch.tensor([inp_ids], device=dev)
        s = torch.tensor([sid], device=dev)
        logits, _, _, kv = self.forward(ids, s)
        out = []
        for _ in range(max_new):
            nxt = int(logits[0, -1].argmax())
            if nxt == EOS: break
            out.append(nxt)
            logits, _, _, kv = self.forward(torch.tensor([[nxt]], device=dev), s, kv=kv, pos_off=kv[0][0].shape[2])
        if backup is not None:
            for b, g in zip(self.blocks, backup): b.gate.data.copy_(g)
        return out

    def zero_gates(self):
        return [b.gate.detach().clone() for b in self.blocks]

    def restore_gates(self, backup):
        for b, g in zip(self.blocks, backup): b.gate.data.copy_(g)


def param_count(m):
    return sum(p.numel() for p in m.parameters())


# ---------------- sequence building ----------------
SIG_BY_NAME = SIG_IDS  # name -> id
NAME_BY_SIG = {v: k for k, v in SIG_IDS.items()}
# meta head class = sid class (0..9); sig ids 3..8 map to meta classes 3..8 directly


def build_seq(inp, sig_name, resp, mx=MAXLEN):
    """ids = enc(inp)[-mx:] + [SEP, SIG] + enc(resp) + [EOS].
    Loss positions: from SEP (predicts SIG) through last resp char (predicts EOS)."""
    in_ids = enc(inp.lower())[-(mx - len(resp) - 3):]
    sig_id = SIG_IDS.get(sig_name) or SIG_IDS["<s_" + sig_name + ">"]
    resp_ids = enc(resp.lower())
    ids = in_ids + [SEP, sig_id] + resp_ids + [EOS]
    T = len(ids)
    pad = mx - T
    ids = ids + [PAD] * pad
    lab = [PAD] * mx
    sep_i = len(in_ids)
    for t in range(sep_i, T - 1):
        lab[t] = ids[t + 1]
    return ids, lab, sep_i


def loss_fn(logits, meta, typ, lab, sep_meta, sep_type, dev):
    lm = F.cross_entropy(logits.reshape(-1, VOCAB).float(), lab.reshape(-1).to(dev), ignore_index=PAD)
    mt = F.cross_entropy(meta.float(), sep_meta.to(dev))
    ty = F.cross_entropy(typ.float(), sep_type.to(dev))
    return lm + 0.3 * mt + 0.1 * ty, lm.detach(), mt.detach(), ty.detach()


def decode(out_ids):
    return "".join(ITOS[i] if i < len(ITOS) else "?" for i in out_ids)


if __name__ == "__main__":
    for D, L, target in [(256, 4, 3197964), (768, 14, 99408396), (1536, 16, 453685260)]:
        m = NSCModel(D=D, L=L, nh=D // 64)
        n = param_count(m)
        print(f"D={D} L={L}: {n} (target {target}, diff {n - target})")
        del m
    # smoke: forward/backward/generate + kv equality
    torch.manual_seed(9)
    m = NSCModel(D=256, L=4, nh=4)
    ids, lab, sep_i = build_seq("what color is the kefume?", "<s_answer>", "the kefume is blue.")
    t = torch.tensor([ids]); s = torch.tensor([3])
    lg, mt, ty, _ = m(t, s)
    print("smoke logits", tuple(lg.shape), "meta", tuple(mt.shape), "type", tuple(ty.shape))
    g = m.generate(enc("what color is the kefume?".lower()), 3, max_new=8)
    print("gen", repr(decode(g)))
    # KV-cache equality proof: generated token sequences must be IDENTICAL
    import random as _r

    @torch.no_grad()
    def gen_full(inp, sid, max_new=20):
        out = []
        cur = list(inp)
        s = torch.tensor([sid])
        for _ in range(max_new):
            logits, _, _, _ = m(torch.tensor([cur]), s)
            nxt = int(logits[0, -1].argmax())
            if nxt == EOS: break
            out.append(nxt); cur.append(nxt)
        return out

    _r.seed(1)
    neq = 0
    for trial in range(6):
        inp = [int(x) for x in torch.randint(11, 100, (12,)).tolist()]
        sid = 3 + trial % 6
        a = gen_full(inp, sid)
        b = m.generate(inp, sid, max_new=20)
        eq = (a == b)
        neq += eq
        print(f"KV-EQ trial {trial}: {'IDENTICAL' if eq else 'DIFFER'} (len {len(a)})")
    print(f"KV-EQ total: {neq}/6 identical")
