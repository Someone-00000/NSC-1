"""Stage A subconscious: 2 blocks D=128 nh=4, type/meta/dec heads, char-code input."""
import torch, torch.nn as nn, torch.nn.functional as F, math, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nsc_world import enc_a, SIG_IDS

MAXA = 240
META_CLASSES = ["answer", "dontknow", "retrieve", "partial", "refuse", "hold"]  # 6
TYPE_CLASSES = ["fact", "compose", "unk", "trap", "math", "meta", "mixed", "other"]  # 8

def type_of(kind):
    if kind in ("color", "place"): return "fact"
    if kind == "compose": return "compose"
    if kind in ("near", "unknown", "meta_u"): return "unk"
    if kind in ("false",) or kind.startswith("syc_"): return "trap"
    if kind == "math": return "math"
    if kind == "meta_k": return "meta"
    if kind == "mixed": return "mixed"
    return "other"

def meta_of(sig):
    return META_CLASSES.index(sig)

class _RMS(nn.Module):
    def __init__(self, d):
        super().__init__(); self.w = nn.Parameter(torch.ones(d))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.w

class BlockA(nn.Module):
    def __init__(self, D=128, nh=4):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.o = nn.Linear(D, D, bias=False)
        self.ln3, self.ln4 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.mlp = nn.Sequential(nn.Linear(D, 4 * D, bias=False), nn.GELU(),
                                 nn.Linear(4 * D, D, bias=False))
        self.nh, self.Dh, self.D = nh, D // nh, D
        self.qn, self.kn = _RMS(D), _RMS(D)

    def forward(self, x, kv=None):
        B, T, D = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = self.qn(q).view(B, T, self.nh, self.Dh).transpose(1, 2)
        k = self.kn(k).view(B, T, self.nh, self.Dh).transpose(1, 2)
        v = v.view(B, T, self.nh, self.Dh).transpose(1, 2)
        if kv is None:
            K, V = k, v
        else:
            K = torch.cat([kv[0], k], 2); V = torch.cat([kv[1], v], 2)
        att = (q @ K.transpose(-2, -1)) / math.sqrt(self.Dh)
        a = (att.softmax(-1) @ V).transpose(1, 2).reshape(B, T, D)
        x = x + self.ln2(self.o(a))
        x = x + self.ln4(self.mlp(self.ln3(x)))
        return x, (K, V)

class StageA(nn.Module):
    def __init__(self, D=128, nh=4, V=106):
        super().__init__()
        self.emb = nn.Embedding(128, D)  # char codes 0..127
        self.blocks = nn.ModuleList([BlockA(D, nh) for _ in range(2)])
        self.type_head = nn.Linear(D, 8)
        self.meta_head = nn.Linear(D, 6)
        self.dec_head = nn.Linear(D, 128)
        self.D = D

    def forward(self, ids):
        x = self.emb(ids)
        pos = torch.zeros(MAXA, self.D, device=ids.device)
        p = torch.arange(MAXA, device=ids.device, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, self.D, 2, device=ids.device, dtype=torch.float32) * (-math.log(10000.0) / self.D))
        pos[:, 0::2] = torch.sin(p * div); pos[:, 1::2] = torch.cos(p * div)
        x = x + pos.unsqueeze(0)[:, : ids.shape[1]]
        for b in self.blocks:
            x, _ = b(x)
        cls = x[:, 0]
        # dec: next-char prediction over input sequence
        dec = self.dec_head(x[:, :-1])
        return self.type_head(cls), self.meta_head(cls), dec

    def tag(self, texts, dev="cpu"):
        self.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), 256):
                batch = [enc_a(t) for t in texts[i:i + 256]]
                mx = max(len(b) for b in batch)
                ids = torch.tensor([b + [0] * (mx - len(b)) for b in batch], device=dev)
                ty, mt, _ = self(ids)
                out += (mt.argmax(-1).tolist(), ty.argmax(-1).tolist())
        return out


def train_stage_a(items, steps=1500, bs=64, lr=2.5e-3, seed=9, log_every=100, ckpt=None, resume=False):
    """items: list of (text, kind, sig). Returns model + final metrics."""
    import random
    torch.manual_seed(seed); rng = random.Random(seed)
    m = StageA()
    if resume and ckpt and os.path.exists(ckpt):
        m.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    start = 0
    if resume and ckpt and os.path.exists(ckpt):
        start = torch.load(ckpt, map_location="cpu")["step"]
        opt.load_state_dict(torch.load(ckpt, map_location="cpu")["opt"])
    hist = []
    for step in range(start, steps):
        batch = rng.sample(items, min(bs, len(items)))
        seqs = [enc_a(t) for t, _, _ in batch]
        mx = max(len(s) for s in seqs)
        ids = torch.tensor([s + [0] * (mx - len(s)) for s in seqs])
        ty, mt, dec = m(ids)
        ty_t = torch.tensor([TYPE_CLASSES.index(type_of(k)) for _, k, _ in batch])
        mt_t = torch.tensor([META_CLASSES.index(s) for _, _, s in batch])
        # dec targets: next char over raw char-code sequence
        dec_t = torch.tensor([s[1:] + [0] * (mx - 1 - len(s) + 1) if len(s) < mx else s[1:mx] for s in seqs])
        L = F.cross_entropy(ty.float(), ty_t) + F.cross_entropy(mt.float(), mt_t) + 1.5 * F.cross_entropy(dec.reshape(-1, 128).float(), dec_t.reshape(-1))
        opt.zero_grad(); L.backward(); opt.step()
        if (step + 1) % log_every == 0 or step == steps - 1:
            with torch.no_grad():
                tacc = (ty.argmax(-1) == ty_t).float().mean().item()
                macc = (mt.argmax(-1) == mt_t).float().mean().item()
            hist.append(dict(step=step + 1, loss=round(L.item(), 4), type_acc=round(tacc, 4), meta_acc=round(macc, 4)))
            print(hist[-1], flush=True)
        if ckpt and ((step + 1) % 300 == 0 or step == steps - 1):
            torch.save(dict(step=step + 1, model=m.state_dict(), opt=opt.state_dict()), ckpt + ".tmp")
            os.replace(ckpt + ".tmp", ckpt)
    return m, hist


if __name__ == "__main__":
    m = StageA()
    print("stage A params:", sum(p.numel() for p in m.parameters()))
