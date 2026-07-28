# Chapter 10 - supervised fine-tuning at character scale.
# Requires: names.txt. ~30s on a GPU, a few minutes on a CPU.
#
# From "Neural Networks: Zero to Hero - The Textbook".

"""Chapter 10 — supervised fine-tuning at a scale you can run on a laptop.

Same base model and same target behaviour as the GRPO script, so the two
mechanisms can be compared directly:

  SFT  learns the behaviour by imitating demonstrations of it.
  GRPO learns the behaviour from a verifier, with no demonstrations at all.

Usage:  python ch10_sft_char.py [pretrain_steps] [sft_steps] [n_demos]
"""
import torch, torch.nn as nn, torch.nn.functional as F, time, sys, copy

torch.manual_seed(1337)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

words = open('names.txt').read().splitlines()
chars = sorted(set(''.join(words)))
stoi = {c: i + 1 for i, c in enumerate(chars)}; stoi['.'] = 0
itos = {i: c for c, i in stoi.items()}
V, block_size = len(stoi), 16

def pack(ws):
    rows = []
    for w in ws:
        ids = ([0] + [stoi[c] for c in w] + [0])[:block_size + 1]
        rows.append(ids + [0] * (block_size + 1 - len(ids)))
    return torch.tensor(rows, dtype=torch.long)

data = pack(words)

class Block(nn.Module):
    def __init__(s, d, h):
        super().__init__()
        s.at = nn.MultiheadAttention(d, h, batch_first=True)
        s.l1, s.l2 = nn.LayerNorm(d), nn.LayerNorm(d)
        s.ff = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
    def forward(s, x, m):
        h = s.l1(x)
        x = x + s.at(h, h, h, attn_mask=m, need_weights=False)[0]
        return x + s.ff(s.l2(x))

class GPT(nn.Module):
    def __init__(s, d=128, h=4, L=4):
        super().__init__()
        s.tok, s.pos = nn.Embedding(V, d), nn.Embedding(block_size+1, d)
        s.blocks = nn.ModuleList([Block(d, h) for _ in range(L)])
        s.lnf, s.head = nn.LayerNorm(d), nn.Linear(d, V)
    def forward(s, idx):
        T = idx.shape[1]
        m = torch.triu(torch.ones(T, T, device=idx.device, dtype=torch.bool), 1)
        x = s.tok(idx) + s.pos(torch.arange(T, device=idx.device))
        for b in s.blocks: x = b(x, m)
        return s.head(s.lnf(x))
    @torch.no_grad()
    def sample(s, n):
        idx = torch.zeros(n, 1, dtype=torch.long, device=device)
        for _ in range(block_size):
            logits = s(idx)[:, -1]
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, -1), 1)], 1)
        out = []
        for row in idx[:, 1:].tolist():
            w = ''
            for t in row:
                if t == 0: break
                w += itos[t]
            out.append(w)
        return out

def satisfies(w):
    return len(w) >= 3 and w[0] == 'k' and w[-1] == 'a'

def rate(m, n=1024):
    s = m.sample(n)
    return sum(satisfies(w) for w in s) / n, s[:8]

PRE = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
SFT = int(sys.argv[2]) if len(sys.argv) > 2 else 400
NDEMO = int(sys.argv[3]) if len(sys.argv) > 3 else 0     # 0 = use all of them

# ---- stage 1: pretrain the base model
model = GPT().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
t0 = time.time()
for step in range(PRE):
    b = data[torch.randint(0, len(data), (64,))].to(device)
    loss = F.cross_entropy(model(b[:, :-1]).reshape(-1, V), b[:, 1:].reshape(-1))
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
print(f"pretrained in {time.time()-t0:.0f}s, final loss {loss.item():.4f}")
r0, s0 = rate(model)
print(f"BASE      satisfies the target {r0*100:5.2f}%   e.g. {s0}")

# ---- stage 2: supervised fine-tuning on demonstrations of the target behaviour
demos = [w for w in words if satisfies(w)]
if NDEMO:
    demos = demos[:NDEMO]
print(f"\nSFT on {len(demos)} demonstrations "
      f"({len(demos)/len(words)*100:.2f}% of the corpus)")
dd = pack(demos)
sft = copy.deepcopy(model)
opt = torch.optim.AdamW(sft.parameters(), lr=1e-4)
t0 = time.time()
for step in range(SFT):
    b = dd[torch.randint(0, len(dd), (min(64, len(dd)),))].to(device)
    loss = F.cross_entropy(sft(b[:, :-1]).reshape(-1, V), b[:, 1:].reshape(-1))
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    if step % 100 == 0 or step == SFT - 1:
        r, _ = rate(sft, 256)
        print(f"  sft {step:4d}  loss {loss.item():.4f}  target rate {r*100:5.1f}%")
r1, s1 = rate(sft)
print(f"AFTER SFT satisfies the target {r1*100:5.2f}%   e.g. {s1}  [{time.time()-t0:.0f}s]")

# how much of the original distribution survived?
uniq = len(set(sft.sample(512)))
print(f"distinct names in 512 samples after SFT: {uniq}")
