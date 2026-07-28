# Chapter 11 - GRPO on a verifiable task, no demonstrations.
# Requires: names.txt. ~30s on a GPU. This is the whole RL algorithm in ~40 lines.
#
# From "Neural Networks: Zero to Hero - The Textbook".

"""Chapter 11 — GRPO on a verifiable task, at a scale you can run in minutes.

Stage 1: pretrain a small character-level GPT on names.txt (the base model).
Stage 2: define a verifier (a plain Python function, no reward model).
Stage 3: run GRPO and watch the reward curve.

The verifier stands in for "check the maths answer" in a real RLVR setup:
it is a function that returns a number, and nothing about GRPO cares what
is inside it.
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, time, copy, sys

torch.manual_seed(1337)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------------------------------------------------------------- data
words = open('names.txt').read().splitlines()
chars = sorted(set(''.join(words)))
stoi = {c: i + 1 for i, c in enumerate(chars)}; stoi['.'] = 0
itos = {i: c for c, i in stoi.items()}
V, block_size = len(stoi), 16

def encode_word(w):
    ids = [0] + [stoi[c] for c in w] + [0]
    return ids[:block_size + 1]

data = []
for w in words:
    ids = encode_word(w)
    ids += [0] * (block_size + 1 - len(ids))
    data.append(ids)
data = torch.tensor(data, dtype=torch.long)
print(f"{len(data):,} names, vocab {V}, block {block_size}")

# ---------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(s, d, h):
        super().__init__()
        s.at = nn.MultiheadAttention(d, h, batch_first=True)
        s.l1, s.l2 = nn.LayerNorm(d), nn.LayerNorm(d)
        s.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
    def forward(s, x, m):
        h = s.l1(x)
        x = x + s.at(h, h, h, attn_mask=m, need_weights=False)[0]
        return x + s.ff(s.l2(x))

class GPT(nn.Module):
    def __init__(s, d=128, h=4, L=4):
        super().__init__()
        s.tok, s.pos = nn.Embedding(V, d), nn.Embedding(block_size + 1, d)
        s.blocks = nn.ModuleList([Block(d, h) for _ in range(L)])
        s.lnf, s.head = nn.LayerNorm(d), nn.Linear(d, V)
    def forward(s, idx):
        T = idx.shape[1]
        m = torch.triu(torch.ones(T, T, device=idx.device, dtype=torch.bool), 1)
        x = s.tok(idx) + s.pos(torch.arange(T, device=idx.device))
        for b in s.blocks:
            x = b(x, m)
        return s.head(s.lnf(x))

    @torch.no_grad()
    def sample(s, n, temp=1.0):
        """Generate n names. Returns (tokens, text)."""
        idx = torch.zeros(n, 1, dtype=torch.long, device=device)
        for _ in range(block_size):
            logits = s(idx)[:, -1] / temp
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, -1), 1)], 1)
        out = []
        for row in idx[:, 1:].tolist():
            w = ''
            for t in row:
                if t == 0: break
                w += itos[t]
            out.append(w)
        return idx, out

# ---------------------------------------------------------------- stage 1: pretrain
model = GPT().to(device)
print(f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
PRETRAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
t0 = time.time()
for step in range(PRETRAIN):
    ix = torch.randint(0, len(data), (64,))
    b = data[ix].to(device)
    logits = model(b[:, :-1])
    loss = F.cross_entropy(logits.reshape(-1, V), b[:, 1:].reshape(-1))
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    if step % 500 == 0 or step == PRETRAIN - 1:
        print(f"  pretrain {step:5d}  loss {loss.item():.4f}  [{time.time()-t0:.0f}s]")

_, samples = model.sample(10)
print("base model samples:", samples)

# ---------------------------------------------------------------- stage 2: the verifier
def reward(name: str) -> float:
    """A programmatic grader. No reward model, no human labels, no gradients.
    Wants: a name starting with 'k' and ending with 'a'."""
    if len(name) < 3:
        return 0.0
    return 1.0 * (name[0] == 'k' and name[-1] == 'a')

_, base = model.sample(512)
base_rate = sum(reward(w) for w in base) / len(base)
print(f"base model satisfies the verifier {base_rate*100:.1f}% of the time")

# ---------------------------------------------------------------- stage 3: GRPO
ref = copy.deepcopy(model).eval()
for p in ref.parameters():
    p.requires_grad = False

G, EPS, BETA, LR = 64, 0.2, 0.02, 1e-5
opt = torch.optim.AdamW(model.parameters(), lr=LR)

def logprobs_of(m, idx):
    """log pi(token_t | everything before t), summed over the generated tokens."""
    logits = m(idx[:, :-1])
    lp = F.log_softmax(logits, -1)
    tok = idx[:, 1:].unsqueeze(-1)
    return lp.gather(-1, tok).squeeze(-1)          # (n, T)

STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 300
hist = []
t0 = time.time()
for step in range(STEPS):
    # --- rollouts: G completions from the same prompt (the start token)
    with torch.no_grad():
        idx, texts = model.sample(G)
        old_lp = logprobs_of(model, idx)
    R = torch.tensor([reward(t) for t in texts], device=device)

    # --- group-relative advantage: no critic, just the group mean
    A = R - R.mean()
    if R.std() > 0:
        A = A / (R.std() + 1e-8)

    # --- mask out padding after the first end token
    gen = idx[:, 1:]
    live = (gen != 0).float()
    live = torch.cat([torch.ones(G, 1, device=device), live[:, :-1]], 1)

    # --- clipped policy-gradient step with a KL leash to the reference
    new_lp = logprobs_of(model, idx)
    ratio = (new_lp - old_lp).exp()
    unclipped = ratio * A.unsqueeze(1)
    clipped = ratio.clamp(1 - EPS, 1 + EPS) * A.unsqueeze(1)
    pg = -torch.min(unclipped, clipped)
    with torch.no_grad():
        ref_lp = logprobs_of(ref, idx)
    kl = (new_lp - ref_lp)
    loss = ((pg + BETA * kl) * live).sum() / live.sum()

    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    hist.append(R.mean().item())
    if step % 25 == 0 or step == STEPS - 1:
        recent = sum(hist[-25:]) / len(hist[-25:])
        print(f"  grpo {step:4d}  reward {R.mean().item():.3f}  "
              f"avg25 {recent:.3f}  kl {kl.mean().item():+.4f}  [{time.time()-t0:.0f}s]")

_, final = model.sample(512)
final_rate = sum(reward(w) for w in final) / len(final)
print(f"\nbefore GRPO: {base_rate*100:.1f}%   after GRPO: {final_rate*100:.1f}%")
print("samples after GRPO:", final[:12])
