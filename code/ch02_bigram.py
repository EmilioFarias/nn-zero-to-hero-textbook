# Chapter 2 — the bigram model, counted and then learned.
# Requires: names.txt in the working directory. ~1 minute on a CPU.
#
# From "Neural Networks: Zero to Hero — The Textbook".

import torch, torch.nn.functional as F

words = open('names.txt').read().splitlines()
print("num words:", len(words), "| shortest:", min(len(w) for w in words), "| longest:", max(len(w) for w in words))
print("first 5:", words[:5])

chars = sorted(list(set(''.join(words))))
stoi = {s: i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}
print("vocab size:", len(itos))

# ---- counting model
N = torch.zeros((27, 27), dtype=torch.int32)
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        N[stoi[ch1], stoi[ch2]] += 1
print("count of '.'->'m' (names starting with m):", N[0, stoi['m']].item())
print("total bigrams:", N.sum().item())

P = (N + 1).float()
P /= P.sum(1, keepdim=True)
print("row 0 sums to:", P[0].sum().item())

g = torch.Generator().manual_seed(2147483647)
samples = []
for _ in range(5):
    out, ix = [], 0
    while True:
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        if ix == 0: break
        out.append(itos[ix])
    samples.append(''.join(out))
print("samples:", samples)

log_likelihood, n = 0.0, 0
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        log_likelihood += torch.log(P[stoi[ch1], stoi[ch2]])
        n += 1
print(f"counting model loss (smoothed): {-log_likelihood/n:.4f}")

# unsmoothed, to show the infinity
P0 = N.float(); P0 /= P0.sum(1, keepdim=True)
ll, m = 0.0, 0
for ch1, ch2 in zip("andrejq", "ndrejq"):
    ll += torch.log(P0[stoi[ch1], stoi[ch2]]); m += 1
print("unsmoothed loss on 'andrejq':", (-ll/m).item())

# ---- neural net version
xs, ys = [], []
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        xs.append(stoi[ch1]); ys.append(stoi[ch2])
xs = torch.tensor(xs); ys = torch.tensor(ys)
num = xs.nelement()
print("number of training examples:", num)

g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True)
for k in range(200):
    xenc = F.one_hot(xs, num_classes=27).float()
    logits = xenc @ W
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdim=True)
    loss = -probs[torch.arange(num), ys].log().mean() + 0.01*(W**2).mean()
    if k % 50 == 0 or k == 199: print(f"  step {k:3d} loss {loss.item():.4f}")
    W.grad = None
    loss.backward()
    W.data += -50 * W.grad
print(f"neural bigram final loss: {loss.item():.4f}")
