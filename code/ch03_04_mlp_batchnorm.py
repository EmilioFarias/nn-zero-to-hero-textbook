# Chapters 3 and 4 — the MLP, and the three initialization variants.
# Requires: names.txt. ~7 minutes on a CPU: it trains three networks of 30,000 steps.
#
# From "Neural Networks: Zero to Hero — The Textbook".

import torch, torch.nn.functional as F, random, math

words = open('names.txt').read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {s: i+1 for i, s in enumerate(chars)}; stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}

block_size = 3
def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context); Y.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)

random.seed(42); random.shuffle(words)
n1, n2 = int(0.8*len(words)), int(0.9*len(words))
Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])
print("train:", tuple(Xtr.shape), "dev:", tuple(Xdev.shape), "test:", tuple(Xte.shape))

def run(fix_init, use_bn, steps=30000, n_hidden=200, n_embd=10, seed=2147483647, label=""):
    g = torch.Generator().manual_seed(seed)
    C  = torch.randn((27, n_embd), generator=g)
    W1 = torch.randn((n_embd*block_size, n_hidden), generator=g)
    W1 = W1 * ((5/3)/math.sqrt(n_embd*block_size)) if fix_init else W1
    b1 = torch.randn(n_hidden, generator=g) * (0.01 if fix_init else 1.0)
    W2 = torch.randn((n_hidden, 27), generator=g) * (0.01 if fix_init else 1.0)
    b2 = torch.randn(27, generator=g) * (0.0 if fix_init else 1.0)
    bngain = torch.ones((1, n_hidden)); bnbias = torch.zeros((1, n_hidden))
    bnmean_running = torch.zeros((1, n_hidden)); bnstd_running = torch.ones((1, n_hidden))
    parameters = [C, W1, b1, W2, b2] + ([bngain, bnbias] if use_bn else [])
    print(f"[{label}] parameters:", sum(p.nelement() for p in parameters))
    for p in parameters: p.requires_grad = True

    first_loss = None
    for i in range(steps):
        ix = torch.randint(0, Xtr.shape[0], (32,), generator=g)
        emb = C[Xtr[ix]]
        hpreact = emb.view(-1, n_embd*block_size) @ W1 + b1
        if use_bn:
            bnmeani = hpreact.mean(0, keepdim=True); bnstdi = hpreact.std(0, keepdim=True)
            hpreact = bngain * (hpreact - bnmeani) / bnstdi + bnbias
            with torch.no_grad():
                bnmean_running = 0.999*bnmean_running + 0.001*bnmeani
                bnstd_running  = 0.999*bnstd_running  + 0.001*bnstdi
        h = torch.tanh(hpreact)
        logits = h @ W2 + b2
        loss = F.cross_entropy(logits, Ytr[ix])
        if i == 0:
            first_loss = loss.item()
            sat = (h.abs() > 0.99).float().mean().item()
            print(f"[{label}] initial loss: {first_loss:.4f}   (theoretical for uniform: {math.log(27):.4f})")
            print(f"[{label}] fraction of hidden units saturated (|h|>0.99) at init: {sat*100:.1f}%")
        for p in parameters: p.grad = None
        loss.backward()
        lr = 0.1 if i < steps*0.6 else 0.01
        for p in parameters: p.data += -lr * p.grad

    @torch.no_grad()
    def split_loss(X, Y):
        emb = C[X]
        hpreact = emb.view(-1, n_embd*block_size) @ W1 + b1
        if use_bn: hpreact = bngain * (hpreact - bnmean_running) / bnstd_running + bnbias
        h = torch.tanh(hpreact)
        return F.cross_entropy(h @ W2 + b2, Y).item()
    tr, dv = split_loss(Xtr, Ytr), split_loss(Xdev, Ydev)
    print(f"[{label}] train {tr:.4f} | dev {dv:.4f}")
    return C, W1, b1, W2, b2

print("\n=== naive init, no batchnorm ===")
run(False, False, label="naive")
print("\n=== fixed init, no batchnorm ===")
run(True, False, label="fixed")
print("\n=== fixed init + batchnorm ===")
C, W1, b1, W2, b2 = run(True, True, label="bnorm")

# sampling
g = torch.Generator().manual_seed(2147483647 + 10)
out_names = []
for _ in range(8):
    out, context = [], [0]*block_size
    while True:
        emb = C[torch.tensor([context])]
        h = torch.tanh(emb.view(1, -1) @ W1 + b1)
        probs = F.softmax(h @ W2 + b2, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        if ix == 0: break
        out.append(itos[ix])
    out_names.append(''.join(out))
print("\nsamples:", out_names)
