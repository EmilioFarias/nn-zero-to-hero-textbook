# Chapter 3 — the learning-rate sweep that produces the table in the book.
# Requires: names.txt. ~2 minutes on a CPU.
#
# From "Neural Networks: Zero to Hero — The Textbook".

import torch, torch.nn.functional as F, random, math
words=open('names.txt').read().splitlines()
chars=sorted(list(set(''.join(words)))); stoi={s:i+1 for i,s in enumerate(chars)}; stoi['.']=0
block_size=3
def build(ws):
    X,Y=[],[]
    for w in ws:
        ctx=[0]*block_size
        for ch in w+'.':
            ix=stoi[ch]; X.append(ctx); Y.append(ix); ctx=ctx[1:]+[ix]
    return torch.tensor(X),torch.tensor(Y)
random.seed(42); random.shuffle(words)
n1=int(0.8*len(words)); Xtr,Ytr=build(words[:n1])
for lr in [0.0001,0.001,0.01,0.1,1.0,10.0]:
    g=torch.Generator().manual_seed(2147483647)
    C=torch.randn((27,10),generator=g); W1=torch.randn((30,200),generator=g)*0.2
    b1=torch.randn(200,generator=g)*0.01; W2=torch.randn((200,27),generator=g)*0.01; b2=torch.zeros(27)
    ps=[C,W1,b1,W2,b2]
    for p in ps: p.requires_grad=True
    for i in range(2000):
        ix=torch.randint(0,Xtr.shape[0],(32,),generator=g)
        h=torch.tanh(C[Xtr[ix]].view(-1,30)@W1+b1)
        loss=F.cross_entropy(h@W2+b2,Ytr[ix])
        for p in ps: p.grad=None
        loss.backward()
        for p in ps: p.data+=-lr*p.grad
    with torch.no_grad():
        h=torch.tanh(C[Xtr].view(-1,30)@W1+b1); full=F.cross_entropy(h@W2+b2,Ytr).item()
    print(f"lr={lr:<8} training loss after 2000 steps: {full:.4f}")
