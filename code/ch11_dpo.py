# Chapter 11 - DPO on the fine-tuned GPT-2 from Chapter 10.
# Requires: a GPU and gpt2-sft/ (run ch10_sft_gpt2.py first). ~4 minutes.
#
# From "Neural Networks: Zero to Hero - The Textbook".

"""Chapter 11 — DPO: learning from preferences instead of a verifier.

The SFT model from Chapter 10 answers in the right format but ignores
instructions like "give three tips" (it gave seven). There is no verifier for
"is this a good answer", so we learn from comparisons instead.

The judge here is a plain function that prefers the shorter of two responses.
That is a stand-in for a human labeller: real preference data comes from
people choosing between two outputs, and a reward model is fitted to those
choices. The mechanism being demonstrated is identical; only the source of
the preference differs.

Usage: python ch11_dpo.py [steps]
"""
import json, torch, torch.nn.functional as F, random, sys, time, copy
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

torch.manual_seed(0); random.seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tok = GPT2TokenizerFast.from_pretrained('gpt2-sft')
tok.pad_token = tok.eos_token
tok.padding_side = 'left'          # decoder-only models must be left-padded
policy = GPT2LMHeadModel.from_pretrained('gpt2-sft').to(device)
ref = GPT2LMHeadModel.from_pretrained('gpt2-sft').to(device).eval()
for p in ref.parameters():
    p.requires_grad = False

chat = lambda q: f"### Instruction:\n{q}\n\n### Response:\n"
prompts = [x['instruction'] for x in json.load(open('alpaca.json'))
           if not x['input'].strip()][:400]

@torch.no_grad()
def rollout(qs, n=48):
    """Two sampled responses per prompt."""
    policy.eval()
    enc = tok([chat(q) for q in qs], return_tensors='pt', padding=True).to(device)
    out = policy.generate(**enc, max_new_tokens=n, do_sample=True, temperature=0.9,
                          top_p=0.95, pad_token_id=tok.eos_token_id,
                          num_return_sequences=2)
    plen = enc['input_ids'].shape[1]
    txt = [tok.decode(o[plen:], skip_special_tokens=True) for o in out]
    return [(txt[2*i], txt[2*i+1]) for i in range(len(qs))]

def judge(a, b):
    """Prefer the shorter response. Returns (winner, loser)."""
    return (a, b) if len(a) <= len(b) else (b, a)

def seq_logp(m, prompt, response):
    p_ids = tok(prompt)['input_ids']
    full = tok(prompt + response, truncation=True, max_length=320)['input_ids']
    x = torch.tensor([full], device=device)
    logits = m(x).logits[:, :-1]
    lp = F.log_softmax(logits, -1).gather(-1, x[:, 1:].unsqueeze(-1)).squeeze(-1)
    return lp[0, len(p_ids)-1:].sum()          # response tokens only

@torch.no_grad()
def mean_len(n=120):
    policy.eval()
    qs = prompts[:n]
    outs = []
    for i in range(0, n, 20):
        enc = tok([chat(q) for q in qs[i:i+20]], return_tensors='pt', padding=True).to(device)
        o = policy.generate(**enc, max_new_tokens=60, do_sample=True, temperature=0.7,
                            top_p=0.9, pad_token_id=tok.eos_token_id)
        plen = enc['input_ids'].shape[1]
        outs += [len(tok.decode(r[plen:], skip_special_tokens=True).strip()) for r in o]
    return sum(outs)/len(outs)

BETA = 0.1
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 150
opt = torch.optim.AdamW(policy.parameters(), lr=2e-6)

before = mean_len()
print(f"mean response length before DPO: {before:.0f} characters")

acc_hist = []
t0 = time.time()
for step in range(STEPS):
    qs = random.sample(prompts, 8)
    pairs = rollout(qs)
    policy.train()
    losses, margins, accs = [], [], []
    for q, (a, b) in zip(qs, pairs):
        if len(a.strip()) < 5 or len(b.strip()) < 5 or a == b:
            continue
        w, l = judge(a, b)
        p = chat(q)
        pi_w, pi_l = seq_logp(policy, p, w), seq_logp(policy, p, l)
        with torch.no_grad():
            rf_w, rf_l = seq_logp(ref, p, w), seq_logp(ref, p, l)
        margin = (pi_w - rf_w) - (pi_l - rf_l)          # implicit reward margin
        losses.append(-F.logsigmoid(BETA * margin))
        margins.append(margin.item())
        accs.append(1.0 if margin.item() > 0 else 0.0)
    if not losses:
        continue
    acc_hist += accs
    loss = torch.stack(losses).mean()
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    opt.step()
    if step % 25 == 0 or step == STEPS - 1:
        run = acc_hist[-40:] or [0]
        print(f"  dpo {step:4d}  loss {loss.item():.4f}  margin {sum(margins)/len(margins):+.3f}  "
              f"pref-acc(last40) {sum(run)/len(run)*100:4.0f}%  [{time.time()-t0:.0f}s]")

after = mean_len()
print(f"\nmean response length: {before:.0f} -> {after:.0f} characters "
      f"({(after-before)/before*100:+.0f}%)")

policy.eval()
for q in ["Give three tips for staying healthy.", "What is the capital of France?"]:
    enc = tok(chat(q), return_tensors='pt').to(device)
    o = policy.generate(**enc, max_new_tokens=60, do_sample=True, temperature=0.7,
                        top_p=0.9, pad_token_id=tok.eos_token_id)
    print(f"\nQ: {q}\nA: {tok.decode(o[0][enc['input_ids'].shape[1]:], skip_special_tokens=True).strip()!r}")
