# Chapter 10 - supervised fine-tuning on the real GPT-2 124M.
# Requires: a GPU, alpaca.json, and `pip install transformers`. ~3 minutes.
#
# From "Neural Networks: Zero to Hero - The Textbook".

"""Chapter 10 — supervised fine-tuning on the real GPT-2 124M from Chapter 9.

Shows the thing that makes an assistant an assistant: the base model does not
answer questions, it continues documents. SFT does not add knowledge, it
teaches a format.

Usage: python ch10_sft_gpt2.py [steps]
"""
import json, torch, torch.nn.functional as F, time, sys, random
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

torch.manual_seed(1337); random.seed(1337)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tok = GPT2TokenizerFast.from_pretrained('gpt2')
tok.pad_token = tok.eos_token
model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
print(f"GPT-2: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")

PROMPTS = ["What is the capital of France?",
           "Give three tips for staying healthy.",
           "Explain what a neural network is."]

@torch.no_grad()
def ask(m, q, template, n=60):
    m.eval()
    ids = tok(template(q), return_tensors='pt').to(device)
    out = m.generate(**ids, max_new_tokens=n, do_sample=True, temperature=0.7,
                     top_p=0.9, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids['input_ids'].shape[1]:], skip_special_tokens=True).strip()

raw = lambda q: q
chat = lambda q: f"### Instruction:\n{q}\n\n### Response:\n"

print("\n" + "="*70)
print("BASE MODEL, asked directly (no fine-tuning)")
print("="*70)
for q in PROMPTS:
    print(f"\nQ: {q}\nA: {ask(model, q, raw)!r}")

# ---------------------------------------------------------------- SFT data
data = [x for x in json.load(open('alpaca.json')) if not x['input'].strip()]
random.shuffle(data)
data = data[:4000]
MAXLEN = 256

def encode(ex):
    text = chat(ex['instruction']) + ex['output'] + tok.eos_token
    ids = tok(text, truncation=True, max_length=MAXLEN)['input_ids']
    prompt_len = len(tok(chat(ex['instruction']))['input_ids'])
    labels = list(ids)
    for i in range(min(prompt_len, len(labels))):
        labels[i] = -100          # train only on the response, not the prompt
    return ids, labels

encoded = [encode(x) for x in data]
print(f"\n{len(encoded)} instruction examples, "
      f"median length {sorted(len(i) for i, _ in encoded)[len(encoded)//2]} tokens")

def batch(bs=8):
    picks = [encoded[random.randrange(len(encoded))] for _ in range(bs)]
    n = max(len(i) for i, _ in picks)
    x = torch.full((bs, n), tok.eos_token_id); y = torch.full((bs, n), -100)
    for r, (ids, labels) in enumerate(picks):
        x[r, :len(ids)] = torch.tensor(ids); y[r, :len(labels)] = torch.tensor(labels)
    return x.to(device), y.to(device)

# ---------------------------------------------------------------- train
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 600
opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-5, total_steps=STEPS, pct_start=0.1)
model.train()
t0 = time.time()
for step in range(STEPS):
    x, y = batch()
    loss = model(input_ids=x, labels=y).loss
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()
    if step % 100 == 0 or step == STEPS - 1:
        print(f"  sft {step:4d}  loss {loss.item():.4f}  [{time.time()-t0:.0f}s]")

print("\n" + "="*70)
print("AFTER SFT, same questions in the chat template")
print("="*70)
for q in PROMPTS:
    print(f"\nQ: {q}\nA: {ask(model, q, chat)!r}")

model.save_pretrained('gpt2-sft'); tok.save_pretrained('gpt2-sft')
print("\nsaved to gpt2-sft/")
