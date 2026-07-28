# Chapter 8 — byte pair encoding from scratch, plus tokenizer comparisons.
# Requires: pip install tiktoken for the comparison section. Runs in seconds.
#
# From "Neural Networks: Zero to Hero — The Textbook".

text = ("Unicode is a standard that assigns a number to every character in every writing "
        "system. UTF-8 is a way of writing those numbers as bytes. Tokenization sits between "
        "raw text and the neural network, and it is responsible for a surprising number of "
        "the strange behaviours that large language models exhibit in practice.") * 4

tokens = list(text.encode("utf-8"))
print("text length in characters:", len(text))
print("length in bytes/tokens:", len(tokens))
print("first 20 byte values:", tokens[:20])

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    newids, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx); i += 2
        else:
            newids.append(ids[i]); i += 1
    return newids

stats = get_stats(tokens)
top = sorted(((v, k) for k, v in stats.items()), reverse=True)[:3]
print("top pairs:", [(bytes(k).decode('utf-8', errors='replace'), v) for v, k in top])

vocab_size = 276
num_merges = vocab_size - 256
ids = list(tokens)
merges = {}
for i in range(num_merges):
    stats = get_stats(ids)
    pair = max(stats, key=stats.get)
    idx = 256 + i
    ids = merge(ids, pair, idx)
    merges[pair] = idx
print("tokens before:", len(tokens), "after 20 merges:", len(ids))
print(f"compression ratio: {len(tokens)/len(ids):.2f}X")

vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]
print("learned tokens:", [vocab[i].decode('utf-8', errors='replace') for i in range(256, 276)])

def decode(ids): return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")
def encode(text):
    tokens = list(text.encode("utf-8"))
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges: break
        tokens = merge(tokens, pair, merges[pair])
    return tokens
s = "the neural network is a standard"
print("encode:", encode(s))
print("roundtrip ok:", decode(encode(s)) == s)

# non-english cost demo
en = "hello how are you"
hi = "नमस्ते आप कैसे हैं"
print(f"english bytes: {len(en.encode('utf-8'))} | hindi bytes: {len(hi.encode('utf-8'))} "
      f"| ratio {len(hi.encode('utf-8'))/len(en.encode('utf-8')):.1f}x")
try:
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    enc4 = tiktoken.get_encoding("cl100k_base")
    print("gpt2 tokens:", len(enc.encode(en)), "vs", len(enc.encode(hi)))
    print("gpt4 vocab:", enc4.n_vocab, "gpt2 vocab:", enc.n_vocab)
except ImportError:
    print("tiktoken not installed (pip install tiktoken)")
