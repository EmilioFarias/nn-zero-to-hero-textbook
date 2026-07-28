# Neural Networks: Zero to Hero — The Textbook

A written, runnable edition of [Andrej Karpathy's *Neural Networks: Zero to Hero*](https://karpathy.ai/zero-to-hero.html).

**[Read it online →](https://claude.ai/code/artifact/a9ecca2e-da63-4c2c-a510-51a41cf3fff0)** · or open [`textbook.md`](textbook.md) · or work through the [notebooks](notebooks/)

## Why I made this

I wanted to take Karpathy's videos and turn them into a textbook that walks you through everything, the code and the concepts together, because I think that is a better way to learn.

Video is hard to learn from and easy to feel like you're learning from. You can watch fifteen hours of someone else typing, nod along the whole way, and end up unable to write any of it yourself. You can't skim a video, you can't search it, you can't sit with one paragraph for ten minutes, and you can't copy a line out of it without scrubbing back and forth. Meanwhile the actual understanding lives in the details that scroll past in a few seconds.

So this is the same course as a book. Every concept is written down and built up from the thing before it, every piece of code is in the text where you need it, and you can move at your own speed.

## What's different from just watching

**It assumes you know nothing.** Derivatives, logarithms, matrix multiplication, probability, and the meaning of the word "model" are all built from scratch before they're used. The rule I wrote it to: you should never need to open a search engine. If you hit a term that isn't explained, that's a bug in the book.

**Every concept is explained three times.** Once for a six-year-old, once for you, and once for someone with a PhD in the field. If you can only repeat the first version, you understand it well enough to teach it. If you can follow the third, you can hold your own in a conversation with a specialist.

**Every code block was actually run.** The outputs printed in the book are real outputs from a real machine, not what I assumed would happen. Six numbers I predicted turned out to be wrong when I measured them, and they're corrected in the text. That's the argument for running code rather than describing it.

**It's built to follow along in Jupyter.** Ten notebooks, 69 runnable cells, one per chapter.

## What's inside

| | |
|---|---|
| [`textbook.md`](textbook.md) | The whole book, ~31,000 words |
| [`index.html`](index.html) | The web edition, self-contained, light and dark |
| [`notebooks/`](notebooks/) | 10 Jupyter notebooks, one per chapter, 69 runnable cells |
| [`code/`](code/) | The standalone scripts, if you'd rather not use notebooks |

The nine chapters follow the nine lectures: micrograd and backpropagation, the bigram model, the MLP, initialization and BatchNorm, manual backprop, WaveNet, building GPT, the tokenizer, and reproducing GPT-2.

## Getting started

```bash
git clone https://github.com/EmilioFarias/zero-to-hero-textbook
cd zero-to-hero-textbook
python3 -m venv .venv && source .venv/bin/activate
pip install torch numpy matplotlib jupyter tiktoken

# the two datasets the whole course uses
curl -O https://raw.githubusercontent.com/karpathy/makemore/master/names.txt
curl -O https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

jupyter notebook notebooks/
```

Chapters 1 through 6 and 8 run on any laptop. Chapter 7 will run on a laptop slowly; the full result wants a GPU. Chapter 9 needs several GPUs or about $10 of rented cloud time.

## Some results from the book

Everything below was measured while writing, not quoted:

| What | Result |
|---|---|
| micrograd MLP | 41 parameters, loss 3.14 → 0.033 |
| Bigram counting model | loss 2.4544, and `inf` on a name containing `jq` |
| Same model as a neural net | 2.48, converging to the same answer |
| MLP with embeddings | 2.2778 |
| The initialization bug | first loss **27.88** where theory says **3.2958**, 61% of neurons dead |
| After fixing initialization | 3.3179 at init, 8.1% saturated, dev loss 2.1481 |
| GPT on Shakespeare | 10.79M parameters, 44 min on one GPU, **val loss 1.4874** (Karpathy reports 1.48) |
| Tokenizer cost, English vs Hindi | 4 tokens vs 30, a 7.5× penalty for the same sentence |

The Shakespeare model's actual output, and an honest note about how its best checkpoint was at step 4,500 rather than at the end, are both in Chapter 7.

## Credit

The course, the curriculum, the code it's based on, and the teaching are all Andrej Karpathy's. This is an unofficial companion I made for my own learning, not affiliated with or endorsed by him.

- Course: <https://karpathy.ai/zero-to-hero.html>
- His code: <https://github.com/karpathy/nn-zero-to-hero> (MIT)
- The lectures themselves are linked at the top of every chapter. Watch them. This book is a companion to them, not a replacement.

The book was assembled with the help of Claude, working from the transcripts of all nine lectures, community notes, and code executed on my machine. Sources and method are in Appendix D.

## License

[MIT](LICENSE) for the code. The prose is mine to share; the ideas being explained are Karpathy's and belong to the people who published the papers he teaches from, all of which are cited in Appendix D.
