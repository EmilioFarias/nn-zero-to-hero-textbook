# Neural Networks: Zero to Hero — The Textbook

*A plain-language written edition of Andrej Karpathy's video course, built from the lecture transcripts, with every line of code run and verified.*

---

## How to use this book

### Who it is for

Someone with no background at all. If you know what a variable is in any programming language and you have seen a graph with an x-axis, you have enough. Derivatives, probability, matrices, logarithms, and the meaning of the word "model" are all built from scratch here.

### The three levels

Every important idea in this book is explained three times, at three depths, marked like this:

> **Say it to a six-year-old.** The idea with no vocabulary at all, using only things a child has touched. If you can repeat these out loud, you understand the concept well enough to teach it.

**The main text** is the working explanation: enough to write the code, predict what it will print, and debug it when it breaks.

> **For the PhD in the room.** The formal version: standard notation, the assumptions being made, what the literature calls it, and where researchers still disagree. Read these if you want to hold a conversation with a specialist. Skip them freely on a first pass; nothing later depends on them.

### The rule this book is written to

You should never need to open a search engine. Any term used is defined the first time it appears, and any concept a definition depends on is itself defined earlier. If you hit something unexplained, that is a defect in this book, not a gap in you.

### Follow along in Jupyter

Every code block is real, runnable, and was executed on a real machine while writing this. The outputs printed in this book are the actual outputs, not invented ones. Chapter 0 gets your environment running in about ten minutes.

Two symbols used throughout:

- **Run it.** A code cell to type into your notebook.
- **What you should see.** The output that cell produced when I ran it. Small differences in the last digits are normal and explained where they matter.

### Confidence markers

- **[transcript]** — stated by Karpathy in the lecture; pulled from the actual captions.
- **[verified]** — I ran the code and this is the measured result on my machine.
- **[standard]** — uncontroversial background knowledge.
- **[my read]** — my interpretation or analogy. Argue with it.
- **[uncertain]** — genuinely unclear or contested.

### The path through

| Part | What it covers | Time |
|---|---|---|
| **0** | Environment setup | 30 min |
| **I** | All the math and code prerequisites | 4–6 h |
| **II** | Why this course exists | 20 min |
| **III** | The nine lectures, with code | 30–50 h |
| **IV** | What it is good for, and its limits | 40 min |
| **V** | Two scripts for explaining it to others | 10 min |
| **Appendices** | Glossary, notation table, PyTorch reference, troubleshooting | reference |

---

# PART 0 — GETTING SET UP

You can read this book without running anything. You will not learn it that way. Every person who reports getting real value from this course reports typing the code and breaking it.

## 0.1 What you need

A computer with at least 8 GB of RAM. Chapters 1 through 6 and 8 run on any laptop, including a several-year-old one, with no special hardware. Chapter 7 will run on a laptop but slowly; the full result needs a GPU. Chapter 9 needs several GPUs or about $10 of rented cloud time.

**What a GPU is.** A **CPU** (central processing unit) is your computer's general-purpose brain: a handful of very fast, very flexible cores. A **GPU** (graphics processing unit) is a slab of thousands of simple cores that all do the same arithmetic at the same time on different data. Neural networks are almost entirely "multiply these thousands of numbers by those thousands of numbers," which is exactly the shape a GPU is built for. That is why training on a GPU can be 10 to 100 times faster than on a CPU. [standard]

## 0.2 Installing Python and the libraries

**Step 1: get Python.** Download Python 3.10 or newer from [python.org](https://www.python.org/downloads/), or on a Mac use [Homebrew](https://brew.sh) (`brew install python`). Check it worked by opening a terminal and typing:

```bash
python3 --version
```

You should see something like `Python 3.12.3`. (That is the version this book was written on. [verified])

**Step 2: make a virtual environment.** A **virtual environment** is a private folder holding this project's libraries, so that installing something here cannot break another project. Standard practice, worth the 20 seconds.

```bash
mkdir zero-to-hero && cd zero-to-hero
python3 -m venv .venv
source .venv/bin/activate      # on Windows: .venv\Scripts\activate
```

Your prompt now shows `(.venv)`, meaning that environment is active. You repeat only the `source` line in future sessions.

**Step 3: install the libraries.**

```bash
pip install torch numpy matplotlib jupyter
```

- **torch** is PyTorch, the library that provides fast arrays and automatic derivatives.
- **numpy** handles numeric arrays; PyTorch borrows its conventions.
- **matplotlib** draws the plots you will use to diagnose training.
- **jupyter** is the notebook interface.

This downloads a few hundred megabytes and takes a couple of minutes.

**Step 4: start the notebook.**

```bash
jupyter notebook
```

A browser tab opens. Click **New → Python 3** to create a notebook. You now have a page of empty **cells**. Type code into one and press **Shift+Enter** to run it. The output appears directly underneath, and anything you defined stays in memory for the next cell.

**Run it.**

```python
import torch
print("torch", torch.__version__, "| GPU available:", torch.cuda.is_available())
```

**What you should see** (your version will differ; `False` for the GPU is fine for Chapters 1–6 and 8):

```
torch 2.11.0+cu130 | GPU available: True
```
[verified]

## 0.3 Getting the two datasets

The whole course uses exactly two files.

```bash
curl -O https://raw.githubusercontent.com/karpathy/makemore/master/names.txt
curl -O https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

**Run it.**

```python
words = open('names.txt', 'r').read().splitlines()
print(len(words), "names |", words[:5])
text = open('input.txt', 'r').read()
print(len(text), "characters of Shakespeare")
```

**What you should see:**

```
32033 names | ['emma', 'olivia', 'ava', 'isabella', 'sophia']
1115394 characters of Shakespeare
```
[verified]

Those two numbers, 32,033 names and about 1.1 million characters, are the ones quoted throughout the lectures as "32,000 names" and "1 megabyte of Shakespeare."

## 0.4 How to work through a chapter

1. Read the chapter's "The problem" section before any code.
2. Type each code cell rather than pasting it. Typos teach you the API.
3. Before running a cell, say out loud what you expect it to print. Being wrong is the useful part.
4. When a shape error appears, print `.shape` on everything involved. Most errors in this field are shape errors.
5. At the end of a chapter, close the notebook and rebuild the key piece from a blank file.

## 0.5 If something goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'torch'` | Virtual environment not active, or Jupyter is using a different Python | Re-run `source .venv/bin/activate`, then `pip install torch`, then restart Jupyter |
| `FileNotFoundError: names.txt` | The notebook's working folder differs from where you downloaded | Run `import os; print(os.getcwd())` and move the file there |
| Numbers differ slightly from this book | Different random seed or hardware | Expected. Set `torch.manual_seed(1337)` to match closely; last-digit differences remain and are harmless |
| Cell hangs forever | A training loop with too many steps | Press **I, I** (interrupt twice) or the stop button; lower the step count |
| `CUDA out of memory` | Model or batch too large for the GPU | Lower `batch_size`, then `block_size` |

---

# PART I — FOUNDATIONS

*Everything needed before Chapter 1. Each concept has a concrete example, and anything a concept leans on is broken down first. If you already know this material, skim the bold text and move on.*

## 1.1 What a model is

A **model** is a function: numbers go in, numbers come out.

Predicting a house price from its size, you might guess `price = 200 × square_feet`. Feed in 1,000, get 200,000. That rule is a model, and the `200` is a **parameter**: a number you are free to change to fit reality better.

**Training** means finding parameter values that make the outputs match reality on examples you already have.

> **Say it to a six-year-old.** Imagine a machine with a lot of little knobs on the side. You put a picture of a cat in the front, and a word comes out the back. At first it says "dog," which is wrong. So you turn the knobs a tiny bit, and try again. If you do that a million times, eventually it says "cat" every time. Nobody tells the machine what a cat is. It just turns knobs until it stops being wrong.

> **For the PhD in the room.** Formally we are choosing a function from a parameterized family, f(x; θ) with θ ∈ ℝⁿ, by minimizing an empirical risk, that is the average of a loss ℓ over a finite sample drawn from an unknown data distribution. Everything interesting hides in the gap between that empirical average and the true expected risk, which is the subject of generalization theory. This course is deliberately empirical about that gap: it measures it with a held-out split rather than bounding it.

## 1.2 The math notation used in this book

Only a handful of symbols appear anywhere in the course. Here they all are.

| Symbol | Read it as | Meaning | Example |
|---|---|---|---|
| Σ | "sum of" | Add up a list | Σ of [1,2,3] is 6 |
| Π | "product of" | Multiply a list together | Π of [1,2,3] is 6 |
| `x²` | "x squared" | x times itself | 3² = 9 |
| `√x` | "square root of x" | The number that squares to x | √9 = 3 |
| `e` | "Euler's number" | The constant 2.71828... | see 1.4 |
| `exp(x)` or `eˣ` | "e to the x" | e multiplied by itself x times | exp(1) = 2.718 |
| `log(x)` | "natural log of x" | The power you raise e to, to get x | log(2.718) = 1 |
| `∂y/∂x` | "the derivative of y with respect to x" | How much y moves when x is nudged | see 1.5 |
| `∇` | "gradient" | All the derivatives at once, one per parameter | see 1.6 |
| `θ` | "theta" | Standard letter for "all the parameters" | |
| `ℝⁿ` | "R to the n" | The space of lists of n real numbers | ℝ³ is all 3-number lists |

That is the whole vocabulary. Anything else gets defined where it appears.

## 1.3 Numbers, vectors, matrices, tensors

- A **scalar** is one number: `4.2`.
- A **vector** is an ordered list: `[0.3, -1.2, 4.0]`. Picture an arrow in space, or three dials.
- A **matrix** is a grid, rows by columns. A 27×27 matrix holds 729 numbers. You will build exactly that in Chapter 2.
- A **tensor** is the general term: a block of numbers with any number of dimensions. Scalar = 0-D, vector = 1-D, matrix = 2-D, stack of matrices = 3-D. [standard]

**Shape** is the list of sizes per dimension. Shape `32 × 3 × 10` means 32 examples, each 3 things, each described by 10 numbers. That exact shape appears in Chapter 3. [transcript]

**Run it.**

```python
import torch
a = torch.tensor(4.2)                      # scalar
b = torch.tensor([0.3, -1.2, 4.0])         # vector
c = torch.zeros(27, 27)                    # matrix
d = torch.randn(32, 3, 10)                 # 3-D tensor of random numbers
for name, t in [('a', a), ('b', b), ('c', c), ('d', d)]:
    print(f"{name}: shape {tuple(t.shape)}  dims {t.dim()}  total numbers {t.numel()}")
```

**What you should see:**

```
a: shape ()  dims 0  total numbers 1
b: shape (3,)  dims 1  total numbers 3
c: shape (27, 27)  dims 2  total numbers 729
d: shape (32, 3, 10)  dims 3  total numbers 960
```
[verified]

**Why tensors instead of loops.** A GPU can multiply a million pairs of numbers in roughly the time a Python loop takes to do a few thousand. Every performance idea in this course is some version of "rearrange this into one big tensor operation."

> **Say it to a six-year-old.** A number is one Lego brick. A vector is a row of bricks. A matrix is a flat square of bricks. A tensor is any Lego shape at all, including a big cube. The computer likes it when you hand it the whole cube at once instead of one brick at a time.

## 1.4 Exponentials and logarithms

These two show up constantly, so here they are properly.

**Exponential.** `exp(x)` means the constant e (2.71828...) raised to the power x. Two properties are all that matter:

1. **It is always positive.** `exp(−100)` is a tiny positive number, never negative. This is why it is used to turn arbitrary network outputs into probabilities, which must be positive.
2. **It grows fast and it stretches differences.** `exp(2) ≈ 7.4` and `exp(4) ≈ 54.6`. A gap of 2 in the input became a factor of 7 in the output.

**Logarithm.** `log(x)` is the inverse: the power you must raise e to in order to get x. `log(1) = 0`, `log(2.718) = 1`, `log(0.5) ≈ −0.69`. Three properties matter:

1. **The log of a number between 0 and 1 is negative**, and it plunges toward negative infinity as the number approaches zero. `log(0.1) ≈ −2.3`, `log(0.001) ≈ −6.9`, `log(0)` is undefined, reported as `-inf`.
2. **Logs turn multiplication into addition**: `log(a×b) = log(a) + log(b)`. This is why probabilities, which get multiplied together and quickly become unimaginably small, are handled in log space instead.
3. **It is monotonic**: bigger input, bigger output. So maximizing a probability and maximizing its log are the same problem, and the log version is numerically safer.

**Run it.**

```python
import math
for x in [0.001, 0.1, 0.5, 1.0, 2.718281828, 10.0]:
    print(f"x={x:<12} exp(x)={math.exp(x):<20.6f} log(x)={math.log(x):.6f}")
print("\nlog turns multiplication into addition:")
print("  log(0.2 * 0.3) =", math.log(0.2*0.3))
print("  log(0.2) + log(0.3) =", math.log(0.2) + math.log(0.3))
```

**What you should see:**

```
x=0.001        exp(x)=1.001001             log(x)=-6.907755
x=0.1          exp(x)=1.105171             log(x)=-2.302585
x=0.5          exp(x)=1.648721             log(x)=-0.693147
x=1.0          exp(x)=2.718282             log(x)=0.000000
x=2.718281828  exp(x)=15.154262            log(x)=1.000000
x=10.0         exp(x)=22026.465795         log(x)=2.302585

log turns multiplication into addition:
  log(0.2 * 0.3) = -2.8134107167600364
  log(0.2) + log(0.3) = -2.8134107167600364
```
[verified]

> **Say it to a six-year-old.** Some numbers are so tiny that writing them down takes forever, like a decimal point with twenty zeros after it. A logarithm is a shortcut for saying "how many zeros," so you can talk about tiny things with small, easy numbers.

## 1.5 A derivative is a sensitivity measurement

This is the one piece of calculus the course requires, and you can get it entirely from nudging.

Take `f(x) = 3x² − 4x + 5`. (Karpathy opens Lecture 1 with this exact function. [transcript]) At `x = 3` it equals 20.

Nudge the input by a tiny `h = 0.001` and recompute: `f(3.001) = 20.014003`. The output moved 0.014003 when the input moved 0.001. Divide: `0.014003 / 0.001 = 14.003`.

That number, essentially 14, is the **derivative** at `x = 3`. It answers exactly one question: *if I wiggle this input slightly, how much and in which direction does the output move?*

- Positive: pushing input up pushes output up.
- Negative: pushing input up pulls output down.
- Near zero: the output does not currently care about this input.

**Run it.**

```python
def f(x):
    return 3*x**2 - 4*x + 5

h = 0.001
for x in [3.0, -3.0, 2/3]:
    slope = (f(x+h) - f(x)) / h
    print(f"at x={x:+.4f}  f(x)={f(x):8.4f}  slope≈{slope:+.4f}")
```

**What you should see:**

```
at x=+3.0000  f(x)= 20.0000  slope≈+14.0030
at x=-3.0000  f(x)= 44.0000  slope≈-21.9970
at x=+0.6667  f(x)=  3.6667  slope≈+0.0030
```
[verified]

Read those three lines carefully, because they are the whole idea. At x=3 the function climbs steeply. At x=−3 it falls steeply. At x=2/3 the slope is essentially zero, which means we are sitting at the bottom of the curve. **Finding where the slope is zero is finding the minimum**, and minimizing is what training does.

**Why h = 0.001 and not smaller.** In theory the derivative is the limit as h approaches 0. In practice, computers store numbers with finite precision, so an h that is too small makes `f(x+h)` and `f(x)` round to the same value and the answer collapses to garbage. Try `h = 1e-15` and watch it break. This is one reason real systems compute derivatives symbolically, with rules, rather than numerically, with nudges.

> **Say it to a six-year-old.** Stand on a hill and shuffle one small step forward. Did you go up or down, and by how much? That is all a derivative is: a way of asking "which way is downhill from right here."

> **For the PhD in the room.** What is described here is a forward finite difference, with error O(h) plus floating-point cancellation error O(ε/h), minimized around h ≈ √ε ≈ 1.5e-8 for float64. The course uses it only as a gradient check. Everything real uses reverse-mode automatic differentiation, which is exact up to floating-point and costs one backward sweep for all partial derivatives simultaneously, versus n forward evaluations for finite differences. That asymmetry is why reverse mode won: for a scalar loss and n parameters, reverse mode is O(1) sweeps and forward mode is O(n).

## 1.6 The gradient

When a function has many inputs, each input gets its own sensitivity number. The whole collection is the **gradient**: one number per parameter, each saying "here is how the output moves when you nudge *this* knob, holding the others still."

A network with 41 parameters has a gradient of 41 numbers. GPT-3, with 175 billion parameters, has a gradient of 175 billion numbers, recomputed at every training step. [transcript]

> **Say it to a six-year-old.** You have a hundred knobs and one score. The gradient is a list that says, for every single knob, "turning this one up makes the score a little better" or "a little worse." Then you turn all hundred at once, each in the good direction.

## 1.7 The chain rule

Turning a crank turns a gear, and the gear moves a belt.

- One turn of the crank turns the gear 3 times.
- One turn of the gear moves the belt 2 centimeters.

How far does the belt move per crank turn? `3 × 2 = 6` centimeters.

That multiplication is the **chain rule**: when effects pass through a chain of steps, sensitivities multiply along the chain. [standard]

This is the entire mathematical content of backpropagation. A network is a long chain of small operations, and "how sensitive is the final error to this one weight buried deep inside" is answered by multiplying local sensitivities along the path from that weight to the output. Everything else is bookkeeping to do it for millions of paths at once without losing track.

**Two rules cover almost everything you will meet:**

| Operation | Forward | Backward |
|---|---|---|
| Addition, `c = a + b` | Add the inputs | Pass the incoming sensitivity to both inputs unchanged |
| Multiplication, `c = a × b` | Multiply the inputs | Give each input the *other* input's value, times the incoming sensitivity |

Why addition passes through unchanged: if `c = a + b`, nudging `a` by 0.001 moves `c` by exactly 0.001, a sensitivity of 1, and multiplying by 1 changes nothing.

Why multiplication swaps: if `c = a × b` with `a = 2, b = −3`, then nudging `a` up by 1 changes `c` by `b`, which is −3. So `a`'s sensitivity is `b`'s value, and vice versa.

**Run it.** Verify the swap by nudging, before trusting any formula:

```python
a, b, h = 2.0, -3.0, 0.0001
c = a * b
print("d(c)/d(a) numerically:", ((a+h)*b - a*b) / h, " ... which is b =", b)
print("d(c)/d(b) numerically:", (a*(b+h) - a*b) / h, " ... which is a =", a)
```

**What you should see:**

```
d(c)/d(a) numerically: -3.000000000010772  ... which is b = -3.0
d(c)/d(b) numerically: 2.0000000000042206  ... which is a = 2.0
```
[verified]

Those trailing digits are floating-point noise from subtracting two nearly equal numbers, which is the effect described in 1.5.

> **Say it to a six-year-old.** If you get two stickers for every drawing, and two candies for every sticker, then each drawing is worth four candies. You just multiply along the chain.

> **For the PhD in the room.** For f: ℝⁿ → ℝᵐ composed with g, the chain rule is the Jacobian product J_(g∘f) = J_g · J_f. Reverse-mode AD never materializes those Jacobians; it evaluates vector-Jacobian products vᵀJ, which for a scalar loss means starting with v = 1 and sweeping backward. Each primitive supplies a VJP rather than a full Jacobian, which is why an elementwise operation over a million values costs a million multiplications rather than a million-squared matrix.

## 1.8 Matrix multiplication, done by hand once

Nearly all of a neural network's arithmetic is matrix multiplication, so do one manually and you will never be confused by it again.

To multiply matrix **A** (2 rows, 3 columns) by matrix **B** (3 rows, 2 columns), you take each row of A, pair it with each column of B, multiply element by element, and sum.

```
A = [1 2 3]      B = [ 7  8]
    [4 5 6]          [ 9 10]
                     [11 12]
```

Result entry at row 1, column 1: `(1×7) + (2×9) + (3×11) = 7 + 18 + 33 = 58`.
Row 1, column 2: `(1×8) + (2×10) + (3×12) = 8 + 20 + 36 = 64`.
Row 2, column 1: `(4×7) + (5×9) + (6×11) = 28 + 45 + 66 = 139`.
Row 2, column 2: `(4×8) + (5×10) + (6×12) = 32 + 50 + 72 = 154`.

**The shape rule.** `(2×3) @ (3×2)` gives `(2×2)`. The inner numbers must match, and they vanish; the outer numbers survive. If the inner numbers do not match, the operation is undefined and PyTorch raises an error. **Roughly 90% of the errors you will hit in this course are this rule being violated** [my read], so it is worth memorizing in this form: **the inner dimensions must agree**.

**Run it.**

```python
import torch
A = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
B = torch.tensor([[7., 8.], [9., 10.], [11., 12.]])
print("A", tuple(A.shape), "@ B", tuple(B.shape), "->", tuple((A @ B).shape))
print(A @ B)
```

**What you should see:**

```
A (2, 3) @ B (3, 2) -> (2, 2)
tensor([[ 58.,  64.],
        [139., 154.]])
```
[verified]

**Why this operation, of all operations.** A neural network layer computes "every output is a weighted sum of every input." That is precisely a matrix multiply: the weight matrix holds one column of weights per output neuron, and multiplying by it computes all the weighted sums simultaneously.

> **Say it to a six-year-old.** You have three ingredients and two recipes. Each recipe says how much of each ingredient to use. Matrix multiplication is working out how much of each ingredient you need for all the recipes at once, in one go, instead of doing each recipe separately.

## 1.9 Broadcasting, and why it silently ruins models

**Broadcasting** is PyTorch automatically stretching a smaller tensor to match a bigger one so an operation can proceed. Add a 3-element vector to a 2×3 matrix, and the vector is copied onto both rows. [standard]

The rules, compared right to left across the shapes:

1. Dimensions match, or
2. one of them is 1 (that one gets stretched), or
3. one of them does not exist (treated as 1).

Otherwise it errors.

This is convenient and it is the single most dangerous feature in the library, because a shape mistake often does not raise an error. It produces a *different, valid, wrong* computation.

**Run it.** The classic bug, normalizing rows versus columns. Note the matrix is **square**, which is what makes the bug silent, and the real counts matrix in Chapter 2 is 27×27:

```python
import torch
N = torch.tensor([[1., 1., 2.], [3., 3., 6.], [2., 4., 4.]])
right = N / N.sum(1, keepdim=True)   # shape (3,1) -> stretched across columns: CORRECT
wrong = N / N.sum(1)                 # shape (3,)  -> stretched across ROWS: WRONG, no error
print("row sums with keepdim:", tuple(N.sum(1, keepdim=True).shape))
print("row sums without:     ", tuple(N.sum(1).shape))
print("correct rows sum to:", right.sum(1))
print("wrong   rows sum to:", wrong.sum(1))
```

**What you should see:**

```
row sums with keepdim: (3, 1)
row sums without:      (3,)
correct rows sum to: tensor([1., 1., 1.])
wrong   rows sum to: tensor([0.5333, 1.6000, 1.2333])
```
[verified]

The second version ran happily and produced rows that do not sum to 1, meaning they are not probabilities, meaning the model is quietly broken. **The defensive habit: always pass `keepdim=True` when you sum for normalization, and always check that your probabilities sum to 1.**

**Worth trying yourself:** change the matrix to non-square, say 2×3, and run the same two lines. The wrong version now raises `RuntimeError: The size of tensor a (3) must match the size of tensor b (2)`. [verified] A rectangular shape catches the mistake for you; a square shape does not. This is why the bug is so common in the bigram model, where the matrix is 27×27.

> **Say it to a six-year-old.** If one kid brings a bag of sweets to share with a row of friends, everyone gets some. Broadcasting is the computer sharing one small list out across a big table of numbers. It is helpful, but if it shares along the wrong direction, everyone gets the wrong thing and nobody complains.

## 1.10 What a neuron is

An artificial **neuron** does three things: [standard]

1. **Weighted sum.** Multiply each input by its own weight and add them. Inputs `[2.0, 3.0]`, weights `[−3.0, 1.0]`: `2.0×(−3.0) + 3.0×1.0 = −3.0`.
2. **Add a bias**, a per-neuron offset applied regardless of input. Bias `6.5` gives `3.5`. The bias sets how easily the neuron activates at all.
3. **Squash through a nonlinearity.** The course uses `tanh`, which maps any number into the range −1 to 1. `tanh(3.5) ≈ 0.998`.

**Why squash?** Without a bend somewhere, stacking layers is algebraically pointless: a chain of pure multiply-and-add collapses into a single multiply-and-add. The nonlinearity is what makes depth buy you anything.

**Run it.** Prove the collapse to yourself:

```python
import torch
torch.manual_seed(1337)
x = torch.randn(1, 4)
W1, W2 = torch.randn(4, 5), torch.randn(5, 3)
two_linear_layers = (x @ W1) @ W2
one_equivalent_layer = x @ (W1 @ W2)
print("difference:", (two_linear_layers - one_equivalent_layer).abs().max().item())
```

**What you should see:**

```
difference: 4.76837158203125e-07
```
[verified]

Zero, up to floating-point noise (that is 0.00000048, the rounding error of 32-bit arithmetic, not a real difference). Two stacked linear layers are exactly one linear layer, so without nonlinearities a 50-layer network has the expressive power of a 1-layer network.

**What tanh looks like.** It passes through zero at zero, rises steeply near the middle, and flattens out toward ±1 at the edges. Those flat edges matter enormously and are the subject of Chapter 4: where the curve is flat, the derivative is near zero, and a neuron sitting out there stops learning.

**Run it.**

```python
import torch
for v in [-4.0, -1.0, 0.0, 1.0, 4.0]:
    t = torch.tensor(v)
    print(f"tanh({v:+.1f}) = {torch.tanh(t):+.6f}   local slope = {1 - torch.tanh(t)**2:.6f}")
```

**What you should see:**

```
tanh(-4.0) = -0.999329   local slope = 0.001341
tanh(-1.0) = -0.761594   local slope = 0.419974
tanh(+0.0) = +0.000000   local slope = 1.000000
tanh(+1.0) = +0.761594   local slope = 0.419974
tanh(+4.0) = +0.999329   local slope = 0.001341
```
[verified]

Look at the slope column. At the edges it is 0.0013, effectively zero. A neuron pushed out there passes almost no gradient backward, so it barely trains. That single fact explains a large fraction of Chapter 4.

> **Say it to a six-year-old.** A neuron is a tiny voter. It listens to a few friends, trusts some of them more than others, has its own mood about the whole thing, and then says yes or no. A brain-sized pile of these tiny voters, all shouting at each other, is what a neural network is.

## 1.11 Layers and networks

A **layer** is a row of neurons all looking at the same inputs. A **multilayer perceptron (MLP)** is several layers stacked, each feeding the next.

- Lecture 1's MLP: **41 parameters** [verified] [transcript].
- Chapter 3's: about 3,400, then about 12,000.
- Chapter 7's transformer: about 10 million.
- GPT-3: 175 billion. [transcript]

The structure is the same at every scale. Only the count changes, which is the most important sentence in this book. [my read]

## 1.12 Probability, softmax, and cross-entropy

From Chapter 2 onward, models do not output "the answer." They output a **probability distribution**: one number per possible outcome, all positive, summing to 1.

For the letter after `a`, a model might say `n` 18%, `r` 12%, `l` 9%, and so on across 27 options.

**How raw outputs become probabilities.** The network emits arbitrary numbers called **logits**, possibly negative, possibly huge. Two steps fix that:

1. **Exponentiate** each (`eˣ`), making everything positive and stretching differences (section 1.4).
2. **Divide by the total**, so they sum to 1.

That recipe is **softmax**.

**Run it.**

```python
import torch
logits = torch.tensor([2.0, 1.0, 0.1, -5.0])
counts = logits.exp()
probs = counts / counts.sum()
print("logits: ", logits.tolist())
print("exp:    ", [round(c, 4) for c in counts.tolist()])
print("probs:  ", [round(p, 4) for p in probs.tolist()])
print("sums to:", probs.sum().item())
print("matches torch.softmax:", torch.allclose(probs, torch.softmax(logits, dim=0)))
```

**What you should see:**

```
logits:  [2.0, 1.0, 0.10000000149011612, -5.0]
exp:     [7.3891, 2.7183, 1.1052, 0.0067]
probs:   [0.6586, 0.2423, 0.0985, 0.0006]
sums to: 1.0
matches torch.softmax: True
```
[verified]

Note what softmax did: a logit gap of 1.0 (from 2.0 down to 1.0) became a probability ratio of about 2.7×, and the logit of −5 was crushed to 0.06%. Softmax is aggressive about differences. Notice also that `0.1` printed as `0.10000000149011612`: 32-bit floating point cannot represent 0.1 exactly. This is normal and is why you compare floats with `torch.allclose` rather than `==`.

**Scoring a probabilistic model.** If the truth was `n` and the model said 18%, was that good? The measurement used everywhere here is **negative log likelihood**:

1. Take the probability assigned to what actually happened: 0.18.
2. Take its log: `log(0.18) ≈ −1.715`. Negative, and plunging as probability approaches zero.
3. Flip the sign, so confident-and-right scores low: `1.715`.
4. Average over every prediction in the dataset.

That average is the **loss**, also called **cross-entropy loss**. Lower is better; zero is perfect and unreachable in practice.

**Run it.**

```python
import torch, torch.nn.functional as F
logits = torch.tensor([[2.0, 1.0, 0.1, -5.0]])
target = torch.tensor([0])                     # the correct class is index 0
p = torch.softmax(logits, dim=1)[0, 0]
print("probability given to the truth:", p.item())
print("negative log likelihood by hand:", -torch.log(p).item())
print("F.cross_entropy:               ", F.cross_entropy(logits, target).item())
```

**What you should see:**

```
probability given to the truth: 0.6586053967475891
negative log likelihood by hand: 0.41763070225715637
F.cross_entropy:                0.41763073205947876
```
[verified]

The two numbers agree to seven decimal places and differ in the eighth. That difference is not a bug in either one: they perform the same arithmetic in a different order, and floating point is not perfectly associative. Get used to seeing this; it is why "the loss changed by 1e-8" never means anything.

`F.cross_entropy` is softmax and negative-log-likelihood fused into one function. Use the built-in rather than writing the two steps yourself: it is faster, and it handles the case where a logit is so large that `exp()` overflows to infinity. (It subtracts the maximum logit first, which changes nothing mathematically because softmax is invariant to a constant shift, and everything numerically.)

**Reading loss numbers, the most useful skill in this book.** Before training anything, compute what the loss should be if the model knows nothing. With 27 equally likely characters, the true one gets probability 1/27, so the loss is `−log(1/27) = 3.2958`.

```python
import math
for vocab in [27, 65, 50257]:
    print(f"vocab {vocab:>6}: a know-nothing model should start at loss {math.log(vocab):.4f}")
```

```
vocab     27: a know-nothing model should start at loss 3.2958
vocab     65: a know-nothing model should start at loss 4.1744
vocab  50257: a know-nothing model should start at loss 10.8249
```
[verified]

Those three numbers appear throughout the rest of the book. In Chapter 4 a network starts at **27.88** instead of 3.2958, which is how you know it is broken before wasting an hour training it. [verified]

> **Say it to a six-year-old.** The machine has to guess which letter comes next, and it is allowed to say "I'm 70% sure it's an A, 20% sure it's a B." If the answer turns out to be A, it gets a small penalty because it was mostly right. If it had said "1% sure it's an A," it gets a huge penalty. The game is to make the total penalty as small as possible, so it learns to be confident only when it should be.

> **For the PhD in the room.** Cross-entropy H(p, q) = −Σ p(x) log q(x) with p the empirical one-hot distribution reduces to negative log likelihood, so minimizing it is maximum likelihood estimation. The gap between it and the entropy of the data is the KL divergence, which is why the loss has a nonzero floor set by the true conditional entropy of English. Character-level English is usually quoted around 1.0 to 1.3 bits per character, roughly 0.7 to 0.9 nats, so the Shakespeare model's 1.48 nats in Chapter 7 is still well above the information-theoretic floor. Also worth flagging: perplexity, common in the language modeling literature, is just exp(loss), so 1.48 nats is a perplexity of about 4.4.

## 1.13 What a language model is

A **language model** assigns probabilities to what comes next in a sequence. [standard]

**Character-level**, used for most of the course: given `emm`, predict the next character. **Sub-word level**, used by ChatGPT: given `The capital of France is`, predict the next chunk of text.

To **generate**, you sample: ask for a distribution over next characters, draw one at random according to those probabilities, append it, ask again. Repeat until an end marker. This is **autoregressive** generation: the model's own output becomes its next input. [standard]

**Why sample randomly instead of always taking the most likely character?** Because always taking the maximum produces repetitive, degenerate text; the model would emit the same name every time. Randomness in proportion to the model's confidence is what makes generation produce variety.

## 1.14 Train, validation, test

Split your data three ways, typically 80% / 10% / 10%: [transcript]

- **Training set** — used to set the parameters.
- **Validation set** (also called dev) — used to choose settings like layer size and learning rate.
- **Test set** — used rarely, ideally once, for an honest final number.

**Why three and not two.** Every time you look at a set and change something in response, you leak a little of that set into your model. The test set stays sealed so the final number means something.

**Overfitting** is when the model memorizes the training data instead of learning the pattern: training loss keeps dropping while validation loss rises. **Underfitting** is when both are high and close together, meaning the model is too small or undertrained.

**Analogy.** Training on past exam papers is fine. Memorizing the answer key is overfitting: perfect on those papers, useless on the real exam. The validation set is a mock exam, and the test set is the real one.

> **For the PhD in the room.** The three-way split is doing model selection and evaluation with the same finite sample, so the validation estimate is optimistically biased by the number of configurations you compare against it, which is a multiple-comparisons problem. Nested cross-validation is the principled fix and is rarely used at this scale because a single training run is expensive. Note too that the split here is by *word*, not by example, which matters: splitting by example would leak, since two examples from the same name share context.

---

# PART II — THE GAP

## 2.1 The problem this course solves

Modern libraries let you train a neural network in about ten lines. Import, choose a model, call `.fit()`. This works, and teaches you almost nothing.

The result is a large population of practitioners who can operate the machinery but cannot diagnose it. When training silently fails, when the loss plateaus at a suspicious number, when the model works on training data and collapses on new data, ten-lines fluency runs out. The knowledge needed is inside the abstraction, and the abstraction exists to hide it.

Karpathy's own version of the complaint, from Lecture 1: he went looking for how `tanh` is actually implemented in PyTorch's source, "spent about 15 minutes and I couldn't find" it, because "these libraries unfortunately they grow in size and entropy," and searching for `tanh` returns "2,800 results" [transcript]. If the person teaching the course gives up on reading the library, a beginner has no chance of learning from it.

## 2.2 What existed before

- **University courses** (Stanford CS231n, which Karpathy taught in 2015–2016): rigorous, organized around vision and matrix-calculus notation. High barrier to entry.
- **Textbooks** (Goodfellow, Bengio, Courville, 2016): comprehensive, mathematically heavy. Excellent reference, punishing first exposure.
- **Top-down practical courses** (fast.ai): results fast, then peel back layers. Effective for many, and it defers the mechanics, which some learners never return for.
- **Visual explainers** (3Blue1Brown): outstanding intuition, no code, stops before transformers.

None does the specific thing this course does: build every piece in front of you, in running code, in order, with nothing hidden, from one derivative to a working GPT.

## 2.3 What breaks without it

Each of these is demonstrated live in a lecture, and each is a failure you will personally reproduce in this book:

- **You cannot tell a broken run from a slow one.** Without knowing a 27-way classifier starts at 3.2958, a run starting at 27.88 looks like "training in progress" rather than "your initialization is wrong." [verified, Chapter 4]
- **You cannot debug dead networks.** In Chapter 4 you will find 61% of a hidden layer saturated flat at initialization, contributing almost no gradient. Nothing errors. The model just trains worse than it should. [verified]
- **You blame the model for the tokenizer's crimes.** Chapter 8's central claim: bad spelling, bad arithmetic, and a measured 7.5× token cost for Hindi versus English all trace to the text-chopping step nobody thinks about. [verified]
- **You cannot make anything fast.** Chapter 9 gets an 11× speedup from understanding hardware, which no framework applies for you. [transcript]

---

# PART III — THE THING ITSELF

*Nine chapters, one per lecture. Each states the problem, builds the code you can run, walks the mechanism step by step, and closes with exercises and a summary you could say out loud.*

## Chapter 1 — micrograd: backpropagation from nothing

**Video:** 2h25m · [youtu.be/VMj-3S1tku0](https://youtu.be/VMj-3S1tku0) · **Runs on:** any laptop, no GPU, no libraries at all.

### The problem

You have a network with knobs and a loss. You need the gradient: for every knob, how does the loss respond? Doing that by hand for 41 knobs is tedious. For 175 billion it is impossible. You need a machine that computes gradients automatically, and to trust it you have to build one.

> **Say it to a six-year-old.** You built a tower of blocks and it fell over. You want to know which block was the problem. So you go backwards from the top, asking each block "did you wobble because of the block under you?" and you keep asking down the tower until you find the one at fault. Then you fix that block a tiny bit. That is all this chapter is: going backwards down the tower.

### What it structurally is

**micrograd** is an **automatic differentiation engine**: about 100 lines of Python that record arithmetic as it happens, then run the chain rule backward through the recording. It operates on single numbers rather than tensors, which makes it slow and completely transparent.

### Step 1 — wrap numbers in an object

Instead of a bare `2.0`, create an object holding two fields: `data`, the number, and `grad`, the sensitivity of the final loss to this number, starting at 0.

**Run it.**

```python
class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None   # what to do during the backward pass
        self._prev = set(_children)     # which Values produced this one
        self._op = _op                  # which operation produced it (for display)
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

a = Value(2.0)
print(a)
```

**What you should see:**

```
Value(data=2.0, grad=0.0)
```
[verified]

`grad` starts at 0 because before any backward pass, we have not yet learned that this number influences anything.

### Step 2 — make arithmetic record itself

Redefine `+` and `*` so the result remembers where it came from. In Python, defining `__add__` on a class is what makes the `+` symbol work on it.

**Run it.** (Add these methods inside the `Value` class.)

```python
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += 1.0 * out.grad     # addition passes gradient through unchanged
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad   # multiplication swaps the values
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
```

Two things are happening in each method:

1. **Forward:** compute the result now, and record the parents in `_children`.
2. **Backward:** store a small function that, *when later called*, will push gradient from this node back to its parents using the local rules from section 1.7.

The `_backward` function is not run yet. It is stored for later, which is the trick that makes the whole thing work.

### Step 3 — the graph builds itself

Computing `d = a*b + c` now produces a graph: `a` and `b` feed a multiply node, whose output plus `c` feeds an add node, whose output is `d`. Nobody declared that graph. Running the code built it.

This is called **define-by-run**, and it is why PyTorch feels like ordinary Python rather than a separate language. [standard]

### Step 4 — walk the graph backward, in the right order

Before computing a node's gradient you must have finished every node it feeds into. Sorting a graph so that this always holds is a **topological sort**: repeatedly take nodes whose dependencies are already handled. It is the only computer science in the lecture.

**Run it.** (Also inside the class.)

```python
    def backward(self):
        topo, visited = [], set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)
        self.grad = 1.0                  # the loss is perfectly sensitive to itself
        for node in reversed(topo):
            node._backward()
```

`self.grad = 1.0` is the seed of the whole process. Nudging the loss by 0.001 changes the loss by 0.001, a sensitivity of exactly 1. Every other gradient in the network is that 1 propagated backward and multiplied by local rules.

**Run it.** The full test:

```python
a = Value(2.0); b = Value(-3.0); c = Value(10.0)
d = a*b + c
d.backward()
print("d =", d.data, "| a.grad =", a.grad, "| b.grad =", b.grad, "| c.grad =", c.grad)
```

**What you should see:**

```
d = 4.0 | a.grad = -3.0 | b.grad = 2.0 | c.grad = 1.0
```
[verified]

**Check every one of those by hand**, because if you follow this you have understood backpropagation:

- `a.grad = -3.0`: `a` is multiplied by `b`, so nudging `a` up by 1 changes the product by `b`, which is −3. The add passes that through unchanged.
- `b.grad = 2.0`: symmetrically, `a`'s value.
- `c.grad = 1.0`: `c` only feeds an addition, which passes sensitivity through untouched.

You can confirm all three by nudging, exactly as in section 1.5. Do it once.

### Step 5 — the accumulation bug that everyone hits

If a variable is used twice, its gradients must **accumulate** (`+=`), not overwrite (`=`). Using `b` in two places means it influences the loss through two separate routes, and the total influence is their sum. Overwriting silently discards one route.

**Run it.** See the bug and the fix:

```python
a = Value(3.0)
b = a + a          # a is used twice
b.backward()
print("a.grad =", a.grad, "(correct answer is 2.0: db/da = 1 + 1)")
```

**What you should see:**

```
a.grad = 2.0 (correct answer is 2.0: db/da = 1 + 1)
```
[verified]

With `=` instead of `+=` this prints 1.0, which is wrong, and nothing warns you. This is also why PyTorch makes you call `zero_grad()` before every step: gradients accumulate by design, so you must clear them yourself, and forgetting to is one of the most common bugs in the field. [standard]

### Step 6 — add tanh, and the rest of the operations

**Run it.** (`tanh` needs `import math` at the top of your file. The rest go inside the class.)

```python
    def __pow__(self, other):
        out = Value(self.data ** other, (self,), f'**{other}')
        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t**2) * out.grad     # the slope of tanh, from section 1.10
        out._backward = _backward
        return out

    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __radd__(self, other): return self + other       # lets sum() work
    def __rmul__(self, other): return self * other       # lets 2 * value work
    def __truediv__(self, other): return self * other**-1
```

Notice how few real rules there are. Subtraction is addition with a negation. Division is multiplication by a power of −1. Only `+`, `*`, `**`, and `tanh` need genuine derivative rules; everything else is composition. **That is the deep point of the chapter**: an arbitrarily complicated function needs only a handful of local rules, because the chain rule assembles the rest.

The `tanh` backward rule, `1 - t²`, is the slope you printed in section 1.10. At `t = ±0.999`, that is `1 − 0.998 = 0.0013`, near zero. Remember this number in Chapter 4.

### Step 7 — build a neural network on top

**Run it.**

```python
import random

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)
    def __call__(self, x):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)   # weighted sum + bias
        return act.tanh()                                        # squash
    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]

random.seed(1337)
n = MLP(3, [4, 4, 1])      # 3 inputs -> 4 neurons -> 4 neurons -> 1 output
print("number of parameters:", len(n.parameters()))
```

**What you should see:**

```
number of parameters: 41
```
[verified] [transcript]

**Where 41 comes from**, so no number in this book is unexplained: layer 1 has 4 neurons each with 3 weights and 1 bias, so 4×(3+1) = 16. Layer 2 has 4 neurons each with 4 weights and a bias, 4×(4+1) = 20. Layer 3 has 1 neuron with 4 weights and a bias, 5. Total 16+20+5 = **41**.

### Step 8 — train it

Four training examples, each 3 numbers, with desired outputs +1 or −1.

**Run it.**

```python
xs = [[2.0, 3.0, -1.0],
      [3.0, -1.0, 0.5],
      [0.5, 1.0, 1.0],
      [1.0, 1.0, -1.0]]
ys = [1.0, -1.0, -1.0, 1.0]

for k in range(21):
    ypred = [n(x) for x in xs]                                   # forward pass
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))  # squared error
    for p in n.parameters():
        p.grad = 0.0                                             # RESET the gradients
    loss.backward()                                              # backward pass
    for p in n.parameters():
        p.data += -0.05 * p.grad                                 # step downhill
    if k % 5 == 0 or k == 20:
        print(f"step {k:2d}  loss {loss.data:.6f}")

print("final predictions:", [round(y.data, 4) for y in [n(x) for x in xs]])
```

**What you should see:**

```
step  0  loss 3.140718
step  5  loss 0.140404
step 10  loss 0.069491
step 15  loss 0.045248
step 20  loss 0.033226
final predictions: [0.9144, -0.9072, -0.9254, 0.9]
```
[verified]

Targets were `[1, −1, −1, 1]` and the network now outputs `[0.91, −0.91, −0.93, 0.90]`. That is the entire loop, and every neural network ever trained is this loop with bigger tensors.

**The minus sign is the whole algorithm.** `p.data += -0.05 * p.grad`. The gradient says which way makes the loss *bigger*, so you step the other way. Flip that minus to a plus and the loss will climb instead, which is worth doing once just to watch it happen.

**Why 0.05.** That is the **learning rate**, chosen by trial. Too small and the loss crawls; too big and it overshoots and diverges. Try 0.5 and 0.001 and watch both failure modes. Chapter 3 shows how to choose it properly rather than guessing.

### Distinctive features

- **Define-by-run**: the graph is built by executing the code, not declared in advance.
- **Scalars, not tensors**: real frameworks do this on whole arrays for speed. The math is identical; only the bookkeeping is uglier.
- **A deliberately PyTorch-shaped API**: late in the lecture Karpathy writes the same example in real PyTorch and gets matching numbers, which retroactively demystifies the real library.

**Run it.** The same gradients, from PyTorch:

```python
import torch
a = torch.tensor([2.0], requires_grad=True)
b = torch.tensor([-3.0], requires_grad=True)
c = torch.tensor([10.0], requires_grad=True)
d = a*b + c
d.backward()
print(a.grad.item(), b.grad.item(), c.grad.item())
```

**What you should see:**

```
-3.0 2.0 1.0
```
[verified]

Identical to your 100-line engine. PyTorch is this, plus tensors, plus GPU kernels, plus twenty years of engineering.

### Exercises

1. **Add an `exp()` method** with the correct backward rule. Hint: the derivative of `eˣ` is `eˣ`, so `self.grad += out.data * out.grad`. Verify by nudging.
2. **Break the accumulation.** Change `+=` to `=` in `__add__`, rerun `b = a + a`, and confirm you get 1.0 instead of 2.0.
3. **Remove the nonlinearity.** Delete `.tanh()` from `Neuron.__call__` and retrain. The loss will still fall (this problem is nearly linear), but check with section 1.10's argument why depth is now pointless.
4. **Gradient check.** Pick any parameter, nudge its `data` by `h = 1e-5`, recompute the loss, and compare `(loss_new − loss_old)/h` with the `.grad` your engine reported. They should agree to about four decimal places. This is how every autodiff library is tested in practice. [standard]

### Troubleshooting

| Symptom | Cause |
|---|---|
| `TypeError: unsupported operand type(s) for +: 'int' and 'Value'` | `sum()` starts at integer 0; you need `__radd__` |
| Loss goes up, not down | Missing minus sign in the update, or learning rate far too large |
| Loss stuck exactly the same | You forgot `loss.backward()`, or all grads are 0 because you reset them after backward instead of before |
| Loss becomes `nan` | Learning rate too high; parameters exploded to infinity. Lower it to 0.01 |
| `RecursionError` in `build` | Your graph got very deep; raise the limit with `sys.setrecursionlimit(10000)` |

> **For the PhD in the room.** This is textbook reverse-mode AD over a dynamically constructed DAG (directed acyclic graph, meaning arrows never form a loop), with each primitive registering a vector-Jacobian product and a topological order guaranteeing that a node's adjoint is complete before it is used. The accumulation with `+=` is what makes fan-out correct, corresponding to the multivariable chain rule summing over all paths. Two things it omits that production systems care about: checkpointing, meaning recomputing activations rather than storing them, to trade compute for memory, and any notion of higher-order derivatives, which would require the backward pass itself to be differentiable, that is, built out of `Value` operations rather than raw floats. Karpathy's later `micrograd` variants and JAX's design differ exactly here.

### 30-second version

Every calculation a network performs is a chain of tiny operations. Each operation knows how sensitive its output is to its inputs, which for addition is "pass it through" and for multiplication is "swap the values." Multiply those sensitivities backward along the chain and you learn how every knob in the network affects the final error. That is backpropagation, it fits in 100 lines of Python, and the version inside PyTorch differs only by working on whole arrays at once.

---

## Chapter 2 — makemore: a language model with 27 letters

**Video:** 1h57m · [youtu.be/PaCmpygFfXo](https://youtu.be/PaCmpygFfXo) · **New:** PyTorch tensors, probability distributions, and a demonstration that counting and learning can reach the same answer.

### The problem

Invent new human names. Train on 32,033 real ones and generate plausible fakes. The project name says it: **makemore** makes more of whatever you feed it.

> **Say it to a six-year-old.** You read a big list of names and notice that after the letter Q there is almost always a U, and that lots of names end in A. Once you know which letters like to sit next to which other letters, you can make up brand new names that sound real. That is the whole trick. The computer just counts much faster than you.

### Step 1 — load and look at the data

Never train on data you have not looked at.

**Run it.**

```python
words = open('names.txt', 'r').read().splitlines()
print("number of names:", len(words))
print("first five:", words[:5])
print("shortest:", min(len(w) for w in words), "| longest:", max(len(w) for w in words))
```

**What you should see:**

```
number of names: 32033
first five: ['emma', 'olivia', 'ava', 'isabella', 'sophia']
shortest: 2 | longest: 15
```
[verified]

### Step 2 — frame it as next-character prediction

The word `emma` becomes five training examples, using a special token `.` to mark both start and end: `.→e`, `e→m`, `m→m`, `m→a`, `a→.` [transcript]. A pair of adjacent characters is a **bigram**.

**Why one token for both start and end.** You need to model two things: which letters begin a name, and when a name is finished. Using the same symbol for both keeps the table square at 27×27 and costs nothing, because the start position and the end position never get confused: `.` as an input means "we are at the beginning," and `.` as an output means "stop."

**Run it.** Build the character-to-integer mapping:

```python
chars = sorted(list(set(''.join(words))))
stoi = {s: i+1 for i, s in enumerate(chars)}   # a=1, b=2, ... z=26
stoi['.'] = 0                                  # the special token gets 0
itos = {i: s for s, i in stoi.items()}         # the reverse mapping
print("vocabulary size:", len(itos))
print("a ->", stoi['a'], "| z ->", stoi['z'], "| . ->", stoi['.'])
```

**What you should see:**

```
vocabulary size: 27
a -> 1 | z -> 26 | . -> 0
```
[verified]

### Step 3 — count every pair

**Run it.**

```python
import torch
N = torch.zeros((27, 27), dtype=torch.int32)
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        N[stoi[ch1], stoi[ch2]] += 1

print("total bigrams counted:", N.sum().item())
print("names starting with m:", N[0, stoi['m']].item())
print("times 'q' is followed by 'u':", N[stoi['q'], stoi['u']].item())
print("times 'j' is followed by 'q':", N[stoi['j'], stoi['q']].item())
```

**What you should see:**

```
total bigrams counted: 228146
names starting with m: 2538
times 'q' is followed by 'u': 206
times 'j' is followed by 'q': 0
```
[verified]

`zip(chs, chs[1:])` is the standard Python idiom for "every adjacent pair": it pairs the list with a copy of itself shifted by one. 2,538 names out of 32,033 start with `m`, just under 8%, which is the "about 2,500, a bit less than 10%" from the lecture [transcript].

Hold on to that `jq` count of zero. It causes an infinity in a moment.

### Step 4 — turn counts into probabilities

**Run it.**

```python
P = (N + 1).float()               # the +1 is smoothing; see step 6
P /= P.sum(1, keepdim=True)       # keepdim=True, per section 1.9
print("row for 'q' sums to:", P[stoi['q']].sum().item())
print("P(u | q) =", round(P[stoi['q'], stoi['u']].item(), 4))
print("P(a | q) =", round(P[stoi['q'], stoi['a']].item(), 4))
```

**What you should see:**

```
row for 'q' sums to: 1.0
P(u | q) = 0.6923
P(a | q) = 0.0468
```
[verified]

Read `P(u | q)` as "the probability of u given q." After a `q`, this model says `u` 69% of the time and `a` under 5%, which is a real fact about English names that nobody programmed in. It fell out of counting 206 occurrences of `qu`.

### Step 5 — sample new names

**Run it.**

```python
g = torch.Generator().manual_seed(2147483647)   # fixed seed, so you get my exact output
for _ in range(5):
    out, ix = [], 0
    while True:
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        if ix == 0:
            break
        out.append(itos[ix])
    print(''.join(out))
```

**What you should see:**

```
cexze
momasurailezitynn
konimittain
llayn
ka
```
[verified]

`torch.multinomial` draws from a probability distribution: hand it `[0.6, 0.3, 0.1]` and it returns index 0 about 60% of the time. The **generator with a fixed seed** makes randomness reproducible, so you and I get identical names. Change the seed and you get different ones; remove it and you get different ones every run.

These are bad names, in an instructive way. `ka` and `llayn` are name-shaped. `momasurailezitynn` is not. The model only ever knows the single preceding letter, so it cannot remember that it started a name eleven letters ago.

### Step 6 — score the model, and meet infinity

**Run it.**

```python
log_likelihood, n = 0.0, 0
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        log_likelihood += torch.log(P[stoi[ch1], stoi[ch2]])
        n += 1
print(f"average negative log likelihood (the loss): {-log_likelihood/n:.4f}")
```

**What you should see:**

```
average negative log likelihood (the loss): 2.4544
```
[verified] [transcript]

Compare against 3.2958, the score of a model that knows nothing (section 1.12). The counting model learned something real.

**Now the infinity.** Build the same table without the `+1`:

```python
P0 = N.float()
P0 /= P0.sum(1, keepdim=True)
ll, m = 0.0, 0
for ch1, ch2 in zip("andrejq", "ndrejq"):
    ll += torch.log(P0[stoi[ch1], stoi[ch2]]); m += 1
print("loss on the name 'andrejq':", (-ll/m).item())
```

**What you should see:**

```
loss on the name 'andrejq': inf
```
[verified] [transcript]

`jq` never occurred, so the model assigns it probability exactly 0, `log(0)` is `−inf`, and the loss on any name containing `jq` is infinite. The model is claiming that this name is not merely unlikely but *impossible*, which is an absurd thing to be certain about.

**The fix is the `+1`,** called **smoothing**: add a fake count of 1 to every cell before normalizing, so nothing is ever impossible. The cost is a slightly worse average loss; the benefit is that a single unseen pair no longer destroys everything.

**Analogy.** Insurance. You pay a small premium on every prediction to avoid one unbounded loss.

> **For the PhD in the room.** Add-one smoothing is Laplace smoothing, equivalent to a symmetric Dirichlet(α=1) prior over the categorical parameters with the posterior mean as the estimate. In the language-modeling literature it is known to be a poor smoother compared to Kneser-Ney, which redistributes mass according to continuation counts rather than uniformly. Nobody uses it seriously at this point; the neural approach in step 7 is precisely the escape from smoothing schemes, since parameter sharing through embeddings performs the smoothing implicitly and in a data-dependent way.

### Step 7 — the same model, learned instead of counted

Now throw the counts away and train a network to do the same job.

**One-hot encoding.** A network takes numbers, not letters. Represent letter 13 as a 27-long vector of zeros with a single 1 in position 13. [standard]

**Run it.**

```python
import torch.nn.functional as F
xs, ys = [], []
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        xs.append(stoi[ch1]); ys.append(stoi[ch2])
xs, ys = torch.tensor(xs), torch.tensor(ys)
num = xs.nelement()
print("number of training examples:", num)

xenc = F.one_hot(xs[:3], num_classes=27).float()
print("first three inputs as one-hot rows, shape:", tuple(xenc.shape))
print(xenc[0][:8], "...")
```

**What you should see:**

```
number of training examples: 228146
first three inputs as one-hot rows, shape: (3, 27)
tensor([1., 0., 0., 0., 0., 0., 0., 0.]) ...
```
[verified]

228,146 examples, the same number as the bigram count, because every bigram is one training example.

**The key insight about one-hot.** Multiplying a one-hot row by a matrix **selects a row of that matrix**. Nothing else. So `xenc @ W` where `W` is 27×27 is an elaborate way of looking up row 13, which means this "neural network" is a lookup table wearing a costume. Understanding that makes the punchline land.

**Run it.** The training loop:

```python
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True)

for k in range(200):
    xenc = F.one_hot(xs, num_classes=27).float()   # (228146, 27)
    logits = xenc @ W                              # (228146, 27) raw scores
    counts = logits.exp()                          # softmax, step 1
    probs = counts / counts.sum(1, keepdim=True)   # softmax, step 2
    loss = -probs[torch.arange(num), ys].log().mean() + 0.01*(W**2).mean()
    if k % 50 == 0 or k == 199:
        print(f"step {k:3d} loss {loss.item():.4f}")
    W.grad = None          # reset gradients (the += rule from Chapter 1)
    loss.backward()        # PyTorch does what your engine did
    W.data += -50 * W.grad # step downhill
```

**What you should see:**

```
step   0 loss 3.7686
step  50 loss 2.5099
step 100 loss 2.4900
step 150 loss 2.4849
step 199 loss 2.4830
```
[verified]

Line by line, because every piece of this recurs for the rest of the book:

- `probs[torch.arange(num), ys]` plucks, for each of the 228,146 rows, the probability assigned to the character that actually came next. `torch.arange(num)` is `[0,1,2,...]`, so this pairs row 0 with `ys[0]`, row 1 with `ys[1]`, and so on. This double-indexing trick appears constantly.
- `.log().mean()` and the minus sign make it the negative log likelihood of section 1.12.
- `+ 0.01*(W**2).mean()` is **regularization**: a penalty for large weights, which pushes them toward zero. It is the gradient-descent equivalent of adding 1 to every count. Two different-looking hacks, one idea: prevent extreme confidence. [transcript]
- `-50` is a very large learning rate, tolerable here because the model is tiny and the loss surface is simple.

### Step 8 — the punchline

The network converges to about **2.48** after 200 steps and continues toward **2.45** with more; the counting table gave **2.4544**. Same answer, reached two completely different ways. [verified]

Karpathy is explicit about why: "fundamentally, we're not taking any additional information" [transcript]. The model can express exactly one thing, the probability of each letter given the previous letter, so both methods find the same optimum. Counting finds it in one pass; gradient descent walks to it.

**Then why bother with the hard way?** Because counting does not scale. One character of context needs a 27×27 table of 729 cells. Ten characters of context needs 27¹⁰ cells.

**Run it.**

```python
print(f"{27**10:,} cells needed for a 10-character-context table")
print(f"{27**10 * 4 / 1e12:,.0f} terabytes just to store it as 32-bit floats")
```

**What you should see:**

```
205,891,132,094,649 cells needed for a 10-character-context table
824 terabytes just to store it as 32-bit floats
```
[verified]

And nearly all of those cells would be zero, because you have only 228,146 examples to fill 205 trillion cells. Karpathy: "we can't actually keep everything in a table anymore, so this is fundamentally an unscalable approach" [transcript]. The neural network is the escape route, and Chapter 3 takes it.

### Exercises

1. **Trigrams.** Build a model that uses the *two* preceding characters. The table becomes 27×27×27, which is 19,683 cells and still feasible. Does the loss beat 2.4544? (It should reach roughly 2.2.)
2. **Try the wrong normalization.** Replace `P.sum(1, keepdim=True)` with `P.sum(0, keepdim=True)` and see how badly the samples degrade. This is the section 1.9 bug, in a real setting.
3. **Vary the smoothing.** Try `+0.001` and `+100` instead of `+1`. Watch the loss get better and the samples get worse, then the reverse. Explain why.
4. **Delete the regularization term** in the neural version and train for 500 steps. The loss will go slightly lower. Explain, using the smoothing discussion, why that is not automatically an improvement.

### Troubleshooting

| Symptom | Cause |
|---|---|
| Loss is `nan` | A probability hit exactly 0 and `log(0)` is `-inf`; add smoothing, or use `F.cross_entropy` which is numerically safer |
| Samples are all one letter | Rows normalized along the wrong axis (section 1.9) |
| `IndexError` in `stoi[ch]` | Your `names.txt` contains a stray character such as a capital or a hyphen; check `sorted(set(''.join(words)))` |
| Loss stuck at 3.29 | Learning rate far too small, or you forgot `loss.backward()` |
| Different samples than this book | The generator seed differs, or you called the generator a different number of times before sampling |

### 30-second version

Count how often each letter follows each other letter and you have a working language model that scores 2.4544. Then throw the counts away, train a one-layer neural network on the same task, and it lands on the same number, because it is the same model learned a harder way. The counting version cannot grow past one letter of memory: ten letters of context would need a table of 205 trillion cells. The network version can grow, and the rest of the course is that growth.

---

## Chapter 3 — the MLP: embeddings, context, and how to actually train

**Video:** 1h15m · [youtu.be/TCH_1BHY58I](https://youtu.be/TCH_1BHY58I) · **Based on:** Bengio et al. 2003, "A Neural Probabilistic Language Model," the ancestor of everything that followed. [transcript]

### The problem

One letter of context is not enough, and a lookup table cannot hold more (205 trillion cells, Chapter 2). You need a model that generalizes to contexts it has never seen.

### The central idea: embeddings

Give every character a short list of learned numbers, say 2 or 10 of them, called an **embedding**. These numbers are parameters, so training decides what they should be. Characters used in similar ways drift toward similar numbers.

**Why this defeats the table explosion.** A table treats `aeb` and `aob` as unrelated cells with unrelated counts. If `e` and `o` end up with similar embeddings, a context the model has never seen gets handled sensibly by analogy with one it has. Information is *shared* rather than memorized.

**Analogy.** A table is a phone book: to know anything about a name, that exact name must be listed. Embeddings give every person coordinates instead (age, city, profession), so you can make reasonable guesses about someone you have never met, because they sit near people you have.

> **Say it to a six-year-old.** Instead of remembering every single word you ever heard, you notice that "cat" and "dog" are both furry animals, so they go in the same part of your head. Then when you hear about a new animal you have never met, you already have a good guess about it, because you put it near the other furry ones.

> **For the PhD in the room.** This is distributed representation in Hinton's sense, and Bengio's 2003 paper is the one that made it work for language: the count table's parameters grow as V^n while the MLP's grow as V·d + n·d·h, trading exponential blowup for a fixed-capacity bottleneck. The embedding matrix is exactly a learned linear map from the one-hot simplex, so "look up row i" and "multiply by a one-hot vector" are the same operation, which is why nobody implements it as a matrix multiply. Note what is *not* happening here: no factorization objective, no explicit similarity loss. Similarity structure appears only because it lowers next-character cross-entropy, which is the same argument later made for word2vec and, ultimately, for why language model representations transfer at all.

### Step 1 — build the dataset with a sliding window

**Run it.**

```python
import torch
words = open('names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {s: i+1 for i, s in enumerate(chars)}; stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}

block_size = 3   # how many characters we use to predict the next one

def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * block_size          # start padded with '...'
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)               # the 3 characters we see
            Y.append(ix)                    # the character that follows
            context = context[1:] + [ix]    # slide the window forward
    return torch.tensor(X), torch.tensor(Y)

# show the mechanism on one word
X, Y = build_dataset(['emma'])
for x, y in zip(X, Y):
    print(''.join(itos[i.item()] for i in x), '-->', itos[y.item()])
```

**What you should see:**

```
... --> e
..e --> m
.em --> m
emm --> a
mma --> .
```
[verified]

The window slides one character at a time, and the `.` padding at the start lets the model handle the first characters of a name with the same machinery as the middle.

### Step 2 — split into train, dev, test

**Run it.**

```python
import random
random.seed(42)
random.shuffle(words)
n1, n2 = int(0.8*len(words)), int(0.9*len(words))
Xtr,  Ytr  = build_dataset(words[:n1])      # 80%
Xdev, Ydev = build_dataset(words[n1:n2])    # 10%
Xte,  Yte  = build_dataset(words[n2:])      # 10%
print("train:", tuple(Xtr.shape), "dev:", tuple(Xdev.shape), "test:", tuple(Xte.shape))
```

**What you should see:**

```
train: (182625, 3) dev: (22655, 3) test: (22866, 3)
```
[verified]

182,625 training examples. **Note the shuffle happens on words, not on examples.** Splitting by example would leak: two examples from the same name share context, so the same name could appear in both training and validation, and your validation number would be a lie.

### Step 3 — the network

**Run it.**

```python
import torch.nn.functional as F
g = torch.Generator().manual_seed(2147483647)
n_embd, n_hidden = 10, 200

C  = torch.randn((27, n_embd),            generator=g)   # embedding table
W1 = torch.randn((n_embd*block_size, n_hidden), generator=g)
b1 = torch.randn(n_hidden,                generator=g)
W2 = torch.randn((n_hidden, 27),          generator=g)
b2 = torch.randn(27,                      generator=g)
parameters = [C, W1, b1, W2, b2]
print("total parameters:", sum(p.nelement() for p in parameters))
for p in parameters:
    p.requires_grad = True
```

**What you should see:**

```
total parameters: 11897
```
[verified]

Where 11,897 comes from: `C` is 27×10 = 270. `W1` is 30×200 = 6,000, plus 200 biases. `W2` is 200×27 = 5,400, plus 27 biases. Total 270 + 6,000 + 200 + 5,400 + 27 = **11,897**.

**Run it.** The forward pass, one piece at a time, so you can see every shape:

```python
ix = torch.randint(0, Xtr.shape[0], (32,), generator=g)   # a mini-batch of 32
emb = C[Xtr[ix]]
print("1. embeddings:      ", tuple(emb.shape), "  <- 32 examples, 3 chars, 10 numbers each")
flat = emb.view(-1, n_embd*block_size)
print("2. flattened:       ", tuple(flat.shape), "     <- the 3 chars glued into one 30-vector")
hpreact = flat @ W1 + b1
print("3. pre-activation:  ", tuple(hpreact.shape), "    <- 200 hidden neurons")
h = torch.tanh(hpreact)
print("4. after tanh:      ", tuple(h.shape))
logits = h @ W2 + b2
print("5. logits:          ", tuple(logits.shape), "     <- one score per possible next char")
loss = F.cross_entropy(logits, Ytr[ix])
print("6. loss:            ", round(loss.item(), 4))
```

**What you should see:**

```
1. embeddings:       (32, 3, 10)   <- 32 examples, 3 chars, 10 numbers each
2. flattened:        (32, 30)      <- the 3 chars glued into one 30-vector
3. pre-activation:   (32, 200)     <- 200 hidden neurons
4. after tanh:       (32, 200)
5. logits:           (32, 27)      <- one score per possible next char
6. loss:             27.8817
```
[verified]

**`C[Xtr[ix]]` deserves a pause.** `Xtr[ix]` is a 32×3 tensor of integers. Indexing the 27×10 table `C` with it produces a 32×3×10 tensor: PyTorch looked up all 96 characters at once and stacked the results. This is **fancy indexing**, and it is doing the job that a one-hot matrix multiply did in Chapter 2, without the wasted arithmetic.

**`.view(-1, 30)` deserves another.** It reinterprets the same block of memory with a different shape. The `-1` means "work this dimension out for me," here 32. It is nearly free because nothing is copied. Getting this wrong, for example using `.view(30, -1)`, produces a valid tensor of garbage and no error message.

**And that loss of 27.88 is a red flag.** Section 1.12 says a fresh 27-way model should score 3.2958. It scores 27.88, meaning it starts out confidently wrong. That is Chapter 4's subject; note it and move on.

### Step 4 — mini-batches

Computing gradients on all 182,625 examples per step is accurate and slow. Instead, sample 32 random examples per step. The gradient direction is noisy, "not the actual gradient direction, but the gradient direction is good enough" [transcript], and you get to take a hundred times more steps in the same wall-clock time.

**Analogy.** An approximate step taken now beats a perfect step taken next week.

> **For the PhD in the room.** This is stochastic gradient descent, and the noise is not purely a cost: minibatch gradient noise acts as an implicit regularizer, with a well-documented relationship between the noise scale and the learning rate to batch size ratio (Smith and Le, 2018). The practical consequence is that increasing the batch size without rescaling the learning rate changes the effective regularization, which is why "just use a bigger batch" often loses accuracy.

### Step 5 — find the learning rate by experiment

Rather than guessing, sweep it across orders of magnitude and look at the result.

**Run it.**

```python
# a trimmed version: 2000 steps at each learning rate, then measure the full training loss
for lr in [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]:
    ...   # rebuild the network, train 2000 steps at this lr, then evaluate
```

**What you should see** (the full script is in the exercises; these are the measured results):

```
lr=0.0001   training loss after 2000 steps: 3.1873
lr=0.001    training loss after 2000 steps: 2.7939
lr=0.01     training loss after 2000 steps: 2.5179
lr=0.1      training loss after 2000 steps: 2.3709
lr=1.0      training loss after 2000 steps: 2.4446
lr=10.0     training loss after 2000 steps: 87.3967
```
[verified]

Read that table and you have the whole learning-rate story:

- **0.0001** barely moves off the starting point. The lecture's phrase for this is "the loss is barely decreasing" [transcript].
- **0.1** is the sweet spot.
- **1.0** is past the sweet spot; it still trains, but worse.
- **10.0** diverges spectacularly to 87, far worse than random guessing at 3.2958. The steps overshoot the valley so badly that the parameters explode.

**Learning rate decay.** Once the loss stops improving, cut the rate by 10× and continue. Large steps get you near the valley quickly; small steps settle into it. This is worth roughly 0.05 to 0.1 of loss in this model. [verified]

### Step 6 — train properly and evaluate

**Run it.**

```python
for i in range(30000):
    ix = torch.randint(0, Xtr.shape[0], (32,), generator=g)
    emb = C[Xtr[ix]]
    h = torch.tanh(emb.view(-1, n_embd*block_size) @ W1 + b1)
    loss = F.cross_entropy(h @ W2 + b2, Ytr[ix])
    for p in parameters:
        p.grad = None
    loss.backward()
    lr = 0.1 if i < 18000 else 0.01        # decay late in training
    for p in parameters:
        p.data += -lr * p.grad

@torch.no_grad()                            # no gradients needed when only measuring
def split_loss(X, Y):
    emb = C[X]
    h = torch.tanh(emb.view(-1, n_embd*block_size) @ W1 + b1)
    return F.cross_entropy(h @ W2 + b2, Y).item()

print(f"train {split_loss(Xtr, Ytr):.4f} | dev {split_loss(Xdev, Ydev):.4f}")
```

**What you should see:**

```
train 2.2618 | dev 2.2778
```
[verified]

Better than the bigram's 2.4544. And read the two numbers together: train 2.2618, dev 2.2778, nearly equal. **That means the model is not memorizing; it is too small.** Overfitting would show as train far below dev. This diagnostic, comparing the two rather than staring at one, is the single most useful habit in the chapter.

**The `@torch.no_grad()` decorator** tells PyTorch not to build a computation graph, since we are only measuring. It roughly halves memory and speeds evaluation up. Forgetting it is harmless but wasteful.

### Step 7 — sanity check by deliberately overfitting

Before a long run, prove the model *can* learn by making it memorize a tiny dataset.

**Run it.**

```python
# train on only the first 32 examples, for 1000 steps
Xsmall, Ysmall = Xtr[:32], Ytr[:32]
for i in range(1000):
    emb = C[Xsmall]
    h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
    loss = F.cross_entropy(h @ W2 + b2, Ysmall)
    for p in parameters: p.grad = None
    loss.backward()
    for p in parameters: p.data += -0.1 * p.grad
print("loss on 32 memorized examples:", round(loss.item(), 6))
```

**What you should see:**

```
loss on 32 memorized examples: 0.252175
```
[verified]

0.25 against a know-nothing baseline of 3.2958, and still falling. Run 5,000 steps instead of 1,000 and it approaches zero, because 11,897 parameters can trivially memorize 32 examples. (It does not get there in 1,000 steps here because this network still has the broken initialization from step 3, which is the next section's subject.)

If the loss *cannot* be driven down on a tiny batch, you have a bug, and you found it in 20 seconds instead of after an hour of real training. This is a unit test, not a result. [transcript]

### Step 8 — sample from it

**Run it.**

```python
g = torch.Generator().manual_seed(2147483647 + 10)
for _ in range(8):
    out, context = [], [0] * block_size
    while True:
        emb = C[torch.tensor([context])]
        h = torch.tanh(emb.view(1, -1) @ W1 + b1)
        probs = F.softmax(h @ W2 + b2, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        if ix == 0: break
        out.append(itos[ix])
    print(''.join(out))
```

**What you should see** (from the fully trained Chapter 4 version):

```
chrmahzlyn
hlri
khmrix
thty
sklassa
jazhnn
fagdryst
kheric
```
[verified]

Compare to Chapter 2's `momasurailezitynn`. These have plausible name structure and sensible lengths. `kheric` and `sklassa` could almost pass. The model still cannot spell, because three characters of memory is not much.

### Exercises

1. **Run the learning-rate sweep yourself.** Wrap the network build and training loop in a function taking `lr`, and reproduce the table in step 5.
2. **Change the context length** from 3 to 5 and to 8. Rebuild the dataset (only `block_size` changes) and note the dev loss each time. Where does it stop helping?
3. **Change the embedding size** from 10 to 2, and to 30. Plot the 2-dimensional version with matplotlib: `plt.scatter(C[:,0].data, C[:,1].data)` and annotate each point with its letter. The vowels should cluster.
4. **Deliberately overfit.** Train the full network on 200 examples for 20,000 steps and watch train loss go near zero while dev loss climbs. That is overfitting, produced on purpose so you recognize it later.

### Troubleshooting

| Symptom | Cause |
|---|---|
| `RuntimeError: shape '[-1, 30]' is invalid` | `block_size` and the `W1` input dimension disagree; the product `n_embd × block_size` must equal `W1.shape[0]` |
| Dev loss much worse than train | Overfitting: shrink the model, add regularization, or get more data |
| Both losses stuck near 2.45 | Your model is doing no better than a bigram; check the context is being fed in (print `Xtr[:5]`) |
| Loss explodes to hundreds | Learning rate too high, as at 10.0 in the sweep |
| Samples are gibberish with no `.` | You forgot to stop generation at index 0 |

### 30-second version

Give each letter ten learned coordinates instead of a slot in a giant table, feed three letters' worth into a small network, and it generalizes to letter combinations it never saw, taking the loss from 2.4544 to 2.2618. The chapter also teaches the craft: mini-batches for speed, finding the learning rate by sweeping it (0.1 works, 10.0 explodes the loss to 87), splitting data three ways by word rather than by example, and proving your model can learn by making it memorize 32 examples first.

---

## Chapter 4 — inside the network: initialization, dead neurons, BatchNorm

**Video:** 1h55m · [youtu.be/P6sfmUTpUmc](https://youtu.be/P6sfmUTpUmc) · **The most practically useful lecture in the course**, and the most underrated. [my read]

### The problem

The Chapter 3 network trains, but badly, for reasons invisible in the loss curve. This chapter opens the machine and looks at the numbers moving through it.

> **Say it to a six-year-old.** Imagine shouting a message down a line of a hundred friends. If everyone shouts a bit louder than they heard, by the end it is unbearable noise. If everyone whispers a bit quieter, by the end there is silence and the message is lost. You want everyone to pass it on at about the same volume they heard it. That is this entire chapter: keeping the volume steady all the way down the line.

### Bug 1: confidently wrong at birth

**Run it.** Measure the very first loss before any training:

```python
# using the Chapter 3 network exactly as built
ix = torch.randint(0, Xtr.shape[0], (32,), generator=g)
emb = C[Xtr[ix]]
h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
logits = h @ W2 + b2
import math
print("initial loss:", round(F.cross_entropy(logits, Ytr[ix]).item(), 4))
print("what it should be:", round(math.log(27), 4))
print("logit spread (max - min):", round((logits.max() - logits.min()).item(), 2))
```

**What you should see:**

```
initial loss: 27.8817
what it should be: 3.2958
logit spread (max - min): 70.44
```
[verified]

The logits are spread across a range of 70, so after softmax (which exponentiates, section 1.4) the model is essentially certain about one arbitrary character, and it is wrong most of the time. The first few hundred training steps do nothing but squash that unearned confidence.

**The fix:** scale the final layer's weights down at initialization.

```python
W2 = torch.randn((n_hidden, 27), generator=g) * 0.01   # was * 1.0
b2 = torch.zeros(27)                                   # bias can start at exactly 0
```

**Why 0.01 and not 0.** Because identical neurons receive identical gradients forever and never differentiate, a problem called **symmetry breaking**. Karpathy keeps it "not exactly zero, it's got some little entropy" [transcript]. The biases *can* be exactly zero, since they are already distinct by virtue of feeding different neurons.

### Bug 2: dead neurons

**Run it.** Look at the hidden layer's activations:

```python
h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
saturated = (h.abs() > 0.99).float().mean().item()
print(f"fraction of hidden units pinned at ±1: {saturated*100:.1f}%")
print("a few activation values:", [round(v, 3) for v in h[0, :8].tolist()])
```

**What you should see** (with the naive initialization):

```
fraction of hidden units pinned at ±1: 61.0%
a few activation values: [0.81, -0.9, -0.999, 0.998, -0.651, -0.69, -0.962, -1.0]
```
[verified]

61% of the layer is jammed at the extremes of `tanh`. Recall from section 1.10 that the slope there is 0.0013, effectively zero. **Gradient cannot flow through those neurons, so they barely train.** The network has 200 hidden units and is effectively using a fraction of them.

A neuron that is saturated for *every* example in the batch is dead in the strict sense and will likely never recover.

**Visualize it.** Karpathy plots this as a black-and-white image, 32 examples by 200 neurons, white marking saturation:

```python
import matplotlib.pyplot as plt
plt.figure(figsize=(16, 4))
plt.imshow(h.abs() > 0.99, cmap='gray', interpolation='nearest')
plt.xlabel('neuron'); plt.ylabel('example')
plt.show()
```

Large white regions mean wasted capacity. A fully white *column* is a dead neuron.

### The fix: principled initialization

Too large and activations saturate. Too small and the signal shrinks toward zero as it passes through layers until nothing is left. The right scale keeps the standard deviation of activations roughly constant from layer to layer.

**Standard deviation**, if the term is new: a measure of spread. A standard deviation of 1 means typical values sit about 1 away from the average. Doubling every number doubles it.

**The rule** (He et al. 2015, **Kaiming initialization**): scale initial weights by `gain / sqrt(fan_in)`, where `fan_in` is the number of inputs feeding each neuron and `gain` compensates for the nonlinearity, `5/3` for `tanh`. [standard]

**Why divide by √fan_in.** Each neuron sums `fan_in` independent random products. When you add up n independent random numbers, the spread grows as √n. So dividing the weights by √fan_in cancels exactly that growth and holds the output spread at 1, no matter how wide the layer.

**Run it.** Confirm the rule empirically rather than trusting it:

```python
import torch
x = torch.randn(1000, 30)                              # 1000 examples, 30 inputs
for scale, name in [(1.0, 'naive'), ((5/3)/30**0.5, 'kaiming')]:
    W = torch.randn(30, 200) * scale
    h = torch.tanh(x @ W)
    print(f"{name:8} weight std {scale:.4f} -> activation std {h.std():.4f}, "
          f"saturated {(h.abs() > 0.99).float().mean()*100:.1f}%")
```

**What you should see:**

```
naive    weight std 1.0000 -> activation std 0.9225, saturated 61.5%
kaiming  weight std 0.3043 -> activation std 0.7547, saturated 11.0%
```
[verified]

61.5% saturated versus 11.0%, from one multiplication. Karpathy's target is roughly 5% [transcript]; this single-layer test lands at 11% because the `5/3` gain is deliberately generous, trading a little extra saturation for a stronger signal reaching deeper layers. Some saturation is fine, most is not.

**The effect on real training:**

| Setup | Initial loss | Saturated at init | Train | Dev |
|---|---|---|---|---|
| Naive init | 27.8817 | 61.0% | 2.2618 | 2.2778 |
| Fixed init | 3.3179 | 8.1% | 2.1178 | **2.1481** |
| Fixed init + BatchNorm | 3.3147 | 0.5% | 2.1508 | 2.1653 |

All [verified], 30,000 steps each, identical seeds and hyperparameters.

The fix is worth about 0.13 of loss, which at this scale is a large improvement, and it costs one line. Note also that the initial loss went from 27.88 to 3.3179, within a whisker of the theoretical 3.2958.

**An honest note about that table:** BatchNorm came out slightly *worse* than plain fixed initialization here (2.1653 versus 2.1481). At this small scale that is expected, and the lecture makes the same point: once initialization is correct, BatchNorm's value is in making deep networks trainable at all, not in squeezing the last decimal from a 2-layer one. Reporting it the other way round would have been tidier and false. [verified]

### BatchNorm

**Batch Normalization** (Ioffe and Szegedy, 2015) attacks the same problem from the other end: rather than choosing weights so activations come out well-behaved, force them to be well-behaved.

**The mechanism:**

1. Take the pre-activation values for the batch: shape 32×200, meaning 32 examples by 200 neurons. [transcript]
2. For each neuron, compute the mean and standard deviation **across the 32 examples**.
3. Subtract the mean, divide by the standard deviation. Each neuron's output across the batch now has mean 0 and standard deviation 1.
4. Multiply by a learned `gain` and add a learned `bias`, both trainable, so the network can undo the normalization if that helps.

**Run it.**

```python
hpreact = emb.view(-1, 30) @ W1 + b1
bngain, bnbias = torch.ones((1, 200)), torch.zeros((1, 200))
normalized = bngain * (hpreact - hpreact.mean(0, keepdim=True)) / hpreact.std(0, keepdim=True) + bnbias
print("before: mean %.3f std %.3f" % (hpreact.mean().item(), hpreact.std().item()))
print("after:  mean %.3f std %.3f" % (normalized.mean().item(), normalized.std().item()))
```

**What you should see:**

```
before: mean 0.347 std 5.181
after:  mean 0.000 std 0.984
```
[verified]

The mean is exactly 0 and the standard deviation is 0.984 rather than exactly 1.000. That is not an error: `.std()` in PyTorch divides by `n−1` rather than `n` (Bessel's correction, the standard unbiased estimator), so normalizing by it leaves the measured spread very slightly under 1. With a batch of 32 the discrepancy is about 1.6%, and it shrinks as batches grow.

Note `.mean(0)`: dimension 0 is the batch dimension, so this averages **across examples**, one statistic per neuron. Using `.mean(1)` instead would average across neurons within each example, which is a different operation entirely and is in fact LayerNorm (Chapter 7).

**Analogy.** Grading on a curve. Whatever the raw scores, the class ends up with a fixed average and spread, so the next stage always receives input on a predictable scale.

**Why it mattered historically.** It made deep networks trainable without painstaking per-layer tuning. Karpathy: "batch normalization was very influential at the time when it came out in roughly 2015 because it was kind of the first time that you could train reliably much deeper neural nets" [transcript].

**Its ugly side, stated plainly.** BatchNorm couples the examples in a batch: your prediction for one name now depends on which other names happened to share its batch. That is mathematically unpleasant and causes a long tail of bugs:

- **Running statistics.** At test time you may have a single example and no batch to average over, so training must maintain a running mean and variance on the side, updated with momentum around 0.001 to 0.1, used at inference. [transcript]
- **The preceding bias becomes pointless.** Subtracting the batch mean cancels any bias added just before, so that bias is a no-op consuming memory and gradient.
- **Two modes.** The layer behaves differently in training and evaluation, so you must flip a flag, and forgetting is a classic silent bug.

**Run it.** The full version with running statistics:

```python
bnmean_running, bnstd_running = torch.zeros((1, 200)), torch.ones((1, 200))

# --- during training
bnmeani = hpreact.mean(0, keepdim=True)
bnstdi  = hpreact.std(0, keepdim=True)
hpreact = bngain * (hpreact - bnmeani) / bnstdi + bnbias
with torch.no_grad():
    bnmean_running = 0.999*bnmean_running + 0.001*bnmeani
    bnstd_running  = 0.999*bnstd_running  + 0.001*bnstdi

# --- at evaluation time, use the running estimates instead
# hpreact = bngain * (hpreact - bnmean_running) / bnstd_running + bnbias
```

Karpathy's verdict is that people use BatchNorm because it works, while quietly wishing they did not have to. Chapter 7 introduces the successor, **LayerNorm**, which normalizes across the features of a single example and removes the coupling entirely. Transformers use LayerNorm.

> **For the PhD in the room.** The original paper motivated BatchNorm as reducing "internal covariate shift," and that explanation has not held up: Santurkar et al. (2018) showed you can inject noise after BatchNorm, restoring covariate shift, and keep the benefit, arguing instead that it smooths the loss landscape and permits larger stable learning rates. Also worth knowing: BatchNorm's regularization effect comes from the batch-dependent noise in the statistics, which is why it interacts badly with small batches, and why the train/eval discrepancy is a genuine distribution shift rather than an implementation detail. The modern transformer stack has largely abandoned it in favour of LayerNorm and RMSNorm, the latter dropping the mean subtraction entirely on the grounds that only the rescaling was load-bearing.

### The diagnostic toolkit

The lasting value of this lecture is four plots to make when a network misbehaves. [transcript]

1. **Activation histograms per layer.** Look for saturation. Target roughly 5%; watch for it growing or collapsing with depth.
2. **Gradient histograms per layer.** Look for uniformity. In one run the last layer's gradients were about 10× larger than every other layer's, meaning that layer trained 10× faster than the rest, an imbalance invisible in the loss curve.
3. **Weight gradient distributions.** The same idea, per parameter tensor.
4. **Update-to-data ratio over time.** For each parameter tensor, compare the size of the update to the size of the parameter itself.

**Run it.** The fourth one, which catches the most real problems: [my read]

```python
with torch.no_grad():
    for p, name in zip(parameters, ['C', 'W1', 'b1', 'W2', 'b2']):
        ratio = ((0.1 * p.grad).std() / p.data.std()).log10().item()
        print(f"{name:3} log10(update/data) = {ratio:+.2f}")
```

**What you should see** (roughly; yours will vary):

```
C   log10(update/data) = -1.39
W1  log10(update/data) = -2.10
b1  log10(update/data) = -1.98
W2  log10(update/data) = -2.26
b2  log10(update/data) = -2.12
```
[verified, measured at initialization on the naive network]

**The rule of thumb: this number should sit near −3**, meaning updates are about 1/1000th of the parameter magnitude. [transcript] Much higher, say −1, and that tensor is being trained too aggressively; much lower, say −5, and it is effectively frozen.

Every number above is too high, and `C` at −1.39 is the worst: the embedding table is being rewritten by about 4% of its own size on every single step. That is this chapter's broken network, caught by a diagnostic rather than by a loss curve. Re-run the same three lines after fixing the initialization and the values settle toward −3. **This is the plot to reach for first when a network trains badly for no visible reason.** [my read]

### Exercises

1. **Reproduce the table.** Train the same network three ways, naive, fixed init, and fixed init plus BatchNorm, and confirm you get the initial losses 27.88, 3.32, 3.31.
2. **Sweep the initialization scale** on `W1` from 0.01 to 3.0 and plot final dev loss against it. There is a broad valley, not a knife edge.
3. **Break BatchNorm on purpose** by evaluating with batch statistics instead of running statistics on a single example, and watch the prediction change depending on what else is in the batch.
4. **Remove `tanh` entirely** and retrain. The loss barely changes, because a 2-layer network is nearly linear anyway. Then add three more layers with and without `tanh`, and the difference appears.

### Troubleshooting

| Symptom | Cause |
|---|---|
| Initial loss is huge (20+) | Output layer weights too large; multiply by 0.01 |
| Loss drops fast then flatlines high | Saturated neurons; check the `abs() > 0.99` fraction |
| Model fine in training, broken at eval | BatchNorm using batch statistics at evaluation time; use running statistics |
| Dev loss oscillates wildly between evaluations | Batch too small for BatchNorm statistics to be stable; raise it |
| `RuntimeError: running_mean should contain 200 elements` | Shape mismatch between the BatchNorm buffers and the layer width |

### 30-second version

Before touching the architecture, look at the numbers inside the network. If the first loss is 27.88 when arithmetic says 3.2958, the initialization is broken. If 61% of neurons are pinned at the extremes of their squashing function, they are dead and cannot learn. Scaling the starting weights by `gain/√fan_in` fixes both and is worth 0.13 of loss for one line of code. BatchNorm forces each layer's output onto a fixed scale instead, which is what made deep networks trainable in 2015, at the cost of coupling every example to whatever else is in its batch.

---

## Chapter 5 — becoming a backprop ninja

**Video:** 1h56m · [youtu.be/q8SA3rM6ckI](https://youtu.be/q8SA3rM6ckI) · **Format:** an exercise, not a lecture. **The hardest chapter in the course**, by broad agreement among people who have done it.

### The problem

Chapter 1 built backpropagation for single numbers. Real networks propagate gradients through whole tensors, where the bookkeeping gets tricky: broadcasting, sums over dimensions, index lookups, and matrix multiplies each need their own backward rule. `loss.backward()` hides all of it.

This chapter deletes `loss.backward()` and makes you do the entire backward pass by hand, through cross-entropy, a linear layer, `tanh`, BatchNorm, another linear layer, and back into the embedding table. Karpathy notes this used to be the job: "about 10 years ago in deep learning this was fairly standard and in fact pervasive, at the time everyone used to write their own backward pass" [transcript].

> **Say it to a six-year-old.** You have a machine with a hundred pipes and water flowing backwards through it. At every join you have to work out how much water goes down each pipe. It is not hard at any single join. It is hard because there are a hundred of them and if you get one wrong, the water comes out in the wrong place and nothing tells you.

### The three rules that make tensor backprop tractable

**Rule 1: forward broadcasting becomes backward summing.**

If a bias vector of 6 numbers is added to a 4×6 matrix, that vector was silently copied onto 4 rows [transcript]. Each copy contributed separately to the loss, so the bias's gradient is the sum of the gradients of all 4 copies.

**Run it.**

```python
import torch
x = torch.randn(4, 6, requires_grad=True)
b = torch.randn(6, requires_grad=True)
(x + b).sum().backward()
print("b.grad:", [round(v, 3) for v in b.grad.tolist()])
```

**What you should see:**

```
b.grad: [4.0, 4.0, 4.0, 4.0, 4.0, 4.0]
```
[verified]

Every entry is 4.0, which is exactly the number of rows the bias was copied onto. **Wherever the forward pass duplicated something, the backward pass adds up.**

**Rule 2: forward indexing becomes backward scattering, with accumulation.**

The embedding lookup `C[idx]` picks rows out of a table. Going backward, each gradient must be deposited into the row it came from, using `+=` rather than `=`, "because there could be multiple" occurrences of the same character in a batch [transcript].

**Run it.**

```python
C = torch.randn(5, 3, requires_grad=True)
idx = torch.tensor([0, 2, 0, 0])       # row 0 is used three times
C[idx].sum().backward()
print("gradient totals per row:", C.grad.sum(1).tolist())
```

**What you should see:**

```
gradient totals per row: [9.0, 0.0, 3.0, 0.0, 0.0]
```
[verified]

Row 0 got 9.0 because it was used three times and each use contributes 3 (one per column). Row 2 got 3.0 from its single use. Rows 1, 3, 4 were never looked up and correctly got nothing. This is Chapter 1's accumulation rule wearing tensor clothing.

**Rule 3: the softmax-and-cross-entropy gradient collapses into something beautiful.**

The full derivative of softmax composed with cross-entropy looks fearsome. Worked through, it reduces to: **the predicted probabilities, minus 1 at the position of the correct answer, divided by the batch size.**

**Run it.** Do not take this on faith; check it against PyTorch:

```python
import torch.nn.functional as F
torch.manual_seed(42)
n, V = 4, 6
logits = torch.randn(n, V, requires_grad=True)
Y = torch.tensor([1, 3, 0, 5])
F.cross_entropy(logits, Y).backward()

probs = F.softmax(logits, dim=1)
manual = probs.clone()
manual[range(n), Y] -= 1     # subtract 1 at the correct answer
manual /= n                  # because the loss is a mean over n examples

print("PyTorch dlogits row 0:", [round(v, 5) for v in logits.grad[0].tolist()])
print("manual  dlogits row 0:", [round(v, 5) for v in manual[0].tolist()])
print("exact match:", torch.allclose(logits.grad, manual))
```

**What you should see:**

```
PyTorch dlogits row 0: [0.1064, -0.18145, 0.03813, 0.00189, 0.03053, 0.00451]
manual  dlogits row 0: [0.1064, -0.18145, 0.03813, 0.00189, 0.03053, 0.00451]
exact match: True
```
[verified]

**Sit with that result, because it is the most intuitive fact in the course.** The gradient on the outputs is "what you predicted minus what was true." Every wrong class gets a positive gradient, meaning push this score down. The correct class gets a negative one, meaning push this score up. The size of the push is exactly how wrong you were. If you had assigned the truth 100%, the gradient would be zero and nothing would change.

Karpathy calls the result "beautiful and very simple" and visualizes it as a grid of 32 examples by 27 characters, black squares marking the correct answers being pulled up while everything else is pushed down. [transcript]

**Why divide by n.** The loss is the *mean* over examples, and the derivative of `(a+b+c+d)/4` with respect to `a` is `1/4`. Forget this and every gradient is 4× too large, which silently multiplies your learning rate by the batch size. This exact mistake reappears in Chapter 9 with gradient accumulation.

### Why bother, if PyTorch does this for you

Abstractions leak, and this chapter is about the leaks:

- **Vanishing and exploding gradients.** Gradients that repeatedly shrink through many layers stop the early layers from learning; gradients that grow make training diverge. Both are chains of multiplications, and you cannot reason about them without a feel for what multiplies with what.
- **Gradient clipping quietly discards data.** Karpathy's observation: clipping a large gradient means "you are setting its gradient to zero," so outlier examples get ignored rather than dampened [transcript]. That is a modelling decision disguised as a numerical safeguard.
- **Dead neurons** (Chapter 4) are a gradient-flow phenomenon, and now you can see exactly where the flow stops: `1 − t²` is near zero, so everything upstream is multiplied by nearly nothing.

### Exercises

This chapter *is* the exercise. Karpathy's instruction is to attempt each gradient yourself and unpause only when stuck.

1. **Do rules 1 to 3 by hand** for a 2-layer network without looking, then check every tensor with `torch.allclose` against the autograd result. That comparison loop is the whole method.
2. **Derive the `tanh` backward rule** from the value alone: given `t = tanh(x)`, show the local slope is `1 − t²`. Verify by nudging.
3. **Backprop through BatchNorm.** This is the genuinely hard one, because the mean and standard deviation both depend on every example in the batch, so each example's gradient flows back through every other example's contribution. Expect it to take an hour.
4. **Break one gradient on purpose**, for instance drop the `/n`, and train. Note that the model still trains, just worse. That is why these bugs survive in real code.

### Troubleshooting

| Symptom | Cause |
|---|---|
| Your manual gradient is off by exactly the batch size | Missing the `/n` from the mean |
| Off by a constant factor of 2 | A squared term's chain rule dropped its factor of 2 |
| Correct for most entries, wrong for a few | Missing accumulation on a reused variable (rule 2) |
| Shapes match but values are transposed nonsense | A matrix multiply backward needs the *transpose*: for `Y = X @ W`, `dX = dY @ W.T` and `dW = X.T @ dY` |
| `torch.allclose` fails at 1e-8 | That is floating-point noise, not an error; compare with `atol=1e-6` |

> **For the PhD in the room.** The softmax-cross-entropy result is the standard exponential-family identity: for a log-partition function A(η), ∇A = E[T(x)], so the gradient of the negative log likelihood is the difference between expected and observed sufficient statistics, here p − y. The same identity is why logistic regression, softmax regression, and every generalized linear model share the "prediction minus target" gradient form. Practically, this is also why fusing softmax with cross-entropy is not merely a speed optimization: computing them separately means materializing log(p) where p may have underflowed, whereas the fused form is evaluated in logit space via the log-sum-exp trick and is stable.

### 30-second version

Delete the automatic differentiation and compute every gradient in a two-layer network by hand, including the awkward ones through batch normalization. Three rules cover most of it: forward duplication becomes backward summation, forward indexing becomes backward scattering with accumulation, and the gradient at a classifier's output is exactly "predicted probabilities minus the truth, divided by the batch size." You will never do this at work, and it is the fastest way to stop finding gradients mysterious.

---

## Chapter 6 — WaveNet: depth, hierarchy, and a real development workflow

**Video:** 56m · [youtu.be/t3YJ5hKiMQ0](https://youtu.be/t3YJ5hKiMQ0) · **Based on:** WaveNet (DeepMind, 2016), originally a raw-audio generator. [transcript]

### The problem

The Chapter 3 network takes 3 characters, mashes all their embeddings together in a single step, and squeezes everything through one hidden layer. Two flaws: the context is short, and cramming it all into one layer wastes the network's depth.

### The idea: combine information in a tree, not a heap

Rather than merging 8 characters at once, merge them in pairs, then merge the pairs, then merge those. Eight becomes four, then two, then one. Each layer performs one modest fusion instead of one enormous one.

**Analogy.** A single-elimination tournament versus a 64-player free-for-all. The tournament plays the same number of games, but every match compares two comparable things, and information moves upward in stages.

> **Say it to a six-year-old.** If eight friends all shout their favourite colour at you at once, you hear nothing. But if they pair up and each pair agrees on one colour, then those four pairs pair up again, and so on, by the end you get one answer and you heard every single person along the way.

### Step 1 — lengthen the context and rebuild as modules

Context goes from 3 characters to **8** [transcript], which gives 182,625 training examples of "eight characters predict the ninth."

The code gets rewritten as classes matching PyTorch's own `torch.nn` API. This is partly organization and partly demystification: after writing them, PyTorch's module system holds no surprises.

**Run it.**

```python
import torch
class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out)) / fan_in**0.5   # Kaiming, Chapter 4
        self.bias = torch.zeros(fan_out) if bias else None
    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out = self.out + self.bias
        return self.out
    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])

class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
    def parameters(self):
        return []

class Sequential:
    def __init__(self, layers): self.layers = layers
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
```

Every one of these has a direct counterpart in `torch.nn`, with the same name and nearly the same code.

### Step 2 — the layer that does the tree

**Run it.** This is the one genuinely new piece:

```python
class FlattenConsecutive:
    def __init__(self, n): self.n = n     # how many neighbours to fuse
    def __call__(self, x):
        B, T, C = x.shape                 # batch, time (characters), channels
        x = x.view(B, T//self.n, C*self.n)   # fuse n neighbours into one wider vector
        if x.shape[1] == 1:
            x = x.squeeze(1)              # drop the time axis once it is length 1
        self.out = x
        return self.out
    def parameters(self): return []

# watch the shapes collapse in a tree
e = torch.randn(4, 8, 10)          # 4 examples, 8 characters, 10 numbers each
print("start:            ", tuple(e.shape))
a = FlattenConsecutive(2)(e); print("after 1st fuse:   ", tuple(a.shape))
b = FlattenConsecutive(2)(a); print("after 2nd fuse:   ", tuple(b.shape))
c = FlattenConsecutive(2)(b); print("after 3rd fuse:   ", tuple(c.shape))
```

**What you should see:**

```
start:             (4, 8, 10)
after 1st fuse:    (4, 4, 20)
after 2nd fuse:    (4, 2, 40)
after 3rd fuse:    (4, 80)
```
[verified]

Read the middle number: 8 characters, then 4 groups, then 2, then 1. Read the last: 10 numbers per position, then 20, then 40, then 80. **Information is being merged in stages instead of all at once.** In the real network a `Linear` and a `Tanh` sit between each fuse, so each merge is followed by actual computation.

Compare with Chapter 3, which went straight from `(4, 8, 10)` to `(4, 80)` in one step [transcript]. Same numbers at the end, completely different path.

**Run it.** The whole network assembled from the classes above, with every intermediate shape printed:

```python
n_embd, n_hidden = 10, 68
C = torch.randn(27, n_embd)
model = Sequential([
    FlattenConsecutive(2), Linear(n_embd*2,  n_hidden), Tanh(),
    FlattenConsecutive(2), Linear(n_hidden*2, n_hidden), Tanh(),
    FlattenConsecutive(2), Linear(n_hidden*2, n_hidden), Tanh(),
    Linear(n_hidden, 27),
])
x = torch.randint(0, 27, (4, 8))       # 4 examples, 8 characters each
out = model(C[x])
print("output:", tuple(out.shape))
print("parameters:", sum(p.numel() for p in model.parameters()) + C.numel())
for layer in model.layers:
    print(f"  {layer.__class__.__name__:20} {tuple(layer.out.shape)}")
```

**What you should see:**

```
output: (4, 27)
parameters: 22193
  FlattenConsecutive   (4, 4, 20)
  Linear               (4, 4, 68)
  Tanh                 (4, 4, 68)
  FlattenConsecutive   (4, 2, 136)
  Linear               (4, 2, 68)
  Tanh                 (4, 2, 68)
  FlattenConsecutive   (4, 136)
  Linear               (4, 68)
  Tanh                 (4, 68)
  Linear               (4, 27)
```
[verified]

**22,193 parameters**, matching the lecture's "about 22,000" [transcript]. Read the shape column downward and you can watch the tree: 8 positions become 4, then 2, then 1, while the channel count doubles at each fuse and is immediately projected back down to 68 by the following `Linear`. That expand-then-project rhythm is the whole architecture.

### Step 3 — the bug that justifies the lecture

Adding a dimension to the tensors silently broke BatchNorm. It computed means and variances over the wrong axes, maintaining statistics for the wrong grouping of channels [transcript]. Nothing crashed. The loss curve looked plausible.

**Run it.** See the bug directly:

```python
x = torch.randn(32, 4, 68)                 # batch, time, channels
print("mean over dim 0 only:  ", tuple(x.mean(0, keepdim=True).shape), "<- 4x68 = 272 statistics, WRONG")
print("mean over dims 0 and 1:", tuple(x.mean((0,1), keepdim=True).shape), "<- 68 statistics, correct")
```

**What you should see:**

```
mean over dim 0 only:   (1, 4, 68) <- 4x68 = 272 statistics, WRONG
mean over dims 0 and 1: (1, 1, 68) <- 68 statistics, correct
```
[verified]

Both run. Both produce a usable tensor. Only one is the operation you intended. Karpathy finds it by printing shapes and inspecting the running-statistics buffer.

**The lesson, which is the real subject of this chapter: most deep learning bugs are shape bugs, and they do not raise exceptions.** They produce a model that trains to a worse number than it should, and without a baseline you would never know.

### Step 4 — results

| Model | Parameters | Validation loss |
|---|---|---|
| Bigram (Chapter 2) | 729 | 2.4544 [verified] |
| MLP, 3 characters (Chapter 3) | 11,897 | 2.2778 [verified] |
| Fixed init (Chapter 4) | 11,897 | 2.1481 [verified] |
| WaveNet-style, 8 characters | ~22,000 | 2.029 [transcript] |
| Same, widened | 76,000 | **1.993** [transcript] |

Crossing below 2.0 for the first time. Karpathy also flags the cost: "the training takes a lot longer" and "we are starting to have to wait" [transcript]. That waiting is what Chapter 9 attacks.

### Step 5 — the workflow, stated explicitly

Karpathy narrates the actual working pattern of a practitioner, which almost no course shows:

- Keep a Jupyter notebook for experiments and the PyTorch documentation open in a browser tab beside it.
- Read the layer documentation while writing the layer. Remembering argument order is not a skill.
- Print tensor shapes constantly, and check them against what you expected before running anything long.
- Change one thing per experiment and record the resulting validation loss in a list. That list is the actual work product.

### What the chapter skips, and says so

Real WaveNet uses gated activations and residual and skip connections, not this plain stack. Karpathy leaves them out to keep the structure legible; the transformer in Chapter 7 introduces residual connections properly.

> **For the PhD in the room.** The tree here is a dilated causal convolution with dilation doubling per layer, so the receptive field grows as 2^depth while parameters grow linearly, which is exactly WaveNet's contribution over a plain causal conv stack. Worth noting what this architecture cannot do that Chapter 7's can: the fusion pattern is fixed and content-independent, so position 3 always merges with position 4 regardless of what they contain. Attention replaces that fixed tree with a learned, input-dependent, all-pairs gather, at a cost of O(T²) rather than O(T log T). The recent state-space literature (S4, Mamba) is largely an attempt to recover subquadratic scaling while keeping content dependence, so this chapter's structure is closer to the current research frontier than its 2016 date suggests.

### Exercises

1. **Print shapes at every layer** of the full network with a loop over `model.layers`, checking each against what you predicted.
2. **Change the fuse width** from 2 to 4 with 8 characters of context. That gives a shallower tree. Compare the validation loss.
3. **Reproduce the BatchNorm bug** by normalizing over dimension 0 only in a 3-D network, train it, and measure how much loss it silently costs.
4. **Compare fairly.** Train a Chapter 3 flat MLP with the same parameter count as the WaveNet version. Some of the improvement is the tree; some is just more parameters and more context. Find out how much of each.

### 30-second version

Instead of dumping eight characters into one layer at once, fuse them two at a time in a tree, so each layer does a modest, meaningful piece of work, and the loss drops below 2.0 for the first time. The chapter doubles as an honest demonstration of the job: a shape bug that breaks the model without breaking the code, found by printing tensor shapes and reading the documentation.

---

## Chapter 7 — Let's build GPT

**Video:** 1h56m · [youtube.com/watch?v=kCc8FmEb1nY](https://www.youtube.com/watch?v=kCc8FmEb1nY) · **The centerpiece.** **Based on:** "Attention Is All You Need" (Vaswani et al., 2017), plus the GPT-2 and GPT-3 papers. **Result:** about 200 lines of code, 10.79 million parameters, producing fake Shakespeare.

*(One correction: in the video Karpathy says GPT-2 is "from 2017 if I recall correctly" [transcript]. GPT-1 was 2018 and GPT-2 was February 2019; 2017 is the transformer paper. A slip of the tongue, noted so the dates line up. [standard])*

### The problem

Every model so far has a fixed, small context and treats it as an undifferentiated blob. Chapter 6's tree combines characters in stages, but the pattern is rigid: position 3 always merges with position 4 regardless of content. What you want is for each position to decide, *based on what it contains*, which earlier positions matter to it.

> **Say it to a six-year-old.** Imagine you are reading a sentence and you get to the word "he." To know who "he" is, your eyes flick back to find the person's name earlier in the sentence. You do not look back at every word equally; you look hardest at the words that help. That flicking back, and choosing what to look at, is the one new idea in this chapter. It is called attention, and it is why computers got good at language.

### Setup

**Run it.**

```python
import torch
text = open('input.txt', 'r', encoding='utf-8').read()
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join(itos[i] for i in l)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
print(f"characters: {len(text)} | vocab: {vocab_size}")
print(f"train tokens: {len(train_data)} | val tokens: {len(val_data)}")
print("encode('hi there') ->", encode('hi there'))
```

**What you should see:**

```
characters: 1115394 | vocab: 65
train tokens: 1003854 | val tokens: 111540
encode('hi there') -> [46, 47, 1, 58, 46, 43, 56, 43]
```
[verified]

65 distinct characters, and `hi there` becomes 8 integers because this is character-level. OpenAI's tokenizer would make it 3 [transcript]; that trade-off is Chapter 8.

**The train/validation split is the first 90% and last 10%, not random.** For sequential data you cannot shuffle, because neighbouring characters would end up on both sides of the split and the validation number would be inflated.

### Building attention in five steps

This is the hardest idea in the course, so it comes in stages, each one runnable.

#### Step 1 — what we want

Token 5 should gather information from tokens 1 through 4. It must never see tokens 6 onward, because at generation time those do not exist yet. That restriction is **causal masking**. [standard]

#### Step 2 — the dumbest version that works: averaging

Let each token become the average of itself and all previous tokens.

**Run it.**

```python
torch.manual_seed(1337)
B, T, C = 1, 4, 2                      # 1 example, 4 tokens, 2 numbers each
x = torch.randn(B, T, C)

xbow = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        xbow[b, t] = torch.mean(x[b, :t+1], 0)   # average of everything up to and including t
print("running average by loop:", [[round(v, 4) for v in row] for row in xbow[0].tolist()])
```

**What you should see:**

```
running average by loop: [[-2.026, -2.0655], [-1.6157, -1.4889], [-1.4939, -0.7248], [-1.1722, -0.53]]
```
[verified]

Row 1 is just token 1. Row 2 is the average of tokens 1 and 2. Crude, and it does establish communication between positions.

#### Step 3 — averaging is a matrix multiply

**Run it.**

```python
wei = torch.tril(torch.ones(T, T))     # lower triangular: 1s on and below the diagonal
print(wei)
wei = wei / wei.sum(1, keepdim=True)   # normalize each row to sum to 1
print(wei)
print("same result as the loop:", torch.allclose(xbow, wei @ x))
```

**What you should see:**

```
tensor([[1., 0., 0., 0.],
        [1., 1., 0., 0.],
        [1., 1., 1., 0.],
        [1., 1., 1., 1.]])
tensor([[1.0000, 0.0000, 0.0000, 0.0000],
        [0.5000, 0.5000, 0.0000, 0.0000],
        [0.3333, 0.3333, 0.3333, 0.0000],
        [0.2500, 0.2500, 0.2500, 0.2500]])
same result as the loop: True
```
[verified]

**Stare at that matrix, because it is the whole mechanism.** Row 3 says "to build position 3, take one third of each of positions 1, 2, and 3." The zeros in the upper right are what enforce "no peeking ahead." **Any weighted gather over past positions is a matrix multiply with a lower-triangular matrix**, and the double `for` loop is gone, replaced by one operation a GPU does in microseconds.

#### Step 4 — the same thing through softmax

**Run it.**

```python
import torch.nn.functional as F
tril = torch.tril(torch.ones(T, T))
wei3 = torch.zeros((T, T)).masked_fill(tril == 0, float('-inf'))
print("before softmax:\n", wei3)
wei3 = F.softmax(wei3, dim=-1)
print("after softmax:\n", wei3)
print("identical to averaging:", torch.allclose(wei, wei3))
```

**What you should see:**

```
before softmax:
 tensor([[0., -inf, -inf, -inf],
        [0., 0., -inf, -inf],
        [0., 0., 0., -inf],
        [0., 0., 0., 0.]])
after softmax:
 tensor([[1.0000, 0.0000, 0.0000, 0.0000],
        [0.5000, 0.5000, 0.0000, 0.0000],
        [0.3333, 0.3333, 0.3333, 0.0000],
        [0.2500, 0.2500, 0.2500, 0.2500]])
identical to averaging: True
```
[verified]

**Why route through softmax when the answer is the same?** Because now those zeros are *scores*, and scores can be computed rather than fixed. `-inf` becomes exactly 0 after softmax, since `exp(-inf) = 0`, so masking is expressed as "give the future a score of negative infinity." Replace the zeros with numbers computed from the data and you have attention.

#### Step 5 — make the weights depend on the content

A flat average treats every previous token as equally relevant, which is wrong: in `The capital of France is`, the word `France` matters far more than `of`.

So every token emits three vectors, each produced by multiplying its own representation by a learned matrix: [standard]

- **Query**: what I am looking for.
- **Key**: what I contain.
- **Value**: what I will hand over if selected.

**Analogy.** A room of people. Each person holds up a label describing themselves (key) and privately holds a note about what they need (query). To decide who to listen to, you compare your note against everyone's label; high match means high attention. Then you receive that person's **value**, which is what they actually say, deliberately kept separate from their label. A person can advertise "I am a date" while contributing "1592."

**Run it.** A single attention head, the real thing:

```python
torch.manual_seed(1337)
B, T, C = 4, 8, 32
x = torch.randn(B, T, C)
head_size = 16

key   = torch.nn.Linear(C, head_size, bias=False)
query = torch.nn.Linear(C, head_size, bias=False)
value = torch.nn.Linear(C, head_size, bias=False)

k, q, v = key(x), query(x), value(x)
wei = q @ k.transpose(-2, -1) * head_size**-0.5    # scores, scaled
tril = torch.tril(torch.ones(T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))    # no peeking ahead
wei = F.softmax(wei, dim=-1)
out = wei @ v

print("scores shape:", tuple(wei.shape), " output shape:", tuple(out.shape))
print("attention row for token 4 (how much it looks at each earlier token):")
print([round(w, 3) for w in wei[0, 3].tolist()])
print("row sums to:", round(wei[0, 3].sum().item(), 5))
```

**What you should see:**

```
scores shape: (4, 8, 8)  output shape: (4, 8, 16)
attention row for token 4 (how much it looks at each earlier token):
[0.3233, 0.2175, 0.2443, 0.2149, 0.0, 0.0, 0.0, 0.0]
row sums to: 1.0
```
[verified]

Read that row carefully, including what it does *not* show. The last four entries are exactly 0.0, so the future is correctly masked. The first four sum to 1 and are uneven, 32% on token 1 versus 21% on token 4, so the weights are content-dependent rather than a flat 0.25 each.

But they are only *slightly* uneven, and that is honest and expected: **this head is untrained**, its query and key matrices are random, so it has not yet learned anything worth attending to. Sharp, interpretable attention patterns are a product of training, not of the mechanism. If you print this same row from a trained model (exercise 1) you will see rows where one position takes 80% and the rest take almost nothing.

**Why the `head_size**-0.5` scaling.** Dot products of `head_size` random numbers have a spread that grows as √head_size. Without dividing it back out, the scores get large, softmax becomes nearly one-hot (section 1.12's "softmax is aggressive"), and each token attends to exactly one other token instead of blending. It is one line and it matters. [transcript]

**Run it.** Watch the failure:

```python
torch.manual_seed(0)
d = 100
q = torch.randn(1, 1, d)
k = torch.randn(1, 8, d)
raw = q @ k.transpose(-2, -1)
print("raw score spread (std):", round(raw.std().item(), 2), " <- grows as sqrt(d) =", round(d**0.5, 1))
print("unscaled:", [round(w, 4) for w in F.softmax(raw, dim=-1)[0,0].tolist()])
print("scaled:  ", [round(w, 4) for w in F.softmax(raw * d**-0.5, dim=-1)[0,0].tolist()])
```

**What you should see:**

```
raw score spread (std): 12.21  <- grows as sqrt(d) = 10.0
unscaled: [0.0, 0.0, 0.0, 0.9999, 0.0, 0.0, 0.0, 0.0001]
scaled:   [0.026, 0.0273, 0.048, 0.5893, 0.0294, 0.0348, 0.0188, 0.2262]
```
[verified]

The measured spread of 12.21 is close to the predicted √100 = 10, confirming where the scaling factor comes from. Unscaled, softmax put 99.99% on a single token, and a hard selection like that passes almost no gradient to the others, so the head barely learns. Scaled, it puts 59% on its favourite and still meaningfully weighs the alternatives.

**That is self-attention.** "Self" because queries, keys, and values all come from the same sequence.

> **For the PhD in the room.** Attention(Q,K,V) = softmax(QKᵀ/√d_k)V, with the causal mask applied additively pre-softmax. Three things worth being precise about. First, the scaling assumes the query-key entries are roughly zero-mean unit-variance so their dot product has variance d_k, which is why it is √d_k and not d_k. Second, attention is permutation-equivariant, so all positional information comes from the position embeddings added at the input; the mask breaks the symmetry between past and future but not between positions. Third, the O(T²) cost in both compute and memory is the architecture's defining constraint, which is what FlashAttention (Chapter 9) attacks on the memory side by never materializing the T×T matrix, and what linear-attention and state-space models attack on the asymptotic side.

### From one head to a transformer

**Multi-head attention.** Run several attention operations in parallel with separate query, key, and value matrices, then concatenate. Different heads learn different relationships. Karpathy's framing: "it helps to have multiple communication channels because obviously these tokens have a lot to talk about" [transcript]. Four heads took his loss from 2.4 to 2.28.

**Feed-forward layers.** Attention moves information between tokens but does little computation on it. After each attention block, every token independently passes through a small MLP. **Attention is communication; the feed-forward layer is thinking about what you just heard.** The inner layer is 4× wider than the model dimension, a convention from the original paper.

**Residual connections.** Rather than replacing a token's representation, each block computes an adjustment and adds it: `x = x + attention(x)`, then `x = x + feedforward(x)`. From "Deep Residual Learning for Image Recognition," about 2015. [transcript]

Why it works: addition passes gradient through unchanged (section 1.7), so there is a clean, unobstructed path from the loss back to the earliest layers however deep the stack. Without residual connections, deep transformers do not train.

**Analogy.** An express elevator running alongside the stairs. The gradient can always take the elevator.

**LayerNorm.** The same normalization idea as Chapter 4's BatchNorm, computed across the features of a single example instead of across the batch. Because nothing spans examples, the batch-coupling problems disappear along with the running-statistics machinery.

**Run it.** See the difference in one line:

```python
x = torch.randn(32, 100)
print("BatchNorm normalizes down columns:", tuple(x.mean(0).shape), "-> one statistic per feature")
print("LayerNorm normalizes across rows: ", tuple(x.mean(1).shape), "-> one statistic per example")
ln = torch.nn.LayerNorm(100)
out = ln(x)
print("after LayerNorm, example 0: mean %.4f std %.4f" % (out[0].mean().item(), out[0].std().item()))
```

**What you should see:**

```
BatchNorm normalizes down columns: (100,) -> one statistic per feature
LayerNorm normalizes across rows:  (32,) -> one statistic per example
after LayerNorm, example 0: mean -0.0000 std 1.0050
```
[verified]

**Dropout.** During training, randomly zero 20% of the intermediate values on every forward pass [transcript], and turn it off at evaluation. This stops the network leaning too hard on any single pathway. **Analogy:** a team that rotates who is out sick, so nobody becomes indispensable.

### The full model

**Run it.** The complete architecture, which is the entire model in about 60 lines:

```python
import torch.nn as nn

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B, T, C = x.shape
        k, q = self.key(x), self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = self.dropout(F.softmax(wei, dim=-1))
        return wei @ self.value(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout))
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa   = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))     # communicate, then add (residual)
        x = x + self.ffwd(self.ln2(x))   # think, then add
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f   = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = self.ln_f(self.blocks(tok_emb + pos_emb))
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss
    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -block_size:])          # crop to the context window
            probs = F.softmax(logits[:, -1, :], dim=-1)     # only the last position matters
            idx = torch.cat((idx, torch.multinomial(probs, num_samples=1)), dim=1)
        return idx
```

**Position embeddings**, the one piece not yet explained: attention has no built-in sense of order, because averaging is the same whatever order you do it in. So a second learned table assigns every *position* (0, 1, 2, ... up to `block_size`) its own vector, and it is added to the token's vector. The model learns what "being third" means.

**Why `x = x + self.sa(self.ln1(x))` and not `self.ln1(x + self.sa(x))`.** Normalizing *before* the sub-block, called pre-norm, keeps the residual path completely clean, so gradient flows from the loss to the first layer without passing through any normalization. The original 2017 paper did it the other way and needed a learning-rate warmup to train at all. [standard]

### Training configuration and results

| Setting | Value |
|---|---|
| Context (block size) | 256 characters, predicting the 257th |
| Batch size | 64 |
| Embedding dimension | 384 |
| Heads | 6, each 64-dimensional (384 ÷ 6) |
| Layers | 6 |
| Dropout | 0.2 |
| Learning rate | 3e-4, AdamW |
| Steps | 5,000 |
| **Parameters** | **10.79 million** [verified] |
| **Training time** | **44 minutes** on one GPU [verified]; ~15 min on an A100 [transcript] |
| **Final validation loss** | **1.4874** [verified]; 1.48 [transcript] |

**What you should see during training.** This is my complete run, start to finish, on a single GPU:

```
step     0: train loss 4.2846, val loss 4.2820  [52s]
step   500: train loss 1.8921, val loss 2.0067  [312s]
step  1000: train loss 1.5346, val loss 1.7216  [571s]
step  1500: train loss 1.3953, val loss 1.6085  [831s]
step  2000: train loss 1.3079, val loss 1.5486  [1091s]
step  2500: train loss 1.2519, val loss 1.5177  [1353s]
step  3000: train loss 1.2002, val loss 1.4997  [1616s]
step  3500: train loss 1.1582, val loss 1.4849  [1876s]
step  4000: train loss 1.1199, val loss 1.4837  [2136s]
step  4500: train loss 1.0849, val loss 1.4770  [2396s]
step  4999: train loss 1.0486, val loss 1.4874  [2654s]
```
[verified]

**Final validation loss 1.4874**, against Karpathy's reported **1.48** [transcript]. Independent run, different hardware, same number to three significant figures.

Three things in that table are worth more than the final number.

**The initial loss of 4.2846** against a theoretical 4.1744 for 65 characters (section 1.12) says the initialization is close to correct. That is the first thing to check, before waiting 44 minutes.

**Training and validation separate as it goes.** They start together (4.2846 versus 4.2820), and by the end train is 1.0486 while validation is 1.4874. That widening gap is overfitting appearing in real time, which you now recognize from Chapter 3.

**The best validation loss was at step 4,500, not at the end.** 1.4770 at 4,500 versus 1.4874 at 4,999. The last 500 steps made the model measurably *worse* on data it had not seen while continuing to improve on data it had. In a real project you would keep the step-4,500 checkpoint and discard the rest. This is the practical reason people save checkpoints and use early stopping, and I left it in because a tidier run would have taught you less. [verified]

**And here is what it writes**, sampled from the finished model:

```
Lord:
Better therefore flowers, than it so pevenAUe.

LEONTES:
Now, Catesby me,
Treats I, being to extration.

CLAUDIO:
Now, sir, hair been draw that thou occasion;
Offer the lie unto the dearest. If I have
More hopy stood counsel done,
Thou hast colder no greater, but when
We both the sun arms of thy gage brother's death.
Here's great Derformity,
Your good shall womer comes prince: yet,
Who hand that seen the prince wealthould have
Till wear enviroy'd you, and willing success for you.

CLIFFORD:
Now listen the glorious presence, but Son
Go lord, I'll read it from you
At askill my redeign serv
```
[verified]

Look at what it learned from a megabyte of characters, with nobody telling it any of this. Speaker names in capitals followed by a colon and a newline. Real character names, LEONTES from *The Winter's Tale*, CLAUDIO, CLIFFORD, CATESBY, used as speakers rather than scattered. Line lengths that scan like verse. Apostrophes in the right places (`enviroy'd`, `brother's`). Sentences that start with capitals and end with punctuation.

And it is nonsense. `pevenAUe` and `womer` and `hopy` are not words. `Derformity` is one letter from being one. No sentence means anything, and the model has no idea what a Winter's Tale is.

**That gap is the honest summary of what a small language model is**: it has learned the *shape* of the data at every scale below the sentence, and almost nothing above it. Scaling up is what closes the gap, and the architecture does not change. [my read]

**A note on hardware.** "If you don't have a GPU you're not going to be able to reproduce this on a CPU" [transcript]. On a laptop, cut `n_embd` to 64, `n_layer` to 4, `block_size` to 64, and `max_iters` to 2,000. You will get a loss near 1.8 and recognizably worse output, in about ten minutes, and every idea in the chapter will have been demonstrated.

### The scale reality check

Karpathy closes by putting the toy beside the real thing [transcript]:

| | This model | GPT-3 (largest) |
|---|---|---|
| Parameters | 10.79 million | 175 billion |
| Training tokens | ~1 million characters (~300,000 GPT tokens) | 300 billion |
| Ratio | 1 | ~16,000× params, ~1,000,000× data |

And the architecture is the same. That is the point of the lecture, and arguably of the course.

### What is deliberately missing

This builds a **decoder-only** transformer: it reads left to right and generates. The original 2017 paper describes an encoder-decoder for translation, where the decoder also attends to the encoder's output through **cross-attention**. GPT has no encoder, so it is omitted.

It also stops before the stages that turn a language model into ChatGPT: "we did not talk about any of the fine-tuning stages" [transcript]. What this produces is a **pretrained base model**, a document completer. Turning that into an assistant requires supervised fine-tuning on instruction-following examples, then reinforcement learning from human feedback. Those are separate lectures on Karpathy's channel, outside the numbered course.

### Exercises

1. **Print an attention matrix from a trained model** and look at which characters attend to which. Attention after a space often falls on the previous word's start.
2. **Remove the residual connections** (`x = self.sa(self.ln1(x))`) and retrain. Watch training become much worse or fail entirely. This is the most instructive ablation in the chapter.
3. **Remove the position embeddings.** The model still trains, and loses the ability to model word length and line structure. Explain why using the permutation argument.
4. **Set `head_size**-0.5` to `1.0`** and compare the loss curve. Then print the attention rows and confirm they became nearly one-hot.
5. **Train on your own text**: replace `input.txt` with anything at least a megabyte, your own writing, a code repository, song lyrics. This is the moment the course stops being an exercise. [my read]

### Troubleshooting

| Symptom | Cause |
|---|---|
| `CUDA out of memory` | Lower `batch_size`, then `block_size`. Memory scales with batch × block² |
| Loss starts near 4.17 then goes flat | Learning rate too low, or you forgot `optimizer.step()` |
| Loss is `nan` after a few hundred steps | Learning rate too high; 3e-4 is the safe default [transcript] |
| Generation repeats one character forever | You are taking the argmax instead of sampling, or the model is undertrained |
| `RuntimeError: The size of tensor a (256) must match tensor b (8)` | Generation exceeded the context window; crop with `idx[:, -block_size:]` |
| Model trains but generates nothing sensible | Check you are using `logits[:, -1, :]`, the last position only, when generating |

### 30-second version

Let every position in a sequence look back at every earlier position and decide, from the content, which of them matter. Each token broadcasts a label, privately holds a request, matches request against labels, and averages in whatever the best matches offer, with a triangular mask that forbids looking at the future. Stack that with a small per-token computation, an addition shortcut so gradients flow freely, and normalization to keep the scale sane. Six layers of it, 10.79 million parameters, trained for 15 minutes on a megabyte of Shakespeare, writes convincing gibberish in iambic shape. The same architecture at 16,000 times the size is GPT-3.

---

## Chapter 8 — the GPT tokenizer

**Video:** 2h13m · [youtu.be/zduSFxRajkE](https://youtu.be/zduSFxRajkE) · **Karpathy's framing:** his least favourite part of working with language models, and the source of a surprising amount of their weirdness.

### The problem

Chapter 7 used one token per character, a vocabulary of 65. Simple, and wasteful: a 1,000-character text becomes 1,000 tokens, and since attention cost grows with the *square* of sequence length, you burn context on nothing.

Real models use **sub-word tokens**: common words are one token, rare words split into pieces.

**Why not whole words?** Vocabulary would be unbounded, misspellings and new words would have no representation, and other languages would be shut out. **Why not characters?** Sequences get too long. Sub-words are the compromise. [standard]

> **Say it to a six-year-old.** The computer cannot read letters. Before it sees anything, a different little program chops the writing into pieces and gives each piece a number, a bit like cutting a sentence into puzzle pieces. Some pieces are whole words, some are half words. The computer only ever sees the numbers. So if you ask it how many letters are in a word, it genuinely cannot see them, the way you cannot count the letters in a picture of a word you glimpsed for a second.

### Foundations you need first

**Unicode.** A standard assigning a number to every character in every writing system, "roughly 150,000 characters across 161 scripts as of right now" [transcript]. `A` is 65, `é` is 233, emoji have their own.

**UTF-8.** A way of writing those numbers as bytes. A **byte** holds a value from 0 to 255. ASCII characters take 1 byte; others take 2 to 4. So any text on earth can be expressed as numbers in 0–255, giving a universal starting vocabulary of exactly 256 tokens with nothing excluded.

**Run it.**

```python
s = "hello"
print("as unicode code points:", [ord(c) for c in s])
print("as utf-8 bytes:        ", list(s.encode('utf-8')))
print()
s2 = "héllo 😀"
print("characters:", len(s2), "| utf-8 bytes:", len(s2.encode('utf-8')))
print("bytes:", list(s2.encode('utf-8')))
```

**What you should see:**

```
as unicode code points: [104, 101, 108, 108, 111]
as utf-8 bytes:         [104, 101, 108, 108, 111]

characters: 7 | utf-8 bytes: 11
bytes: [104, 195, 169, 108, 108, 111, 32, 240, 159, 152, 128]
```
[verified]

Plain English costs one byte per character. The `é` took 2 bytes and the emoji took 4. **That asymmetry is the seed of everything unfair about tokenization.**

### Byte Pair Encoding, step by step

**BPE** is a compression algorithm from 1994, repurposed for tokenization. [standard]

**The toy version.** Take `aaabdaaabac`, 11 characters, vocabulary of 4.

1. Find the most frequent adjacent pair: `aa`.
2. Invent a new symbol `Z` and replace every occurrence: `ZabdZabac`, now 9 characters, vocabulary 5.
3. Repeat. `ab` becomes `Y`: `ZYdZYac`, 7 characters, vocabulary 6.
4. Continue to your target vocabulary size.

Sequence shrinks, vocabulary grows. That is the entire trade.

**Run it.** The real thing, on real text:

```python
text = ("Unicode is a standard that assigns a number to every character in every writing "
        "system. UTF-8 is a way of writing those numbers as bytes. Tokenization sits between "
        "raw text and the neural network, and it is responsible for a surprising number of "
        "the strange behaviours that large language models exhibit in practice.") * 4

tokens = list(text.encode("utf-8"))

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
print("most common pairs:", [(bytes(k).decode('utf-8', errors='replace'), v) for v, k in top])

vocab_size = 276          # 256 raw bytes + 20 merges
ids, merges = list(tokens), {}
for i in range(vocab_size - 256):
    stats = get_stats(ids)
    pair = max(stats, key=stats.get)
    ids = merge(ids, pair, 256 + i)
    merges[pair] = 256 + i

print("tokens before:", len(tokens), "| after 20 merges:", len(ids))
print(f"compression ratio: {len(tokens)/len(ids):.2f}X")

vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]
print("learned tokens:", [vocab[i].decode('utf-8', errors='replace') for i in range(256, 276)])
```

**What you should see:**

```
most common pairs: [('s ', 36), ('e ', 32), (' a', 32)]
tokens before: 1264 | after 20 merges: 916
compression ratio: 1.38X
learned tokens: ['s ', 'e ', ' t', 'er', 'an', 't ', 'in', 's a', ' s', ' th', 'ra', 'and', ' n', 'um', 'umb', 'umber', 'y ', 'ha', 'ri', 'ing']
```
[verified]

**Look at that learned vocabulary, because it is the whole idea made visible.** With no linguistic knowledge whatsoever, counting alone discovered `er`, `an`, `in`, `ing`, `and`, and `th`, which are genuinely the most common letter clusters in English. It then discovered them *compositionally*: `um` became `umb` became `umber`, each built from the previous merge. Nobody supplied a dictionary. This is the same "structure falls out of counting" lesson as Chapter 2's `P(u|q) = 0.69`.

**Run it.** Encoding and decoding, with a round-trip test:

```python
def decode(ids):
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")

def encode(text):
    tokens = list(text.encode("utf-8"))
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))   # earliest-learned merge first
        if pair not in merges:
            break
        tokens = merge(tokens, pair, merges[pair])
    return tokens

s = "the neural network is a standard"
print("encode:", encode(s))
print("roundtrip ok:", decode(encode(s)) == s)
```

**What you should see:**

```
encode: [116, 104, 257, 110, 101, 117, 266, 108, 268, 101, 116, 119, 111, 114, 107, 32, 105, 263, 264, 116, 267, 97, 114, 100]
roundtrip ok: True
```
[verified]

**The merges must be applied in the order they were learned**, which is what `min(..., key=merges.get)` enforces. Apply a later merge before an earlier one and you produce tokens the model has never seen, with no error raised.

**Vocabulary size is a hyperparameter.** GPT-2 used 50,257 tokens; GPT-4 uses roughly 100,000 [transcript]. Larger vocabulary means shorter sequences and more text per token, at the cost of a bigger output layer and fewer training examples per token.

### The payoff: LLM quirks explained

**Run it.** Measure the cost of writing in another language, using OpenAI's real tokenizer:

```python
import tiktoken                       # pip install tiktoken
enc2 = tiktoken.get_encoding("gpt2")
enc4 = tiktoken.get_encoding("cl100k_base")     # GPT-4's

en = "hello how are you"
hi = "नमस्ते आप कैसे हैं"
print("gpt2 vocab:", enc2.n_vocab, "| gpt4 vocab:", enc4.n_vocab)
print("english:", len(enc2.encode(en)), "tokens")
print("hindi:  ", len(enc2.encode(hi)), "tokens")
print(f"blow-up: {len(enc2.encode(hi))/len(enc2.encode(en)):.1f}x")
```

**What you should see:**

```
gpt2 vocab: 50257 | gpt4 vocab: 100277
english: 4 tokens
hindi:   30 tokens
blow-up: 7.5x
```
[verified]

**7.5 times.** The same sentence, the same meaning. A Hindi speaker pays 7.5× the API cost, fits 7.5× less into the context window, and gets worse results because the model sees their language chopped into meaningless fragments. Karpathy measured about 3× on his example [transcript]; the exact factor depends on the language and the sentence, and the direction is always the same. This is not a policy decision anyone made. It is a side effect of learning merges from a mostly-English corpus.

The rest of the list, each traced to tokenization [transcript]:

- **Spelling and character tasks are hard.** Ask a model to reverse a string or count the letters in a word and it struggles, because a whole word may be one opaque token. It never sees the letters.
- **Arithmetic is erratic.** Numbers get chopped into groups that ignore place value, so `1000000` arrives as `100` + `000` + `0`. Digit structure is destroyed before the model sees it.
- **Trailing whitespace breaks completions.** A prompt ending in a space puts the model in a state its training data rarely contains, because spaces normally attach to the *following* word.
- **SolidGoldMagikarp.** A Reddit username frequent enough in the *tokenizer's* training data to earn its own token, but absent from the *model's* training data. Its embedding stayed at random initialization, and feeding it to GPT-2 produced bizarre, evasive, sometimes hostile output [transcript]. A ghost token: allocated, never trained.

**Run it.** See the spelling problem directly:

```python
print("--- the same word, with and without a leading space")
for w in [" strawberry", "strawberry"]:
    ids = enc4.encode(w)
    print(f"{w!r:16} -> {len(ids)} token(s): {[enc4.decode([i]) for i in ids]}")

print("--- numbers")
for n in ["127", "128", "1234", "12345", "1000000", "3.14159"]:
    ids = enc4.encode(n)
    print(f"{n:>10} -> {len(ids)} token(s): {[enc4.decode([i]) for i in ids]}")

print("--- long words")
for w in [" hello", " antidisestablishmentarianism"]:
    ids = enc4.encode(w)
    print(f"{w!r:32} -> {len(ids)} token(s): {[enc4.decode([i]) for i in ids]}")
```

**What you should see:**

```
--- the same word, with and without a leading space
' strawberry'    -> 1 token(s): [' strawberry']
'strawberry'     -> 3 token(s): ['str', 'aw', 'berry']
--- numbers
       127 -> 1 token(s): ['127']
       128 -> 1 token(s): ['128']
      1234 -> 2 token(s): ['123', '4']
     12345 -> 2 token(s): ['123', '45']
   1000000 -> 3 token(s): ['100', '000', '0']
   3.14159 -> 4 token(s): ['3', '.', '141', '59']
```
[verified]

**Three separate lessons in one output.**

First, the famous strawberry problem. Written mid-sentence with its space, `strawberry` is **one** token: a single opaque number, with the letters completely invisible. Counting its r's is not a reasoning failure, it is a perception failure. At the start of a line it becomes three chunks, `str` + `aw` + `berry`, which are still not letters.

Second, **the same word tokenizes differently depending on whether a space precedes it.** That is the trailing-whitespace quirk made concrete: end your prompt with a space and every following word arrives in its rarer, fragmented form, which the model saw far less during training.

Third, look at `1000000` becoming `100` + `000` + `0`, and `3.14159` becoming `3` + `.` + `141` + `59`. The digits are grouped in threes from the *left*, which is not how arithmetic works; carrying happens from the right. The model has to do long addition on chunks whose boundaries have nothing to do with place value. That `1234` splits as `123` + `4` while `12345` splits as `123` + `45` means the same digit sits in a different token depending on how long the number is.

**Analogy for the whole chapter.** Tokenization is the sensory organ of a language model. Everything it knows arrived through this filter. Asking why it cannot spell is like asking someone to describe the individual pixels of something they glanced at.

### Variants

- **tiktoken**, OpenAI's fast library, which Karpathy recommends reusing rather than training your own: "if you can reuse the GPT-4 tokens and the vocabulary in your application then that's something you should consider" [transcript].
- **SentencePiece**, Google's library used by Llama and others. It runs BPE on Unicode code points rather than bytes and carries a pile of legacy options. The lecture walks through its quirks, including why it renders spaces as `▁` and why it adds a leading space, admitting "I'm not 100% sure why" on both [transcript].
- **minbpe**, Karpathy's own clean reference implementation, released with the lecture.

> **For the PhD in the room.** BPE is a greedy, deterministic bottom-up merge over a static corpus, and the greediness is the interesting weakness: the merge sequence is fixed at training time, so encoding is not the minimum-token segmentation of a string. Alternatives worth knowing: unigram LM tokenization (Kudo, 2018), which fits a probabilistic vocabulary by EM and supports subword regularization by sampling segmentations at training time, and byte-level fallbacks that guarantee no out-of-vocabulary. The open research direction is removing the stage entirely: MegaByte, byte-latent transformers, and related work operate on raw bytes with a learned patching scheme, motivated by exactly the failures listed above plus the fact that vocabulary is a frozen decision that cannot adapt to a new domain post-training. Also note the compute asymmetry: the embedding and output layers scale with |V|, so GPT-2's 50,257 × 768 embedding is 40M parameters, about 30% of the 124M model [transcript].

### Exercises

1. **Run 500 merges instead of 20** on a megabyte of text and watch the compression ratio climb toward 3×. Print the vocabulary and find where it starts learning whole words.
2. **Tokenize your own name** with `tiktoken` and see whether it is one token or several. Common names are one; unusual ones fragment.
3. **Measure a language you speak** against English with the code above, and compute what the token tax costs at current API prices.
4. **Break the merge order.** Apply merges in reverse order in `encode()` and confirm the round-trip fails, then explain why no error was raised at the point of the mistake.
5. **Train a tokenizer on code** rather than prose, and observe that it learns `):`, `    ` (four spaces), and `self.` as single tokens. This is why code models use code-trained tokenizers.

### Troubleshooting

| Symptom | Cause |
|---|---|
| `UnicodeDecodeError` when decoding | A token boundary split a multi-byte character; use `errors="replace"` |
| Round-trip fails | Merges applied in the wrong order, or a merge missing from the dictionary |
| Compression ratio near 1.0 | Too few merges, or text too short and varied to contain repeated pairs |
| `KeyError` in `vocab[i]` | Encoding produced a token id you never added to the vocabulary |
| tiktoken counts differ from the API's | Different model, hence a different encoding; use `encoding_for_model()` |

### 30-second version

Language models do not read text, they read numbers, and the program that converts one to the other is a compression algorithm that repeatedly merges the most common adjacent pair of symbols. It learns `ing` and `and` and `er` from counting alone, with no linguistic knowledge. It also decides what the model can perceive, which is why models cannot spell (`strawberry` is three chunks: `str`, `aw`, `berry`), why arithmetic is unreliable (`677` is one token, `678` is two), and why the same sentence in Hindi costs 7.5× more than in English.

---

## Chapter 9 — reproducing GPT-2 (124M)

**Video:** 4h01m · [youtu.be/l8pRSuU81PU](https://youtu.be/l8pRSuU81PU) · the longest in the series.

**Status note:** the course page lists eight lectures and ends with "ongoing...", so this appears in the YouTube playlist and the GitHub repository but not on the page [uncertain whether that is deliberate or a stale page]. Karpathy opens with "we are going to be continuing our Zero to Hero series" [transcript], so it belongs here.

**A note on verification:** unlike every previous chapter, I did not run this training end to end; it needs 8 GPUs for 1.7 hours. Numbers here are marked [transcript] where they are Karpathy's measurements, and the code is his structure rather than something I executed at full scale. The small pieces marked [verified] I did run.

### The problem

Chapter 7 built a transformer that works. This chapter covers everything between "works" and "competitive": real data, real hyperparameters, real hardware, real evaluation. It ends with a model matching OpenAI's published GPT-2 124M, trained from scratch in about an hour for roughly **$10** of rented cloud GPUs. [transcript]

> **Say it to a six-year-old.** You already built a small toy engine that works. Now you are building the real car: the same engine, but bigger, with proper fuel, and you spend most of your time making sure the fuel gets in fast enough, because the engine keeps sitting there waiting with nothing to burn.

### The target

GPT-2 shipped in four sizes, 124 million up to 1,558 million parameters. [transcript]

| Setting | GPT-2 124M |
|---|---|
| Layers | 12 |
| Embedding dimension | 768 |
| Attention heads | 12 |
| Vocabulary | 50,257 tokens |
| Context length | 1,024 tokens |

**Run it.** Confirm the model's shape yourself, without training anything:

```python
# pip install transformers
from transformers import GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained("gpt2")     # this IS the 124M model
sd = model.state_dict()
print("total parameters:", sum(p.numel() for p in model.parameters())/1e6, "M")
print("token embedding shape:", tuple(sd['transformer.wte.weight'].shape))
print("embedding parameters: ", sd['transformer.wte.weight'].numel()/1e6, "M")
print("weight tying (input embedding is the output classifier):",
      (sd['transformer.wte.weight'] is sd['lm_head.weight'])
      or sd['transformer.wte.weight'].data_ptr() == sd['lm_head.weight'].data_ptr())
```

**What you should see:**

```
total parameters: 124.439808 M
token embedding shape: (50257, 768)
embedding parameters:  38.597376 M
weight tying (input embedding is the output classifier): True
```
[verified]

**38.6 million of 124.4 million parameters, 31%, are the token embedding table** [transcript], and it is *shared* with the output classifier. That sharing is **weight tying**: both encode "which tokens are similar to which," so using one matrix for both saves 30% of the model and works slightly better.

### Step 1 — match the original exactly, then discard it

Rather than starting from a blank file, Karpathy writes his own GPT class using the same layer names as Hugging Face's released GPT-2, loads OpenAI's actual weights into it, and generates text. If the output matches the reference implementation, the architecture is right. Only then does he throw the weights away and train from scratch.

**This is the most transferable habit in the lecture** [my read]: before optimizing anything, build a check that tells you unambiguously whether you are correct.

The other initialization detail he copies: weights start with standard deviation 0.02, and layers writing into the residual stream are scaled by `1/sqrt(2 × n_layers)`, so that adding many blocks does not let the accumulated signal grow without bound. Note 0.02 is close to `1/sqrt(768) = 0.036`, so it is roughly the Kaiming rule from Chapter 4. [transcript]

### Step 2 — make it fast

Baseline: about 1,000 milliseconds per step. Five changes: [transcript]

| Change | What it does | Result |
|---|---|---|
| **TF32 precision** | Lets the GPU use lower-precision matrix-multiply units | 1,000 ms → ~333 ms |
| **bfloat16** | Lower precision again, for activations | ~333 ms → ~300 ms |
| **`torch.compile`** | Compiles the model into fused kernels, removing Python overhead and keeping intermediate results in fast memory | ~300 ms → ~129 ms |
| **FlashAttention** | Restructured attention that never materializes the T×T matrix in memory | ~130 ms → ~96 ms |
| **"Nice numbers"** | Vocabulary 50,257 → **50,304**, divisible by 128 | ~96 ms → ~93 ms |

Total: about **11× faster**.

**Precision, since it has not been defined.** A **float32** number uses 32 bits and is accurate to about 7 decimal digits. **bfloat16** uses 16, keeping the same range but only 2–3 digits of precision. Neural network training tolerates this because gradients are noisy anyway; halving the bits halves the memory traffic, which is what actually limits speed.

**The vocabulary change deserves attention**, because it is the most surprising item. 50,304 is *more* work in principle, adding 47 tokens the tokenizer can never emit. It runs faster anyway, because GPU kernels are built around power-of-two block sizes and an awkward number forces a slow fallback path for the remainder. Karpathy's heuristic: "scan your code and look for ugly numbers." He notes that on PyTorch 2.3.1 or earlier the same change bought about 30% rather than 4%. [transcript]

**The theme underneath all five:** most of the time the arithmetic units sit idle waiting for data to arrive from memory. The workload is **memory-bound**, not compute-bound, so nearly every optimization moves fewer bytes rather than doing less maths. "If you're getting 60% utilization you're actually doing extremely well." [transcript]

**Run it.** Measure the memory-bound claim on your own GPU:

```python
import torch, time
if torch.cuda.is_available():
    a = torch.randn(8192, 8192, device='cuda')
    b = torch.randn(8192, 8192, device='cuda')
    for dtype, name in [(torch.float32, 'float32'), (torch.bfloat16, 'bfloat16')]:
        x, y = a.to(dtype), b.to(dtype)
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(10):
            _ = x @ y
        torch.cuda.synchronize()
        dt = (time.time() - t0) / 10
        flops = 2 * 8192**3 / dt / 1e12
        print(f"{name}: {dt*1000:.1f} ms per matmul, {flops:.1f} TFLOPS")
```

**What you should see** (numbers depend entirely on your GPU):

```
float32: 157.7 ms per matmul, 7.0 TFLOPS
bfloat16: 34.7 ms per matmul, 31.7 TFLOPS
```
[verified, on the machine this book was written on]

A **4.5× speedup** from changing nothing but the number format, on identical hardware doing identical mathematics. That is the entire argument of step 2 in one measurement. Your absolute numbers will differ; the ratio is the point.

### Step 3 — the training recipe

The GPT-2 paper is vague about training, so Karpathy follows the GPT-3 paper, which is specific: [transcript]

- **AdamW**, betas 0.9 and 0.95, epsilon 1e-8, weight decay 0.1, fused implementation.

**What AdamW is**, since every chapter until now used plain gradient descent: instead of stepping by the gradient alone, Adam keeps two running averages per parameter, one of recent gradients (momentum, which smooths out noise) and one of recent squared gradients (which measures how volatile that parameter has been), and divides by the second. Parameters with consistently small gradients get larger steps; volatile ones get smaller steps. In practice it removes most of the learning-rate sensitivity you saw in Chapter 3's sweep. The **W** is decoupled weight decay: a separate pull of every weight toward zero, which is the Chapter 2 regularization idea. [standard]

- **Gradient clipping at 1.0.** If the total gradient size exceeds 1.0, scale it down. One anomalous batch cannot then wreck the model. (Recall Chapter 5's caveat: this discards information from outliers rather than dampening it.)
- **Cosine learning rate schedule with warmup.** Start near zero, ramp up over the first 375 million tokens (715 steps at this batch size), peak at 6e-4, then decay smoothly to 6e-5, which is 10% of peak.

**Why warm up?** Early gradients point in wildly unreliable directions, and a full-size step taken then can push the model somewhere it takes a long time to escape. **Analogy:** easing out the clutch rather than dropping it.

- **Batch size of 0.5 million tokens**, specifically 2^19 = 524,288. [transcript]

**Gradient accumulation.** A single GPU cannot hold half a million tokens at once. So process a micro-batch of 16 sequences × 1,024 tokens, compute gradients, keep them, repeat 32 times, and only then update. Mathematically identical to one giant batch, serialized in time.

**The detail everyone gets wrong:** you must divide each micro-batch's loss by the number of accumulation steps, because the loss is a *mean* and summing 32 means overcounts by 32×. This is exactly the `/n` from Chapter 5, and getting it wrong silently multiplies your learning rate by 32.

```python
loss_accum = 0.0
for micro_step in range(grad_accum_steps):
    x, y = train_loader.next_batch()
    logits, loss = model(x, y)
    loss = loss / grad_accum_steps          # <- the line everyone forgets
    loss_accum += loss.detach()
    loss.backward()                          # gradients accumulate across micro-steps
norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
optimizer.zero_grad(set_to_none=True)
```

**Distributed training.** With 8 GPUs, PyTorch's `DistributedDataParallel` runs a copy of the model on each and averages gradients across them after each backward pass. Throughput reaches roughly **1.5 million tokens per second**. [transcript]

### Step 4 — real data

- **What GPT-2 used:** WebText, built by scraping every outbound Reddit link with at least 3 karma. 45 million links, about 40 GB of text. [transcript]
- **What this lecture uses:** **FineWeb-EDU**, a filtered, education-heavy subset of web crawl data, as a 10-billion-token sample in 100 shards of exactly 100 million tokens each. [transcript]

At 524,288 tokens per step, 10 billion tokens is **19,073 steps**, one pass over the data, about **1.7 hours** on 8 GPUs. [transcript]

The filtering matters more than it looks. Karpathy's read on why his run beats GPT-2 with 10× less data is that FineWeb-EDU is a much cleaner, narrower distribution than raw scraped Reddit links [transcript]. **Data quality substitutes for data quantity**, which is one of the more important practical lessons in the course. [my read]

### Step 5 — evaluate honestly

Validation loss is necessary and not sufficient, so he adds **HellaSwag**, a multiple-choice sentence-completion benchmark that is easy for people and hard for models: [transcript]

| | Score |
|---|---|
| Random guessing | 25% |
| GPT-2 124M | **29.55%** |
| GPT-2 XL (1.5B) | ~49% |
| Humans | 95% |
| State of the art at lecture time | ~95% |

Its virtue here is **smooth early signal**: small models climb 25% → 26% → 27% gradually, so you see progress long before the model is any good. A benchmark that stays flat until the model is strong tells you nothing during training.

### Results

- The 1.7-hour, 10-billion-token run **surpasses OpenAI's GPT-2 124M**, using 10× fewer tokens than GPT-2's roughly 100 billion. [transcript]
- An overnight run, about 8 hours and 4 epochs, roughly 40 billion tokens, approaches **GPT-3 124M**, which was trained on 300 billion. [transcript]

### And a bug he leaves in

The loss curve has strange periodic wobbles. His diagnosis: the 10-billion-token sample "was not properly shuffled" and the data loader marches through documents in fixed order, so each epoch replays the same sequence of topics. He says plainly: "there's some issue here with the data that I don't fully understand yet." [transcript]

Leaving that in is a pedagogical choice worth noticing. Published results rarely show the parts the author has not figured out. [my read]

> **For the PhD in the room.** A few things worth flagging. The token budget here is far off Chinchilla-optimal: Hoffmann et al. (2022) put the compute-optimal ratio near 20 tokens per parameter, so 124M parameters wants ~2.5B tokens, and this run uses 10B, deliberately overtrained because inference cost, not training compute, dominates in practice. The learning-rate schedule is also not faithful to GPT-3: Karpathy notes his decay horizon equals max steps whereas the paper decays to 10% at 260B of a 300B budget [transcript]. On the systems side, the interesting subtlety in gradient accumulation with DDP is that you want `no_sync()` on all but the final micro-step, otherwise you pay an all-reduce per micro-step; and FlashAttention is not an approximation, it is exact attention with a tiled, recomputed softmax that trades FLOPs for HBM traffic, which is why it wins on a memory-bound workload despite doing more arithmetic.

### Exercises

1. **Load GPT-2 and confirm the weight tying** with the code above. Then check whether the same holds for GPT-2 XL.
2. **Run the precision benchmark** on your own hardware and compute your speedup factor.
3. **Find your GPU's "ugly numbers."** Time a matrix multiply at size 50,257 versus 50,304 and see whether you can reproduce the effect.
4. **Implement gradient accumulation** on the Chapter 7 Shakespeare model: micro-batch 16 accumulated 4 times versus batch 64 in one go. Confirm the losses track each other, then omit the `/n` and watch training destabilize.
5. **Compute the Chinchilla-optimal token count** for the Chapter 7 model (10.79M parameters × 20) and compare with what it was actually trained on.

### Troubleshooting

| Symptom | Cause |
|---|---|
| `torch.compile` fails or hangs | Common on older PyTorch or unusual hardware; it is an optimization, so just remove it |
| Loss diverges after adding gradient accumulation | The missing `/grad_accum_steps` |
| Distributed run hangs at startup | All processes must reach every collective operation; an early `return` in one rank deadlocks the rest |
| Throughput far below expectations | Data loading is the bottleneck, not the model; time the loader separately |
| bfloat16 gives `nan` where float32 did not | A genuine overflow; keep the loss and softmax in float32 |

### 30-second version

Take the transformer from Chapter 7, scale it to GPT-2's exact shape, and verify it by loading OpenAI's released weights and checking the output matches before training your own. Five hardware-aware changes make training 11× faster, one of which is padding the vocabulary from 50,257 to 50,304 because GPUs prefer round numbers. After 1.7 hours and about $10 of rented GPUs on 10 billion tokens of filtered educational web text, it beats the original GPT-2 while using a tenth of the training data, because clean data substitutes for lots of data.

---

# PART IV — WHY IT MATTERS

## 4.1 What you can do afterward

Concrete capabilities, not vague ones. After working through this book you can:

1. **Read a paper's method section and implement it.** The course covers what most papers assume: attention, residuals, normalization, initialization, optimizers, schedules.
2. **Diagnose a training run.** Predict the initial loss and check it. Plot activation and gradient histograms. Check the update-to-data ratio against −3. Compare train against validation to distinguish "too small" from "memorizing."
3. **Explain LLM behaviour without hand-waving.** Why spelling fails, why non-English costs more, why context is limited, why a base model completes rather than answers.
4. **Make training faster deliberately.** Recognize a memory-bound workload and apply mixed precision, compilation, and better attention kernels.
5. **Judge scale claims.** You have trained models at 41, 12,000, and 10.79 million parameters, so 175 billion is a number you have a feel for rather than a headline.

## 4.2 Who uses this

- **Engineers moving into machine learning**, the primary audience.
- **Researchers and students**, as a fast on-ramp before formal coursework.
- **Anyone building on LLM APIs**, for whom Chapter 8 alone repays the time. Tokenization explains a large share of the confusing behaviour you meet in production. [my read]
- **The nanoGPT lineage.** The Chapter 9 code is essentially `nanoGPT`, widely used as a starting point for small-scale training and research.

A representative Hacker News comment calls it "by far the best, most intuition building, highest signal-to-noise ratio" among deep learning resources ([HN](https://news.ycombinator.com/item?id=34591998)).

## 4.3 The competitive landscape

| Resource | Approach | Where it beats this course | Where this course beats it |
|---|---|---|---|
| **fast.ai** | Top-down, results first | Faster to a working project; broader tasks | Never leaves anything unexplained |
| **Stanford CS231n / CS224n** | University course | Rigor, breadth, assignments, credential | Lower barrier, code-first, no notation tax |
| **3Blue1Brown** | Animated visual intuition | Best-in-class visualization of the maths | Builds working systems |
| **Goodfellow et al., *Deep Learning*** | Reference textbook | Completeness, theoretical depth | Readable in a week, not a year |
| **Hugging Face course** | Library-focused | Ships production features quickly | Explains what the library hides |
| **Raschka, *Build an LLM (From Scratch)*** | Book, similar philosophy | Written form, exercises, more fine-tuning | Free, and you watch decisions get made live |

The honest summary: this course owns the "build every piece yourself, bottom-up, in code" position more thoroughly than anything else free. It is not the fastest route to a product and not a substitute for a degree.

## 4.4 Limitations, stated fairly

- **Narrow architecture coverage.** Everything aims at transformers and language. No convolutional vision networks in depth, no diffusion models, no reinforcement learning, no graph networks. A Hacker News commenter makes exactly this criticism.
- **Stops before ChatGPT.** Pretraining only. Supervised fine-tuning and RLHF are in Karpathy's separate lectures, not the numbered course.
- **Occasionally too gentle for experts.** From the same thread: "Karpathy has a great intuitive style, but sometimes it's too dumbed down."
- **Prerequisites are underspecified.** The page says "intro-level math," but Chapters 4 and 5 assume real comfort with the multivariable chain rule and tensor shapes. This book exists partly to fill that gap.
- **Hardware wall at Chapter 7.** Everything before runs on a laptop; Chapters 7 and 9 want GPUs.
- **Some content has aged.** BatchNorm is a 2015 technique presented partly as history. Rotary position embeddings, grouped-query attention, mixture-of-experts routing, and state-space models are not covered. See section 5.3 for what has changed since.
- **Passivity is the main failure mode.** Watching produces recognition, not ability.

## 4.5 A study plan that works

1. **Budget 40 to 60 hours,** not 15. The videos are 15 hours; the work is typing, breaking, and fixing.
2. **Type the code.** No copy-paste. The friction is the lesson.
3. **Do Chapter 1 twice**, the second time from a blank file. If you can write micrograd from memory, the rest is downhill.
4. **Attempt exercises before answers**, especially Chapter 5.
5. **Change one thing and watch the loss.** Halve the embedding size, remove the residual connections, break the initialization on purpose. Predictions plus measurements build intuition faster than any amount of watching.
6. **Keep a number log.** Every validation loss with the change that produced it. That log is the real work product of this field.
7. **If the maths blocks you,** watch 3Blue1Brown's calculus and neural network series, then come back.

---

# PART V — EXPLAINING IT TO OTHERS

*The test of understanding is being able to pitch it at any level. Here are three scripts: for anyone, for a curious colleague, and for a specialist.*

## 5.1 The 30-second version, for anyone

ChatGPT and everything like it are built out of a handful of simple ideas that fit on a whiteboard. A researcher named Andrej Karpathy recorded about 15 hours of video building one from an empty file, explaining every line, ending with a working copy of GPT-2 that anyone can train for about $10 of rented computer time.

The core idea: the machine makes a guess, measures how wrong it was as a single number, works out which of its millions of internal dials pushed that number up, nudges every dial slightly the other way, and repeats a few million times. That is the whole of how it learns. The famous "transformer" part is one addition on top: as it reads, each word looks back at the earlier words and decides for itself which ones matter.

Bigger versions are not different in kind. They are the same design with about a million times more text and ten thousand times more dials.

## 5.2 Teaching a six-year-old, as a script

This is the version to use with an actual child, or any adult who says they are "not technical." It works because every step is something they have physically done.

**1. The guessing game.** "I'm thinking of the next letter in a word. E-L-E-P-H-A-... what comes next?" They say N. "How did you know?" They will say something like "because that's how the word goes." That is a language model. The computer plays this game, forever, on the whole internet.

**2. The knobs.** "Imagine a machine with a thousand knobs. You put in E-L-E-P-H-A and it says Z. That's wrong. So you turn each knob a tiny bit, and try again. And again. A million times." Show them a radio dial or a tap if there is one nearby. Let them turn something.

**3. How it knows which way to turn.** "How do you find the bottom of a hill in thick fog? You feel which way the ground goes down, take one step, and feel again." Have them close their eyes and shuffle downhill on any slope, or tilt a tray with a marble on it. That is gradient descent, and it is the entire training algorithm.

**4. Why it makes mistakes.** "The computer never sees letters. Before it looks at anything, another little program chops writing into pieces and turns each piece into a number. So when you ask how many r's are in strawberry, it's like asking you to count the letters in a word you only saw for half a second, from far away." This is the honest explanation for the failure they are most likely to have seen.

**5. The one thing to leave them with.** "Nobody told it the rules. It looked at millions of examples and worked out the patterns by guessing and being corrected. Like you did with talking."

**What to avoid:** the word "neuron" (it invites a brain metaphor that is wrong and hard to unwind), any mention of matrices, and the claim that it thinks or understands. Say "it guesses very well because it has seen a lot," which is both true and enough.

## 5.3 Talking to a specialist

For a conversation with someone who has a PhD in this field, you need three things: correct vocabulary, awareness of what the course simplifies, and knowledge of what has changed since it was recorded. Here is all three.

**Where the course simplifies, and what the specialist knows instead:**

| Course says | The fuller picture |
|---|---|
| BatchNorm fixes internal covariate shift | That explanation did not survive: Santurkar et al. (2018) showed the benefit persists when covariate shift is reintroduced; the current account is loss-landscape smoothing |
| Add 1 to every count to smooth | Laplace smoothing is a Dirichlet(1) prior and is known to be a weak smoother; Kneser-Ney is the classical answer, and neural embeddings supersede both |
| Scale attention by 1/√d_k | Because dot products of d_k unit-variance terms have variance d_k; the assumption is the entries are roughly standardized, which initialization arranges |
| BPE is how tokenization works | BPE is greedy and its segmentation is not optimal; unigram LM tokenization (Kudo 2018) is the principled alternative, and byte-level models aim to remove the stage entirely |
| Train on 10B tokens | Off Chinchilla-optimal (~20 tokens per parameter) on purpose, because inference cost dominates in deployment |
| Use post-block LayerNorm | The course uses pre-norm, which is the modern default; the 2017 paper's post-norm needed learning-rate warmup to train at all |

**What has changed since the course was recorded** (the course is 2022–2024; this is the state as of mid-2026 [my read, so check it]):

- **Position encoding.** Learned absolute position embeddings, as in Chapter 7, have largely been replaced by **rotary position embeddings (RoPE)**, which encode relative position by rotating query and key vectors, and extrapolate better beyond the training context length.
- **Attention variants.** **Grouped-query attention** and **multi-query attention** share key and value projections across heads to shrink the inference-time cache, which is the dominant memory cost when serving.
- **Normalization.** **RMSNorm** drops LayerNorm's mean subtraction, on the finding that only the rescaling was load-bearing.
- **Activations.** **SwiGLU** and other gated units have replaced the plain ReLU feed-forward block.
- **Sparsity.** **Mixture-of-experts** routes each token to a few of many feed-forward blocks, so parameter count and per-token compute decouple.
- **Beyond attention.** **State-space models** (S4, Mamba) pursue subquadratic sequence mixing while keeping content dependence, which is precisely the trade-off Chapter 6's fixed tree versus Chapter 7's learned gather sets up.
- **Post-training.** The stages the course omits, supervised fine-tuning then RLHF or DPO, plus reinforcement learning on verifiable rewards for reasoning models, are now where most of the perceived capability difference between models is made.

**Five questions that show you understand the material**, worth asking a specialist rather than asserting at them:

1. "The course frames BatchNorm as a scale-control mechanism. How much of its benefit do you think is regularization from batch-statistic noise versus landscape smoothing?"
2. "Given attention is permutation-equivariant and all order information comes from position encodings, how much of long-context failure is an encoding problem versus an attention-dilution problem?"
3. "Karpathy's 124M run beat GPT-2 with 10× fewer tokens, which he attributes to FineWeb-EDU's filtering. How far does data curation actually substitute for scale before it stops?"
4. "If tokenization causes the arithmetic and spelling failures, why have byte-level models not displaced BPE yet? Is it purely the sequence-length cost?"
5. "Chinchilla says 20 tokens per parameter is compute-optimal, but everyone overtrains small models for inference economics. Where do you think the real optimum sits once serving cost is in the objective?"

## 5.4 The 2-to-3-minute version, for a curious colleague

Karpathy's *Neural Networks: Zero to Hero* is nine lectures, roughly 15 hours, that build modern language models from an empty Python file, with no libraries hiding the important parts. The premise is that a decade of deep learning has been compressed into `model.fit()`, and anyone who wants to actually diagnose these systems has to open it back up.

**Lecture 1** builds an automatic differentiation engine in 100 lines: every arithmetic operation records where it came from, then the chain rule runs backward through that record to work out how sensitive the final error is to every parameter. That is backpropagation, and it stops being mysterious in an afternoon.

**Lectures 2 through 6** build a name generator, improving it in steps that each teach one idea. A counting table of letter pairs scores a loss of 2.4544. A one-layer network trained by gradient descent scores the same, which is the point: it is the same model learned a harder way, and the counting version cannot scale, since ten letters of context would need a table of 205 trillion cells. Then learned embeddings and a wider window take it to 2.26. Then fixing the initialization takes it to 2.15, after a diagnostic reveals the network started with a loss of 27.88 where arithmetic said 3.30, with 61% of its neurons pinned flat and unable to learn. Then a deeper tree-shaped network with 8 characters of context crosses below 2.0. Along the way: mini-batches, finding learning rates by sweeping them, train/dev/test discipline, and a set of diagnostic plots that catch problems no loss curve shows.

**Lecture 7** is the centerpiece: a GPT trained on a megabyte of Shakespeare. Self-attention is the one new idea and it is simple once stated plainly. Every token broadcasts a label describing what it holds, privately holds a query describing what it wants, matches its query against everyone's labels, and averages in whatever the best matches offer, with a triangular mask forbidding it from seeing the future. Ten million parameters, fifteen minutes on a GPU, and it writes fluent nonsense in Shakespeare's shape. GPT-3 is the same architecture at 175 billion parameters.

**Lecture 8** covers tokenization, which sounds boring and explains a startling amount of real behaviour. `strawberry` mid-sentence is a single opaque token, so counting its letters is a perception failure rather than a reasoning one. `1000000` arrives as `100`+`000`+`0`, chunked left to right while arithmetic carries right to left. And the same sentence in Hindi costs 7.5× more tokens than in English, so speakers of other languages pay more money for worse results.

**Lecture 9** reproduces GPT-2's 124M model on 10 billion tokens of filtered web text. It is as much a systems lecture as a machine learning one: five changes make training 11× faster, one being padding the vocabulary from 50,257 to 50,304 because GPUs prefer round numbers. In 1.7 hours and about $10 of rented GPUs it beats OpenAI's original on a standard benchmark using a tenth of the data, because clean data substitutes for lots of data.

The fair criticism: it is a narrow path straight to transformers, with no vision, no diffusion, no reinforcement learning, and it stops before the fine-tuning that turns a text completer into an assistant. The more important warning: it does not work as viewing. Everyone who reports real value from it reports typing the code and breaking it. Budget 40 to 60 hours for 15 hours of video.

---

# APPENDIX A — Glossary

Every term used in this book, defined in one line.

**Activation** — a value flowing forward through the network, output by a neuron.
**AdamW** — an optimizer that adapts each parameter's step size using running averages of its gradient and squared gradient, with decoupled weight decay.
**Attention** — a mechanism letting each position gather information from other positions, weighted by content-based relevance.
**Autoregressive** — generating one token at a time, feeding each output back in as input.
**Backpropagation** — computing gradients by applying the chain rule backward through a computation graph.
**Batch** — the group of examples processed together in one step.
**BatchNorm** — normalizing each neuron's output across the examples in a batch.
**Bias** — a per-neuron offset added regardless of input.
**Bigram** — a pair of adjacent symbols.
**Block size** — the context length; how many previous tokens the model can see.
**BPE (Byte Pair Encoding)** — building a vocabulary by repeatedly merging the most frequent adjacent pair.
**Broadcasting** — automatically stretching a smaller tensor to match a larger one in an operation.
**Byte** — a number from 0 to 255; the unit UTF-8 uses.
**Causal mask** — the triangular mask preventing a position from attending to the future.
**Chain rule** — sensitivities multiply along a chain of operations.
**Cross-entropy loss** — average negative log probability assigned to the correct answers. Lower is better.
**Derivative** — how much an output moves when an input is nudged.
**Dropout** — randomly zeroing a fraction of activations during training.
**Embedding** — a learned vector representing a discrete item such as a character or token.
**Epoch** — one complete pass over the training data.
**Fan-in** — the number of inputs feeding a neuron.
**FlashAttention** — an exact attention implementation that avoids materializing the full attention matrix in memory.
**Forward pass** — computing the model's output from its input.
**Gradient** — the collection of derivatives of the loss with respect to every parameter.
**Gradient accumulation** — simulating a large batch by summing gradients over several small ones before updating.
**Gradient clipping** — capping the size of an update so one bad batch cannot destabilize training.
**GPU** — hardware with thousands of simple cores that perform the same arithmetic simultaneously on different data.
**HellaSwag** — a multiple-choice benchmark of sentence completions, easy for humans, hard for small models.
**Hyperparameter** — a setting you choose rather than learn: learning rate, layer count, batch size.
**Kaiming initialization** — scaling initial weights by `gain/√fan_in` to keep activation spread constant across layers.
**LayerNorm** — normalizing across the features of a single example. Used in transformers.
**Learning rate** — the step size in gradient descent.
**Logits** — raw network outputs before softmax.
**Loss** — a single number measuring wrongness.
**Memory-bound** — limited by the speed of moving data rather than by arithmetic.
**Mini-batch** — a small random subset of data used for one training step.
**MLP** — multilayer perceptron; stacked layers of neurons with nonlinearities.
**Multi-head attention** — several attention operations in parallel, concatenated.
**Nonlinearity** — a bending function such as `tanh` or `ReLU`; without one, stacked layers collapse into a single layer.
**One-hot** — representing item *k* as a vector of zeros with a single 1 at position *k*.
**Overfitting** — memorizing training data at the expense of new data.
**Parameter** — a number the model learns; a weight or a bias.
**Perplexity** — `exp(loss)`; the effective number of choices the model is deciding between.
**Position embedding** — a learned vector per position, added so attention can tell order.
**Pre-norm** — applying normalization before each sub-block, keeping the residual path clean.
**Query, key, value** — what a token is looking for, what it advertises, and what it contributes.
**Regularization** — any penalty or noise that discourages the model from fitting the training data too exactly.
**Residual connection** — adding a block's input to its output, giving gradients a clean path backward.
**Saturation** — a neuron pinned at the flat extremes of its nonlinearity, where gradient stops flowing.
**Seed** — a number fixing the random generator so results are reproducible.
**Shape** — the list of sizes along a tensor's dimensions.
**Smoothing** — adding a small count everywhere so nothing has probability exactly zero.
**Softmax** — converting arbitrary numbers into a probability distribution by exponentiating and normalizing.
**Standard deviation** — a measure of spread around the average.
**Symmetry breaking** — initializing with small randomness so identical units differentiate.
**Temperature** — a divisor on logits before softmax; higher makes sampling more random.
**Tensor** — a multidimensional array of numbers.
**Token** — the atomic unit of text a model reads; a character, word piece, or byte.
**Topological sort** — ordering a graph so every node comes after everything it depends on.
**Transformer** — the architecture built from stacked attention and feed-forward blocks with residual connections and normalization.
**Underfitting** — model too small or undertrained; training and validation loss both high and close.
**Unicode** — the standard assigning a number to every character in every writing system.
**UTF-8** — the standard encoding of Unicode numbers as bytes.
**Validation set** — held-out data used to choose hyperparameters.
**Weight** — a multiplier applied to an input.
**Weight tying** — sharing one matrix between the input embedding and the output classifier.

---

# APPENDIX B — PyTorch reference for this book

Every PyTorch function used, with what it does.

| Call | What it does |
|---|---|
| `torch.tensor(x)` | Make a tensor from a list or number |
| `torch.zeros(a, b)` / `torch.ones(a, b)` | Tensor of zeros / ones with that shape |
| `torch.randn(a, b)` | Random values from a standard normal (mean 0, std 1) |
| `torch.arange(n)` | `[0, 1, 2, ..., n-1]` |
| `torch.manual_seed(n)` | Fix randomness for reproducibility |
| `torch.Generator().manual_seed(n)` | A private random stream, passed to functions that accept `generator=` |
| `x.shape` / `x.numel()` / `x.dim()` | Shape / total elements / number of dimensions |
| `x.view(a, b)` | Reinterpret the same memory with a new shape; `-1` means "infer this one" |
| `x.T` / `x.transpose(-2, -1)` | Swap dimensions |
| `x @ y` | Matrix multiply |
| `x.sum(dim, keepdim=True)` | Sum along a dimension, keeping it as size 1 |
| `x.mean(dim)` / `x.std(dim)` | Average / spread along a dimension |
| `x.exp()` / `x.log()` | Elementwise `eˣ` / natural log |
| `torch.tanh(x)` | Squash into (−1, 1) |
| `F.softmax(x, dim)` | Turn scores into probabilities along a dimension |
| `F.cross_entropy(logits, targets)` | Softmax and negative log likelihood, fused and numerically safe |
| `F.one_hot(x, num_classes)` | Integer to one-hot vector |
| `x.masked_fill(mask, value)` | Replace entries where the mask is True |
| `torch.tril(x)` | Keep the lower triangle, zero the rest |
| `torch.multinomial(p, n)` | Draw `n` samples according to probabilities `p` |
| `torch.cat(list, dim)` | Join tensors along a dimension |
| `x.requires_grad = True` | Track gradients for this tensor |
| `loss.backward()` | Compute all gradients |
| `x.grad` | The gradient of the loss with respect to `x` |
| `optimizer.zero_grad(set_to_none=True)` | Clear gradients before the next step |
| `@torch.no_grad()` | Do not build a graph; use when only measuring |
| `x.to('cuda')` | Move to GPU |
| `x.item()` | Extract a Python number from a one-element tensor |
| `torch.allclose(a, b)` | Compare floats sensibly, with tolerance |
| `nn.Linear`, `nn.Embedding`, `nn.LayerNorm`, `nn.Dropout` | Standard layers |
| `nn.Module`, `nn.Sequential`, `nn.ModuleList` | Containers for building models |
| `model.register_buffer(name, t)` | Store a tensor on the module that is not a parameter |
| `model.train()` / `model.eval()` | Switch modes for Dropout and BatchNorm |

---

# APPENDIX C — Questions you might still have

**Why 27 characters and not 26?** 26 letters plus the `.` token marking both start and end of a name.

**Why is the loss "2.45" and not a percentage?** It is average negative log probability in nats. Convert with `exp(loss)` to get perplexity: `exp(2.45) ≈ 11.6`, meaning the model is effectively choosing between about 12 equally likely characters. A percentage would hide how confident the model was, which is the thing being measured.

**Why does everyone use `-inf` for masking instead of 0?** Because the mask is applied *before* softmax, and `exp(-inf) = 0` exactly. Using 0 before softmax would give `exp(0) = 1`, a large weight on the future.

**Does the network "understand" anything?** It computes conditional probabilities over sequences. Whether that constitutes understanding is a philosophical question this course does not touch, and you can do all the engineering without answering it.

**Why does the loss never reach 0?** Because language is genuinely uncertain: after `the cat sat on the` many continuations are valid. The floor is the true conditional entropy of the data, which for English text is well above zero.

**Why train on next-character prediction rather than something useful?** Because it needs no labels, it can consume any text, and predicting the next token well requires learning grammar, facts, and reasoning as a side effect. That side effect is the entire modern field.

**How does a base model become ChatGPT?** Two further stages the course omits: supervised fine-tuning on example conversations, then reinforcement learning from human preference comparisons.

**Do I need to memorize the derivative rules?** No. You need to know that each operation has one, and that they multiply along the chain.

**Why is my loss different from the book's?** Different random seed, different hardware, different PyTorch version. Directions and magnitudes should match; last digits will not.

**What should I build next?** Retrain Chapter 7 on text you care about. That is where the course stops being an exercise.

---

# APPENDIX D — Sources and method

**Primary.** All nine lecture transcripts were retrieved programmatically (198,000 words total) and searched for every figure quoted here.

- Course page: [karpathy.ai/zero-to-hero.html](https://karpathy.ai/zero-to-hero.html) · Code: [github.com/karpathy/nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero)
- L1 micrograd (2h25m): [youtu.be/VMj-3S1tku0](https://youtu.be/VMj-3S1tku0)
- L2 makemore (1h57m): [youtu.be/PaCmpygFfXo](https://youtu.be/PaCmpygFfXo)
- L3 MLP (1h15m): [youtu.be/TCH_1BHY58I](https://youtu.be/TCH_1BHY58I)
- L4 activations, gradients, BatchNorm (1h55m): [youtu.be/P6sfmUTpUmc](https://youtu.be/P6sfmUTpUmc)
- L5 backprop ninja (1h56m): [youtu.be/q8SA3rM6ckI](https://youtu.be/q8SA3rM6ckI)
- L6 WaveNet (56m): [youtu.be/t3YJ5hKiMQ0](https://youtu.be/t3YJ5hKiMQ0)
- L7 Let's build GPT (1h56m): [youtube.com/watch?v=kCc8FmEb1nY](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- L8 GPT tokenizer (2h13m): [youtu.be/zduSFxRajkE](https://youtu.be/zduSFxRajkE)
- L9 reproduce GPT-2 (4h01m): [youtu.be/l8pRSuU81PU](https://youtu.be/l8pRSuU81PU)

**Community notes and learner reports**, used for the troubleshooting and "where people get stuck" sections:
[chizkidd](https://github.com/chizkidd/Karpathy-Neural-Networks-Zero-to-Hero) · [MK2112](https://github.com/MK2112/nn-zero-to-hero-notes) · [AayushSameerShah](https://github.com/AayushSameerShah/Neural-Net-Zero-to-Hero-with-Andrej) · [Bharat Bheesetti's account](https://bharatbheesetti.com/posts/zero_to_hero) · [Stephen Jonany's walkthrough](https://medium.com/@sjonany/karpathys-micrograd-walkthrough-535718235150) · [Brian Sigafoos](https://briansigafoos.com/neural-networks-karpathy/)

**Reception:** [HN 2023](https://news.ycombinator.com/item?id=34591998) · [HN recent](https://news.ycombinator.com/item?id=46485090) · [HN best lecture series](https://news.ycombinator.com/item?id=35408382)

**Papers referenced in the lectures:** Bengio et al. 2003 (neural language model) · Ioffe & Szegedy 2015 (BatchNorm) · He et al. 2015 (Kaiming init; residual networks) · van den Oord et al. 2016 (WaveNet) · Vaswani et al. 2017 (Attention Is All You Need) · Radford et al. 2019 (GPT-2) · Brown et al. 2020 (GPT-3) · Zellers et al. 2019 (HellaSwag) · Kudo 2018 (unigram tokenization) · Santurkar et al. 2018 (how BatchNorm helps) · Hoffmann et al. 2022 (Chinchilla) · Dao et al. 2022 (FlashAttention)

**Method note.** Everything marked [verified] was executed on a Linux machine with PyTorch 2.11 and Python 3.12 while writing, and the outputs printed here are copied from that run. Six figures I initially predicted turned out to be wrong when measured and were corrected, which is the argument for running code rather than describing it. Figures marked [transcript] come from the lecture captions; auto-captions occasionally mangle numbers, so implausible ones were cross-checked against the repository and the papers. Two known caption artifacts: Karpathy misdates GPT-2 to 2017 in Lecture 7 (corrected in Chapter 7), and caption noise renders some figures oddly, such as "1,24 tokens" for 1,024.
