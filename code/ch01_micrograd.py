# Chapter 1 — micrograd: a scalar autograd engine and a 41-parameter MLP.
# Requires: no dependencies beyond the standard library. Runs in under a second.
#
# From "Neural Networks: Zero to Hero — The Textbook".

import math, random

class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
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
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out
    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def backward(self):
        topo, visited = [], set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

# --- demo 1: numerical derivative
def f(x): return 3*x**2 - 4*x + 5
h = 0.001; x = 3.0
print("f(3.0) =", f(3.0))
print("numerical derivative at x=3:", (f(x+h) - f(x)) / h)

# --- demo 2: a tiny graph
a = Value(2.0); b = Value(-3.0); c = Value(10.0)
d = a*b + c
d.backward()
print("d =", d.data, "| a.grad =", a.grad, "| b.grad =", b.grad, "| c.grad =", c.grad)

# --- demo 3: the MLP
class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0.0)
    def __call__(self, x):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()
    def parameters(self): return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout): self.neurons = [Neuron(nin) for _ in range(nout)]
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    def parameters(self): return [p for n in self.neurons for p in n.parameters()]

class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
    def __call__(self, x):
        for layer in self.layers: x = layer(x)
        return x
    def parameters(self): return [p for l in self.layers for p in l.parameters()]

random.seed(1337)
n = MLP(3, [4, 4, 1])
print("number of parameters:", len(n.parameters()))

xs = [[2.0,3.0,-1.0],[3.0,-1.0,0.5],[0.5,1.0,1.0],[1.0,1.0,-1.0]]
ys = [1.0, -1.0, -1.0, 1.0]

for k in range(21):
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
    for p in n.parameters(): p.grad = 0.0
    loss.backward()
    for p in n.parameters(): p.data += -0.05 * p.grad
    if k % 5 == 0 or k == 20:
        print(f"step {k:2d}  loss {loss.data:.6f}")
print("final predictions:", [round(y.data, 4) for y in [n(x) for x in xs]])
