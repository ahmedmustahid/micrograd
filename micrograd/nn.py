import random
from micrograd.engine import Value

class Neuron:
    def __init__(self, nin) -> None:
        self.w = [random.uniform(-1,1) for _ in range(nin)]
        self.b = random.uniform(-1,1)

    def parameters(self):
        return self.w + [self.b]

    def __call__(self,x):
        act = sum(((wi*xi) for wi, xi in zip(self.w,x)), self.b)
        out = act.tanh()
        return out

class Layer:
    def __init__(self, nin, nout) -> None:
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def parameters(self):
        params = []

        for n in self.neurons():
            ps = n.parameters()
            params.extend(ps)
        return params

    def __call__(self,x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs)==1 else outs

class MLP:
  def __init__(self, nin, nouts):
      sz = [nin] + nouts
      self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
  def __call__(self, x):
      for layer in self.layers:
          x = layer(x)
      return x

  def parameters(self):
      return [p for layer in self.layers for p in layer.parameters()]       