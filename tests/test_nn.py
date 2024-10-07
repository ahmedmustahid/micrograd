from micrograd.nn import Neuron
from micrograd.engine import Value
import random

random.seed(10)


def test_neuron():
    x = [Value(data=random.uniform(10,25)) for _ in range(3)]
    n = Neuron(3)
    out = n(x)
    assert abs(out.data - 1.0) < 1e-6

