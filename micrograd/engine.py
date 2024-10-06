import math


class Value:
    def __init__(self, data, children=(), _op='', label='') -> None:
        self.data = data
        self.children = children
        self._prev = set(children)
        self._op = _op
        self.label = label
        self.grad = 0.0
        self._backward = lambda : None

    def __repr__(self) -> str:
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(data=self.data + other.data, children=(self, other), _op='+')
        
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out

    def __radd__(self, other): #other + self
        return self + other
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(data=self.data * other.data, children=(self, other), _op='*')

        def _backward():
            self.grad += other.grad * out.grad
            other.grad += self.grad * out.grad
        self._backward = _backward

        return out
    
    def __rmul__(self, other):
        return self * other
    




