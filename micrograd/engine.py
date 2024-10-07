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
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out
    
    def __rmul__(self, other):
        return self * other
    

    def __pow__(self, other):
        assert isinstance(other, (int, float)), 'other must be int/float'
        out = Value(data=self.data**other, children=(self, ), _op=f'**{other}')

        def _backward():
            self.grad += other*self.data**(other -1) * out.grad
        out._backward = _backward

        return out

    def __truediv__(self, other):
        return self * other**(-1)
    
    def __neg__(self):
        return self * (-1)
    
    def __sub__(self, other):
        return self + (-other)
    
    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)

                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0

        for node in reversed(topo):
            node._backward()



