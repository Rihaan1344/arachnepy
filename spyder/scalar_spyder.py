import numpy as np
class ScalarSpyder:

    def __init__(self, data, parents = None):
        self.data = data
        self.grad = 0
        self.parents = parents if parents else []
        self._backward = lambda: None

    def __mul__(self, other):
        other = other if isinstance(other, ScalarSpyder) else ScalarSpyder(other)
        out = ScalarSpyder(self.data * other.data, parents = [self, other])

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __add__(self, other):
        other = other if isinstance(other, ScalarSpyder) else ScalarSpyder(other)
        out = ScalarSpyder(self.data + other.data, parents = [self, other])

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, ScalarSpyder) else ScalarSpyder(other)
        out = ScalarSpyder(self.data - other.data, parents = [self, other])

        def _backward():
            self.grad += out.grad
            other.grad -= out.grad

        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only int/float values supported"
        out = ScalarSpyder(self.data ** other, parents = [self,])

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def __repr__(self):
        return f"ScalarSpyder(data: {self.data}, grad: {self.grad})"

    def retrace(self):
        self.clean_web()
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited: 
                visited.add(v)
                for parent in v.parents:
                    build_topo(parent)
                topo.append(v)
        build_topo(self)
        
        self.grad = 1

        for v in reversed(topo):
            v._backward()

    def clean_web(self):
        visited = set()
        def visit_children(v):
            if v not in visited: 
                visited.add(v)
                v.grad = 0

                for parent in v.parents:
                    visit_children(parent)
        visit_children(self)

# %%
x = ScalarSpyder(2)
a = x**2


