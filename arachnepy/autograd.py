import numpy as np
class ScalarSpyder:

    __array_priority__ = 1000

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

import numpy as np 
import numbers 
class Spyder:

    __array_priority__ = 1000

    def __init__(self, data: np.ndarray, parents = None, requires_grad = True):
        self.data = np.asarray(data, dtype=float)
        self.grad = np.zeros_like(self.data)
        self.parents = parents if parents else []
        self._backward = lambda: None
        self.shape = self.data.shape
        self.ndim = self.data.ndim
        self.requires_grad = requires_grad

    def _unbroadcast(self, grad, original_shape) -> np.ndarray:
        """helper function to handle 2D broadcasting of gradient across 2D array"""
    
        temp_grad = np.asarray(grad)

        while temp_grad.ndim > len(original_shape): # remove extra dimensions
            temp_grad = np.sum(temp_grad, axis = 0)
        
        for axis, size in enumerate(original_shape):
            # sum across stretched axis
            if size == 1 and temp_grad.shape[axis] != 1: 
                # if no. of elements is 1 but grad has more than 1 element
                temp_grad = np.sum(temp_grad, axis = axis, keepdims = True)

        return temp_grad.reshape(original_shape)

    def unstick(self):
        """return numpy array with now gradient tracking"""
        return self.data
    
    def _arachnify(self, other):
        """function to coerce other into a spyder"""
        if isinstance(other, Spyder):
            return other
        if isinstance(other, (numbers.Real, np.generic, np.ndarray)):
            return Spyder(other)
        raise TypeError(
                f"Received object of type {type(other)}, was unable to coerce to Spyder object"
                f"\nOnly supporting scalars, np.ndarrays and other spyders!"
                )
    
    def __repr__(self):
        return f"Spyder(data = {self.data}, grad = {self.grad}, parents = {self.parents})"

    def __matmul__(self, other):
        """matric multiplication supporting only upto 2 dimensions"""
        other = self._arachnify(other)

        if self.ndim not in (1, 2) or other.ndim not in (1, 2):
            raise ValueError("Only supporting 1D and 2D matmul!\n"
                             f"Received arrays of dimensions {self.ndim}, {other.ndim}.")

        if self.shape[-1] != other.shape[0]:
            raise ValueError("Shape mismatch!"
                             f"Received arrays of shapes {self.shape}, {other.shape}")

        out = Spyder(self.data @ other.data, parents = [self, other])
        a, b = self.data, other.data

        def _backward():
            if self.ndim == 2 and other.ndim == 2:
                self.grad += out.grad @ b.T
                other.grad += a.T @ out.grad

            elif self.ndim == 1 and other.ndim == 1:
                self.grad += out.grad * b
                other.grad += out.grad * a

            elif self.ndim == 2 and other.ndim == 1:
                self.grad += np.outer(out.grad, b)
                other.grad += a.T @ out.grad

            elif self.ndim == 1 and other.ndim == 2:
                self.grad += out.grad @ b.T
                other.grad += np.outer(out.grad, a)

        out._backward = _backward
        return out
    
    def __pow__(self, other):
        assert (isinstance(other, (numbers.Real, np.generic))), "only supporting int/float powers for now"

        out = Spyder(self.data ** other, parents = [self]) 

        def _backward():
            self.grad += (other * (self.data) ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def __add__(self, other):        
        other = self._arachnify(other)
        if other.ndim > 2 or self.ndim > 2:
            raise ValueError("Only supporting operations on "
                             "scalars, 1D vectors, or 2D matrices!\n"
                             f"Received arrays of dimensions: {self.ndim}, {other.ndim}")
        
        out = Spyder(self.data + other.data, parents=[self, other])

        def _backward():
            self.grad += self._unbroadcast(out.grad, self.shape)
            other.grad += other._unbroadcast(out.grad, other.shape)
        
        out._backward = _backward
        return out

    def __sub__(self, other):
        other = self._arachnify(other)
        out = Spyder(self.data - other.data, parents=[self, other])

        def _backward():
            self.grad += self._unbroadcast(out.grad, self.shape)
            other.grad += other._unbroadcast(-out.grad, other.shape)

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = self._arachnify(other)
        out = Spyder(self.data * other.data, parents=[self, other])
    
        def _backward():
            self.grad += self._unbroadcast(out.grad * other.data, self.shape)
            other.grad += other._unbroadcast(out.grad * self.data, other.shape)
            
        out._backward = _backward
        return out

    def __truediv__(self, other):
        other = self._arachnify(other)

        return self * (other ** -1)

    def __neg__(self):
        out = Spyder(-self.data, parents = [self])
        def _backward():
            self.grad -= out.grad
        out._backward = _backward
        return out

    def __rsub__(self, other):
        return -self + other 

    def __radd__(self, other):
        return self + other # addition is commutative 

    def __rmul__(self, other):
        return self * other 

    def __rtruediv__(self, other):
        return other * (self ** -1) 

    def sum(self, axis = None, keepdims = False):
        out = Spyder(np.sum(self.data, axis = axis, keepdims=keepdims), parents = [self,])

        def _backward():
            if axis is None:
                self.grad += np.ones_like(self.data) * out.grad
                return
            grad = out.grad if keepdims else np.expand_dims(out.grad, axis=axis)    
            self.grad += np.broadcast_to(grad, self.data.shape)
                
        out._backward = _backward
        return out

    def mean(self, axis = None, keepdims = False):
        n = self.data.size if axis is None else np.size(self.data, axis = axis)
        return self.sum(axis = axis, keepdims = keepdims) / n

    def max(self, axis = None, keepdims = True):
        out_data = np.max(
            self.data,
            axis = axis,
            keepdims = keepdims
        )

        mask_data = out_data if keepdims else np.expand_dims(out_data, axis)

        out = Spyder(out_data, parents = [self])

        def _backward():
            mask = (self.data == mask_data)
            counts = np.sum(mask, axis = axis, keepdims = True)

            if keepdims or axis is None:
                grad = out.grad
            else:
                grad = out.grad if keepdims else np.expand_dims(out.grad, axis = axis)

            self.grad += (
                mask * np.broadcast_to(grad, self.data.shape) / counts
                )

        out._backward = _backward
        return out


    def __getitem__(self, idx):
        out = Spyder(self.data[idx], parents = [self])

        def _backward():
            self.grad[idx] += out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Spyder(np.maximum(0, self.data), parents = [self])

        def _backward():
            self.grad += (self.data > 0) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        x = self.data

        out_data = np.empty_like(x)

        positive = x >= 0 # prepare boolean mask
        negative = ~positive

        out_data[positive] = 1 / (1 + np.exp(-x[positive])) 
        out_data[negative] = np.exp(x[negative]) / (1 + np.exp(x[negative]))

        out = Spyder(out_data, parents = [self])

        def _backward():
            self.grad += out.data * (1 - out.data) * out.grad

        out._backward = _backward
        return out

    def log(self):
        x = np.clip(self.data, 1e-8, None)
        out = Spyder(np.log(x), parents = [self]) # clip x value to make sure log doesnt go boom

        def _backward():
            mask = (self.data >= 1e-8)
            self.grad += mask * (1 / x) * out.grad 

        out._backward = _backward
        return out

    def exp(self):
        out_data = np.exp(self.data)
        out = Spyder(out_data, [self])

        def _backward():
            self.grad += out_data * out.grad 

        out._backward = _backward
        return out
    
    def retrace(self):
        if self.requires_grad == False:
            raise AttributeError("No gr")
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for parent in v.parents:
                    build_topo(parent)
                topo.append(v)
        build_topo(self)

        self.grad = np.ones_like(self.data)
        for v in reversed(topo): 
            v._backward()

    def clean_webs(self):
        visited = set()
        def visit_parents(v):
            if v not in visited:
                visited.add(v)
                v.grad = np.zeros_like(v.data)
                for parent in v.parents:
                    visit_parents(parent)
        visit_parents(self)