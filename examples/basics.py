import numpy as np
from arachnepy import Spyder

# example 1

x = Spyder(np.array([2.0, 3.0]))

y = x * x + 2 * x + 1

y.retrace()

print("y:", y.data)
print("dy/dx:", x.grad)

print("-" * 50)
# example 2

x = Spyder(np.array([1.0, 2.0, 3.0]))

y = x ** 2
y = y.sum()

y.retrace()

print("x:", x.data)
print("y:", y.data)
print("gradient:", x.grad)