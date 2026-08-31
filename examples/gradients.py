import numpy as np
from arachnepy import Spyder

x = Spyder(np.array([1.0, 2.0, 3.0]))
w = Spyder(np.array([0.5, 2.0, -1.0]))
b = Spyder(2.0)

y = (x * w).sum() + b
loss = y ** 2

loss.retrace()

print("output:", y.data)
print("loss:", loss.data)

print("dx:", x.grad)
print("dw:", w.grad)
print("db:", b.grad)