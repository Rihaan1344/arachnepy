import numpy as np
from arachnepy.autograd import Spyder


x = Spyder(np.array([
    [0., 0.],
    [0., 1.],
    [1., 0.],
    [1., 1.]
]))

y = np.array([
    [0.],
    [1.],
    [1.],
    [0.]
])

lr = 1

initialization_strength = 0.1

rng = np.random.default_rng()
architecture = [2, 6, 1]

weights = []
biases = []

for n in range(len(architecture) - 1):

    in_features = architecture[n]
    out_features = architecture[n + 1]

    weights.append(
        Spyder(
            rng.normal(
                loc=0,
                scale=initialization_strength,
                size=(in_features, out_features)
            )
        )
    )

    biases.append(
        Spyder(
            rng.normal(
                loc=0,
                scale=initialization_strength,
                size=(out_features,))
        )
    )


for epoch in range(5000):

    inp = x

    # Hidden layers
    for w, b in zip(weights[:-1], biases[:-1]):
        inp = (inp @ w + b).sigmoid()

    # Output layer
    pred = (inp @ weights[-1] + biases[-1]).sigmoid()

    error = (
        -(y * pred.log() + (1 - y) * (1 - pred).log())
    ).mean()

    error.retrace()

    print("loss:", error.data)

    for w, b in zip(weights, biases):
        w.data -= lr * w.grad
        b.data -= lr * b.grad

    if epoch % 500== 0:
        print(f"epoch {epoch}, loss = {error.data}")