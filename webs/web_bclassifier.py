from spyder.spyder import Spyder
import numpy as np
class Web: 
    def __init__(self, layer_size, learning_rate = 0.001, intialization_strength = 0.01, threshold = 0.5, random_state = 11, epochs = 1000):

        """
        x, y -> np.ndarray or spyder
        layer_size -> (input features, layer 1 neurons, layer 2 neurons, ..., outputs)
        """
        self.rng = np.random.default_rng(random_state)
        self.layer_size = layer_size
        self.learning_rate = learning_rate
        self.intialization_strength = intialization_strength
        self.epochs = epochs
        self.threshold = threshold

    def spin(self, x, y):

        x = x if isinstance(x, Spyder) else Spyder(x)
        y = y if isinstance(y, Spyder) else Spyder(y)

        self.weights = []
        self.biases = []

        for n in range(len(self.layer_size) - 1):
        
            in_features = self.layer_size[n]
            out_features = self.layer_size[n + 1]
        
            self.weights.append(Spyder(self.rng.normal(loc = 0, scale = self.intialization_strength, size = (in_features, out_features))))
            self.biases.append(Spyder(self.rng.normal(loc = 0, scale = self.intialization_strength, size = (out_features,))))

        for epoch in range(self.epochs):
            inp = x
            for w, b in zip(self.weights[:-1], self.biases[:-1]):
                inp = (inp @ w + b).sigmoid()

            pred = (inp @ self.weights[-1] + self.biases[-1]).sigmoid()
            error = (-(y * pred.log() + (1 - y) * (1 - pred).log())).mean() 
            
            error.retrace()

            for w, b in zip(self.weights, self.biases):
                w.data -= self.learning_rate * w.grad
                b.data -= self.learning_rate * b.grad

            if epoch % 50 == 0:
                print(f"epoch {epoch}, loss = {error.data}")

    def predict_proba(self, x):
        inp = x if isinstance(x, Spyder) else Spyder(x)
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            inp = (inp @ w + b).sigmoid()
        pred = inp @ self.weights[-1] + self.biases[-1]
        return pred

    def predict(self, x):
        return ((self.predict_proba(x)).sigmoid().data >= self.threshold)

    def score(self, x, y):
        return np.mean(y == self.predict(x)) # accuracy