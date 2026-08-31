import numpy as np
from spyder.spyder import Spyder
class Web: 
    def __init__(self, layer_size, learning_rate = 0.001, intialization_strength = 0.01, random_state = 11, epochs = 1000):

        """
        x, y -> np.ndarray 
        x.shape = (samples, features)
        y.shape = (samples,)
        target is expected to be label encoded.
        layer_size -> (input features, layer 1 neurons, layer 2 neurons, ..., outputs)
        """
        self.rng = np.random.default_rng(random_state)
        self.layer_size = layer_size
        self.learning_rate = learning_rate
        self.intialization_strength = intialization_strength
        self.epochs = epochs

    def spin(self, x, y):

        y = np.expand_dims(y, axis = 1)
        self.distribution = np.unique(y)
        y = np.array(self.distribution == y, dtype = float)

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
                inp = (inp @ w + b).relu()

            logits = (inp @ self.weights[-1] + self.biases[-1])
            logits = logits - logits.max(axis = 1, keepdims = True)
            pred = logits.exp() / logits.exp().sum(axis = 1, keepdims = True)
            error = (-(y * pred.log())).sum(axis = 1, keepdims = True).mean()
            
            error.retrace()

            for w, b in zip(self.weights, self.biases):
                w.data -= self.learning_rate * w.grad
                b.data -= self.learning_rate * b.grad

            if epoch % 50 == 0:
                print(f"epoch {epoch}, loss = {error.data}")

    def predict_proba(self, x):
        inp = x if isinstance(x, Spyder) else Spyder(x) # coerce inputs to spyder

        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            inp = (inp @ w + b).relu() # relu activation for hidden layers

        logits = (inp @ self.weights[-1] + self.biases[-1]) # logits from final layer
        logits = logits - logits.max(axis = 1, keepdims = True)  # shift logits before .exp() to get smaller values

        pred = logits.exp() / logits.exp().sum(axis = 1, keepdims = True) # softmax
        
        return pred  # return probability distribution

    def predict(self, x):
        return self.distribution[np.argmax(self.predict_proba(x).data, axis = 1)]
    
    def score(self, x, y):
        return np.mean(y == self.predict(x)) # accuracy