import numpy as np
from typing import Literal

class LinearLoom:
    def __init__(self, solver: Literal["sgd", "bgd"] = "bgd"):
        self.solver = solver

    def spin(self, x, y, learning_rate = 0.01, initialization_strength = 0.01, verbose = 0, epochs=1000):

        self.weights = np.random.randn(x.shape[1]) * initialization_strength
        self.bias = 0.0

        if self.solver == "sgd":
            for epoch in range(epochs):

                total_loss = 0

                for xi, yi in zip(x, y):

                    prediction = self.weights @ xi + self.bias
                    error = prediction - yi

                    total_loss += error**2

                    djdw = 2 * error * xi
                    djdb = 2 * error

                    self.weights -= learning_rate * djdw
                    self.bias -= learning_rate * djdb

                if verbose > 0 and epoch % 100 == 0:
                    print("Epoch:", epoch, "MSE:", total_loss / x.shape[0])

        elif self.solver == "bgd":

            x, y = x.astype(np.float32), y.astype(np.float32)

            for epoch in range(epochs):

                prediction = x @ self.weights + self.bias
                error = prediction - y

                djdw = (2 / len(x)) * (x.T @ error)
                djdb = (2 / len(x)) * np.sum(error)

                self.weights -= learning_rate * djdw
                self.bias -= learning_rate * djdb

                if verbose > 0 and epoch % 50 == 0:
                    print(epoch, np.mean(error**2))

    def predict(self, x):
        return (x @ self.weights + self.bias)

    def score(self, x, y):
        y_pred = self.predict(x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        return 1 - ss_res/ss_tot

class LogisticLoom:
    def __init__(self, solver: Literal["sgd", "bgd"] = "bgd", threshold = 0.5):
        self.solver = solver
        self.threshold = threshold

    def sigmoid(x):
        return 1 / (1 + np.e**(-x))

    def spin(self, x, y, learning_rate = 0.01, initialization_strength = 0.01, verbose = 0, epochs=1000):

        self.weights = np.random.randn(x.shape[1]) * initialization_strength
        self.bias = 0.0

        if self.solver == "sgd":

            for epoch in range(epochs):

                total_loss = 0

                for xi, yi in zip(x, y):

                    prediction = self.weights @ xi + self.bias # linear result
                    prediction = self.sigmoid(prediction) # sigmoid squishes into probabilities
                    prediction = np.clip(prediction, 1e-15, 1-1e-15) # prevent prediction = 1, sigmoid = -inf, and training error.
                    error = prediction - yi

                    total_loss += -(yi*np.log(prediction) + (1-yi)*np.log(1-prediction))

                    djdw = error * xi
                    djdb = error

                    self.weights -= learning_rate * djdw
                    self.bias -= learning_rate * djdb

                if verbose > 0 and epoch % 100 == 0:
                    print("Epoch:", epoch, "Cross-Entropy Loss:", total_loss / len(x.shape[0]))

        elif self.solver == "bgd":

            x = x.astype(np.float32)

            for epoch in range(epochs):
                        
                prediction = x @ self.weights + self.bias # linear result
                prediction = self.sigmoid(prediction) # sigmoid squishes into probabilities
                prediction = np.clip(prediction, 1e-15, 1-1e-15) # prevent prediction = 1, sigmoid = -inf, and training error.
                error = prediction - y

                total_loss = -np.mean(y*np.log(prediction) + (1-y)*np.log(1-prediction))

                djdw = (x.T @ error) / x.shape[0]
                djdb = np.mean(error)

                self.weights -= learning_rate * djdw
                self.bias -= learning_rate * djdb

                if verbose > 0 and epoch % 100 == 0:
                    print("Epoch:", epoch, "Cross-Entropy Loss:", total_loss)

    def predict_proba(self, x):
        return self.sigmoid(x @ self.weights + self.bias)

    def predict(self, x):
        return (self.predict_proba(x) >= self.threshold).astype(np.int32)

    def score(self, x, y):
        return np.mean(y == self.predict(x)) # accuracy