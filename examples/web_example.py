from arachnepy import WebClassifier
from sklearn.datasets import load_iris

data = load_iris()

X = data.data
y = data.target

web = WebClassifier(
    layer_size=(4, 8, 8, 3),
    learning_rate=0.1,
    intialization_strength = 0.5,
    epochs=1500,
    random_state=11
)

web.spin(X, y)
pred = web.predict(X)
print(f"Predictions : {pred[:10]}")
print(f"Accuracy: {web.score(X, y)}")