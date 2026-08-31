# ArachnePy

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

*ArachnePy* is a small machine learning library I built to let me see and understand the inner workings behind larger machine-learning libraries like scikit-learn and PyTorch.

It was really fun building this, and if you're curious, you can look at the notebooks(exp-nb1 and autograd, in that order) to see how I progressed from a simple sum function to a mini tensor autograd & multi-layer neural networks with softmax!

The autograd system takes heavy inspiration from [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd), with an extension of some tensor operations.

# Installation

```bash
pip install git+https://github.com/Rihaan1344/arachnepy.git
```

Or clone locally:

```bash
git clone https://github.com/Rihaan1344/arachnepy.git

cd ArachnePy

pip install -e .
```

# Features
1. Linear models - Linear Regression and Logistic Regression
2. Multi-layer Neural Networks - for Regression, Binary Classification, and Multi-class Classification.
3. Autograd systems - both Scalar and Tensor.

# Note

This library was not built for the purpose of being fast. It has been built purely as a learning experience. If you're looking to try making these things from scratch, I would highly recommend looking at the source code in these files. In my opinion, the architecture programmed here is far simpler than those seen in more sophisticated libraries, hence more understandable :)

Also do note that all these models have been modelled with ONLY batch gradient descent, without any optimizers. So yes, this library is still quite limited, but perhaps I'll return and add more stuff in the future.

## Project Structure

```text
Spyder
├── ScalarSpyder
└── Tensor Spyder

Loom
├── LinearLoom
└── LogisticLoom

Web
├── WebRegressor
└── WebClassifier
```

# Naming Convention

| Name | Meaning | Example|
| --- | --- | --- |
| Loom | Linear model / 'Regression'| Linear Loom, Logistic Loom |
| Spyder | Autograd system | Spyder, ScalarSpyder |
| Web | Neural Network | WebRegressor, WebClassifier|
| Retrace | Backward function | ` loss.retrace() / Spyder.retrace() `|
| Spin | .fit() / training | ` web.spin(), loom.spin()` |

# Quick Example

```python
from arachnepy import WebClassifier
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

x, y = load_iris(return_X_y=True, as_frame=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size = 0.33,
    random_state = 11
)

x_scaler = StandardScaler()
x_train, x_test = x_scaler.fit_transform(x_train.to_numpy()), x_scaler.transform(x_test.to_numpy())

web = WebClassifier(
    [x_train.shape[1], 10, 5, np.unique(y_train).size], learning_rate = 0.5, 
    initialization_strength = 0.01, 
    epochs = 1500
    )

web.spin(x_train, y_train.to_numpy())

pred = web.predict(x_train)

print("Train Accuracy:", np.mean(pred == y_train.to_numpy()))

pred = web.predict(x_test)

print("Test accuracy:", np.mean(pred == y_test.to_numpy()))
```

# Results

Current implementation stats:

| Model | Dataset | Accuracy |
|---------|---------|---------|
| WebClassifier | Iris | 96% test accuracy, 100% train accuracy|
| WebRegressor | Salary Prediction, synthetic dataset with ~250k records | ~97% R2 score after training on only 5k records(due to computational limitations)

# Future Plans

- Additional activation functions
- Optimizers (Momentum, RMSProp, Adam)
- More tensor operations(broadcasting!)
- Convolutional layers

