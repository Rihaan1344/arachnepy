# ArachnePy

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

*ArachnePy* is a small machine learning library I built to let me see and understand the inner working of how larger libraries like sklearn, pytorch, etc... work.

It was really fun building this, and if you're curious, you can look at the notebooks(exp-nb1 and autograd, in that order) to see how I progressed from a simple sum function to a mini tensor autograd & multi-layer neural networks with softmax!

The autograd system takes heavy inspiration from [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd), with an extension of some tensor operations.

## Installation

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

