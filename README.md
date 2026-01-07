# ML From Scratch

A project implementing machine learning algorithms from scratch using only PyTorch's autograd.

> This project reorganizes learning materials from **University of Chicago STAT 37710 (Machine Learning)** into a modern structure.

## Features

- **From-scratch implementation**: Built without `torch.nn.Linear`, `torch.optim.SGD`, etc.
- **PyTorch autograd only**: Backpropagation implemented using only automatic differentiation
- **Learning-friendly**: Numbered notebooks for step-by-step learning
- **sklearn-style API**: `fit()`, `predict()`, `transform()` interface

## Quick Start

### 1. Environment Setup

```bash
# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup project
git clone https://github.com/username/ml-from-scratch.git
cd ml-from-scratch
uv sync
```

### 2. Run Notebooks

```bash
uv run jupyter notebook notebooks/
```

### 3. Data

- Datasets like MNIST, CIFAR-10 are **automatically downloaded on first run**
- No additional setup required!

## Project Structure

```
ml-from-scratch/
├── mlfs/                          # Core package
│   ├── nn/                        # Neural network modules
│   │   ├── layers.py              # Linear, Conv2d, MaxPool2d
│   │   ├── activations.py         # ReLU, Sigmoid, Softmax
│   │   ├── losses.py              # CrossEntropyLoss, MSELoss
│   │   ├── optim.py               # SGD, Adam
│   │   └── models.py              # MLP, CNN
│   ├── classical/                 # Traditional ML
│   │   ├── clustering.py          # KMeans, SpectralClustering
│   │   ├── reduction.py           # PCA, LLE, Isomap
│   │   └── ensemble.py            # AdaBoost
│   ├── probabilistic/             # Probabilistic models
│   │   ├── em.py                  # EM Algorithm, GMM
│   │   └── sparse.py              # Sparse Coding
│   └── utils/                     # Utilities
│       ├── data.py                # Data loading
│       └── viz.py                 # Visualization
└── notebooks/                     # Tutorial notebooks
```

## Learning Path

| # | Notebook | Topic | Key Concepts |
|---|----------|-------|--------------|
| 01 | `perceptron` | Perceptron | Single neuron, Gradient descent |
| 02 | `mlp` | Multi-layer Perceptron | Backpropagation, Hidden layers |
| 03 | `cnn_mnist` | CNN (MNIST) | Convolution, Pooling |
| 04 | `cnn_cifar` | CNN (CIFAR) | Color images, Deep networks |
| 05 | `kmeans` | K-Means | Clustering, K-Means++ |
| 06 | `spectral_clustering` | Spectral Clustering | Graph Laplacian |
| 07 | `pca` | PCA | Dimensionality reduction, Eigendecomposition |
| 08 | `lle` | LLE | Locally Linear Embedding |
| 09 | `isomap` | Isomap | Geodesic distance, Manifold |
| 10 | `ensemble` | Ensemble | AdaBoost, Weak classifiers |
| 11 | `em_algorithm` | EM Algorithm | GMM, Expectation Maximization |
| 12 | `face_detection` | Face Detection | Haar-like, Viola-Jones |
| 13 | `sparse_coding` | Sparse Coding | Sparse representation, Dictionary |

## PyTorch Usage Rules

### Allowed (Can use)
```python
import torch
import torch.nn.functional as F

# Tensor operations
x = torch.randn(32, 784)
y = x @ weight + bias

# Automatic differentiation
loss.backward()
param.grad

# Basic functions
F.relu(x), F.softmax(x, dim=1)
```

### Prohibited (Must implement)
```python
# Layers - must implement yourself
torch.nn.Linear      # → mlfs.nn.layers.Linear
torch.nn.Conv2d      # → mlfs.nn.layers.Conv2d

# Optimizers - must implement yourself
torch.optim.SGD      # → mlfs.nn.optim.SGD
torch.optim.Adam     # → mlfs.nn.optim.Adam
```

## Usage Example

```python
from mlfs.nn.layers import Linear
from mlfs.nn.optim import SGD
from mlfs.utils.data import load_mnist

# Load data
X_train, y_train = load_mnist(train=True)

# Create model (custom implemented layers)
layer = Linear(784, 10)

# Forward pass
output = layer.forward(X_train[:32])

# Optimizer (custom implemented)
optimizer = SGD(layer.parameters(), lr=0.01)
```

## License

MIT License

## References

- This project was created for educational purposes.
- Based on University of Chicago STAT 37710 (Machine Learning) course materials.
