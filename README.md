# Neural Network From Scratch

> Built using only NumPy. No PyTorch. No TensorFlow. 

A fully functional neural network implemented from first principles — every operation derived mathematically and coded by hand. Trained on MNIST handwritten digit recognition, achieving **96.93% test accuracy**.

---

## Why This Project

I came into this project sceptical about how to approach machine learning — whether to study theory first or build immediately. I chose to build. With no prior Python experience, I picked up the language on the go while implementing every concept from scratch. The goal was simple: understand what actually happens inside a neural network, not just how to call library functions. This project is proof of that understanding and the foundation for everything that comes next.

---

## Project Structure

```
neural-network-from-scratch/
├── matrix.py                  # Matrix operations in raw Python
├── matrix_numpy.py            # Same operations in NumPy (benchmarked)
├── neural_network.py          # Full neural network implementation
└── training_demo.ipynb        # MNIST training with visualizations
```

---

## Phases

### Phase 1 — Matrix Operations

Implemented core matrix operations from scratch in raw Python lists, then reimplemented using NumPy to understand the performance gap.

**Functions implemented:**

| Function | Mathematical Operation |
|---|---|
| `multiply(A, B)` | Element (i,j) = dot product of row i of A with column j of B |
| `transpose(A)` | Maps A_ij → A_ji, converts m×n to n×m |
| `dot_product(u, v)` | Computes Σ u_i · v_i |
| `scalar_multiply(A, c)` | Multiplies every element A_ij by scalar c |

Time complexity of matrix multiplication: **O(n³)** — three nested loops, each running n times.

**Benchmark results (500×500 matrix multiplication):**

| Method | Time | 
|---|---|
| Raw Python | 4.608s |
| NumPy | 0.00188s |
| Speedup | **~2,450x** |

NumPy achieves this by calling optimized BLAS routines written in compiled C/Fortran rather than interpreted Python loops.

---

### Phase 2 — Forward Pass

Built the neural network's prediction machinery from scratch.

**Implemented:**
- `sigmoid(z)` — squishes values to (0, 1)
- `relu(z)` — max(0, z), introduces nonlinearity
- `softmax(z)` — converts raw scores to probability distribution, numerically stabilized
- `Layer` class — stores weights, biases, and intermediate values (z, inputs) for backpropagation
- `NeuralNetwork` class — chains layers together, automatically assigns ReLU to hidden layers and Softmax to output layer

**Key insight:** A network with linear activations everywhere collapses into a single matrix multiplication regardless of depth. Nonlinear activations (ReLU) allow the network to learn complex, non-linearly separable patterns.

**Weight initialization:**
- Hidden layers (ReLU): He initialization — `weights * sqrt(2/n)` — compensates for ReLU zeroing half its inputs
- Output layer (Softmax): Xavier initialization — `weights * sqrt(1/n)`

---

### Phase 3 — Backpropagation

Derived and implemented the full backpropagation algorithm from first principles using the chain rule.

**Derivation for a single layer:**

$$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial W} = \frac{2(\hat{y}-y)}{n} \cdot \mathbb{1}[z>0] \cdot X$$

**Softmax + Cross-Entropy gradient simplification:**

When softmax and cross-entropy are combined, their gradients cancel elegantly:

$$\frac{\partial L}{\partial z} = \hat{y} - y$$

This is derived by expanding the chain rule through both functions simultaneously — the softmax Jacobian and cross-entropy derivative cancel, leaving only the prediction error.

**Verified learning on random data:**

| | Loss |
|---|---|
| Start | 2.47 |
| After 100 steps | 0.26 |
| Reduction | ~90% |

---

### Phase 4 — MNIST Training

Trained on the full MNIST dataset of 70,000 handwritten digit images.

**Architecture:** `[784 → 128 → 64 → 10]`

**Training configuration:**

| Parameter | Value |
|---|---|
| Epochs | 20 |
| Batch size | 32 |
| Learning rate | 0.01 |
| Loss function | Cross-entropy |
| Optimizer | Mini-batch gradient descent |

**Results:**

| Epoch | Loss | Test Accuracy |
|---|---|---|
| 1 | 1192.96 | 90.21% |
| 5 | 332.62 | 94.52% |
| 10 | 205.41 | 95.99% |
| 20 | 109.03 | **96.93%** |

---

## Key Concepts Implemented From Scratch

- Matrix multiplication — O(n³) complexity, NumPy vectorization
- ReLU and Sigmoid — activation functions with hand-derived derivatives
- Softmax — numerically stable implementation (max subtraction trick)
- Cross-entropy loss — derived from maximum likelihood estimation
- Backpropagation — chain rule applied through multiple layers
- He initialization — variance 2/n, designed for ReLU networks
- Xavier initialization — variance 1/n, designed for symmetric activations
- Mini-batch gradient descent — shuffled batches, stable convergence
- One-hot encoding — converting integer labels to probability targets

---

## How To Run

**Prerequisites:**
```bash
pip install numpy matplotlib scikit-learn pandas jupyter
```

**Run the notebook:**
```bash
jupyter notebook training_demo.ipynb
```

**Or import the network directly:**
```python
from neural_network import NeuralNetwork
import numpy as np

nn = NeuralNetwork([784, 128, 64, 10])
output = nn.forward(X)          # forward pass
nn.backward(gradient, lr=0.01)  # backward pass
```

---

