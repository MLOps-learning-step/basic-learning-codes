# Machine Learning from Scratch

A hands-on journey learning Machine Learning fundamentals by implementing classic algorithms from scratch in Python.

This repository documents my learning process, building each algorithm step by step — starting from the simplest neuron model (Perceptron) and progressing toward more advanced techniques. The goal is to deeply understand **how and why** these algorithms work, not just how to call them from a library.

## Learning Path

### 01 — Perceptron

The starting point: the simplest neural network model. A single-layer binary classifier that learns by adjusting weights based on misclassification errors.

**What you'll find:**

- `perceptron.py` — Full Perceptron class with `fit` and `predict` methods.
- `perceptron_example.py` — Step-by-step walkthrough showing how `zip()` works during training, how weights update, and how predictions are made.

**Key concepts:** weight initialization, dot product, unit step function, learning rate (`eta`), epochs.

---

### 02 — Adaline (Adaptive Linear Neuron)

An improvement over the Perceptron: instead of updating weights based on a threshold, Adaline uses a **continuous cost function** (Sum of Squared Errors) and minimizes it using gradient descent.

**What you'll find:**

- `adaline_gd.py` — Adaline with **Batch Gradient Descent**. Updates weights using the entire dataset per epoch.
- `adaline_sgd.py` — Adaline with **Stochastic Gradient Descent (SGD)**. Updates weights one sample at a time, which converges faster and supports online learning.

**Key concepts:** cost function, gradient descent, batch vs. stochastic updates, feature standardization, convergence, learning rate tuning.

---

### 03 — Iris Classification (Putting It All Together)

A practical application that trains and compares all three models (Perceptron, Adaline GD, Adaline SGD) on the classic **Iris dataset**.

**What you'll find:**

- `iris_classification.py` — Loads the Iris dataset, standardizes features, trains each model, and visualizes decision boundaries and cost convergence.

**Key concepts:** real-world dataset, data preprocessing, decision boundary visualization, model comparison.

---

### 04 — Logistic Regression (Regularization)

Exploring how the regularization parameter **C** affects model weights in Logistic Regression using scikit-learn.

**What you'll find:**

- `regularization_parameter.py` — Trains multiple Logistic Regression models with different C values and plots how weights change.

**Key concepts:** regularization, overfitting vs. underfitting, inverse regularization strength (C), scikit-learn.

---

## Tech Stack

- **Python 3**
- **NumPy** — Linear algebra and array operations
- **Pandas** — Data loading and manipulation
- **Matplotlib** — Visualization
- **scikit-learn** — Used starting from Chapter 04 for comparison

## How to Use

```bash
# Clone the repo
git clone https://github.com/MLOps-learning-step/basic-learning-codes.git
cd basic-learning-codes

# Install dependencies
pip install numpy pandas matplotlib scikit-learn

# Run any script
python 03_iris_classification/iris_classification.py
```

## Roadmap

- [x] Perceptron
- [x] Adaline (Gradient Descent)
- [x] Adaline (Stochastic Gradient Descent)
- [x] Iris dataset classification
- [x] Logistic Regression regularization
- [ ] Decision Trees
- [ ] Support Vector Machines (SVM)
- [ ] K-Nearest Neighbors (KNN)
- [ ] Neural Networks (multi-layer)
- [ ] Model evaluation & cross-validation

## About

Created by **Jose David Angarita Pertuz** as part of a self-directed Machine Learning study. Inspired by *"Python Machine Learning"* by Sebastian Raschka.

If you find this useful for your own learning, feel free to fork it and follow along!
