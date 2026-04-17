---
applyTo: "**"
---

# Number Recognition — Domain Skills & Vocabulary

> **Book reference:** This project follows *Neural Networks and Deep Learning* by Michael Nielsen (available free at [neuralnetworksanddeeplearning.com](http://neuralnetworksanddeeplearning.com)). All terminology, notation, and implementation choices mirror the book exactly.

> **LLM Instruction:** When assisting with this project, always use the exact terms and notation defined in this file. Anchor every explanation in the bolded vocabulary below before expanding. Cross-reference related terms explicitly (e.g., "a **sigmoid neuron** applies **σ** to its **weighted input z** to produce its **activation**"). Use Nielsen's variable names (`w`, `b`, `z`, `a`, `δ`, `η`, `λ`, `C`) in all code and explanations. Do not use synonyms for defined terms.

---

## 1. Data Terms

| Term | Definition |
|---|---|
| **pixel** | A single numeric value in an image. In MNIST, each pixel is an integer 0–255 representing brightness (0 = black, 255 = white). |
| **grayscale** | An image format where each pixel has exactly one value representing intensity, rather than separate R, G, B channels. |
| **image** | A 2D grid of **pixels** with shape `(28, 28)` in MNIST. Flattened into a 784-dimensional vector before being fed into the network. |
| **label** | The correct answer for a given sample — an integer 0–9 identifying which digit the **image** shows. |
| **sample** | One (image, label) pair in the **dataset**. |
| **dataset** | The full collection of **samples**. MNIST contains 70 000 samples. |
| **training set** | The 50 000 **samples** the network sees and learns from. |
| **validation set** | 10 000 held-out **samples** used to monitor performance and tune hyperparameters during training. Does not influence **weights** directly. |
| **test set** | 10 000 **samples** used only for final evaluation after training is complete. |
| **split** | Dividing the **dataset** into **training set**, **validation set**, and **test set**. Nielsen uses a 50k/10k/10k split. |
| **normalization** | Rescaling **pixel** values from 0–255 to 0.0–1.0 so inputs are in a consistent range for the **sigmoid function**. |
| **flattening** | Reshaping a 2D **image** `(28, 28)` into a 784-dimensional column vector — the input format expected by the **Network** class. |
| **one-hot encoding** | Representing an integer **label** (e.g., `3`) as a 10-dimensional vector with a `1` at index 3 and `0` everywhere else. Used as the target output `y` during training. |
| **mini-batch** | A small random subset of the **training set** (Nielsen uses size 10) processed together before updating **weights**. Enables **stochastic gradient descent**. |
| **mini-batch size** | The number of **samples** in one **mini-batch**. |

---

## 2. Neuron & Network Terms

| Term | Definition |
|---|---|
| **perceptron** | The historical precursor to the **sigmoid neuron**. Outputs 0 or 1 based on a weighted sum threshold. Not differentiable, so cannot be trained with **gradient descent**. |
| **sigmoid neuron** | A neuron that uses the **sigmoid function** σ as its **activation function**. Small changes in **weights** and **biases** produce small, predictable changes in output — enabling gradient-based learning. |
| **sigmoid function (σ)** | The activation function $\sigma(z) = \frac{1}{1+e^{-z}}$. Outputs a value in (0, 1). Used in all hidden and output neurons in Chapters 1–3. |
| **weighted input (z)** | $z = w \cdot a + b$ — the raw scalar passed into **σ** for a single neuron, or the vector $z^l = w^l a^{l-1} + b^l$ for a whole layer. |
| **activation (a)** | The output of a neuron after applying the **activation function**: $a = \sigma(z)$. For layer $l$: $a^l = \sigma(z^l)$. |
| **weight (w)** | A learnable parameter connecting one **activation** to another. Stored as matrices $w^l$ where $w^l_{jk}$ connects neuron $k$ in layer $l-1$ to neuron $j$ in layer $l$. |
| **bias (b)** | A learnable scalar added to the **weighted input** of each neuron. Stored as column vectors $b^l$. |
| **layer** | A group of neurons that operate in parallel. Indexed $l = 1, 2, \ldots, L$. |
| **input layer** | Layer 1 — receives the flattened 784-dimensional **image** vector. Its "activations" are just the input values. |
| **hidden layer** | Any layer between the **input layer** and the **output layer**. Nielsen uses one hidden layer of 30 neurons initially. |
| **output layer** | Layer $L$ — produces a 10-dimensional **activation** vector. The predicted **label** is `argmax` of this vector. |
| **architecture** | The list of layer sizes passed to the **Network** class, e.g. `[784, 30, 10]`. Determines the dimensions of all **weight** matrices and **bias** vectors. |
| **Network class** | The Python class (in `src/network.py`) that holds all **weights** and **biases** and implements `feedforward`, `SGD`, `update_mini_batch`, and `backprop`. |
| **dense layer** | A layer where every neuron is connected to every neuron in the previous layer. All layers in Chapters 1–3 are dense. Also called a fully-connected layer. |
| **convolutional layer** | Introduced in Chapter 6. A layer that slides a small **filter** across the **image**, learning spatial features rather than using full connections. |
| **filter** | A small matrix of **weights** in a **convolutional layer** that detects a specific spatial pattern (e.g., an edge). |
| **feature map** | The output grid produced by convolving one **filter** across the input. |
| **pooling layer** | Introduced in Chapter 6. Reduces the spatial size of a **feature map** by taking the max or average over small regions. |
| **softmax** | An alternative output **activation function** (introduced in Ch. 3 exercises) that converts raw scores into a proper probability distribution. |
| **ReLU** | Rectified Linear Unit — `max(0, x)`. Mentioned as an alternative activation in Chapter 3; used in CNN experiments in Chapter 6. |

---

## 3. Cost & Training Terms

| Term | Definition |
|---|---|
| **cost function (C)** | A scalar measuring how wrong the network's output is compared to the true **label**. Nielsen uses "cost" — not "loss". The goal of training is to minimize $C$. |
| **quadratic cost** | The first cost function introduced: $C = \frac{1}{2n}\sum_x \|y(x) - a^L(x)\|^2$. Simple but suffers from **learning slowdown** when the output neuron is saturated. |
| **cross-entropy cost** | Introduced in Chapter 3: $C = -\frac{1}{n}\sum_x [y \ln a^L + (1-y)\ln(1-a^L)]$. Eliminates the **learning slowdown** by canceling the $\sigma'(z)$ term in the **gradient**. |
| **learning slowdown** | The phenomenon where a saturated **sigmoid neuron** (σ near 0 or 1) has near-zero $\sigma'(z)$, causing its **weight** and **bias** gradients to be tiny — making learning very slow. Cured by switching to **cross-entropy cost**. |
| **forward pass** | Computing activations layer by layer: $a^l = \sigma(w^l a^{l-1} + b^l)$ for $l = 1$ to $L$. Produces the network's **prediction**. |
| **prediction** | The digit class corresponding to `argmax(a^L)` — the output neuron with the highest **activation**. |
| **gradient** | The vector of partial derivatives $\partial C / \partial w$ and $\partial C / \partial b$ indicating how much each **weight** and **bias** contributes to **C**. |
| **gradient descent** | Updating weights and biases opposite to the **gradient**: $w \to w - \eta \frac{\partial C}{\partial w}$. |
| **stochastic gradient descent (SGD)** | Gradient descent applied per **mini-batch** rather than the full **training set**. Much faster in practice. Nielsen's `SGD` method shuffles the training data, creates **mini-batches**, and calls `update_mini_batch` for each one. |
| **learning rate (η)** | The scalar controlling step size in **gradient descent**: $w \to w - \eta \frac{\partial C}{\partial w}$. Nielsen uses `η = 3.0` in Chapter 1. |
| **epoch** | One full pass through the entire **training set**. |
| **backward pass** | Running **backpropagation** to compute the **gradient** of **C** with respect to every **weight** and **bias** in the network. |
| **convergence** | The state where **C** stops meaningfully decreasing across **epochs**. |
| **overfitting** | When the network memorizes the **training set** — high **training accuracy** but declining **validation accuracy**. Diagnosed by plotting both curves across epochs. |
| **underfitting** | When the network is too small or training too short — high **cost** on both **training set** and **validation set**. |

---

## 4. Backpropagation Terms

| Term | Definition |
|---|---|
| **delta (δ)** | The "error" in a neuron — defined as $\delta^l_j = \partial C / \partial z^l_j$. The central quantity in backpropagation. |
| **output error (δ^L)** | The **delta** for the output layer. Computed by **BP1**. |
| **Hadamard product (⊙)** | Element-wise multiplication of two vectors of the same shape: $(u \odot v)_j = u_j v_j$. Used in **BP1** and **BP2**. |
| **BP1** | $\delta^L = \nabla_a C \odot \sigma'(z^L)$ — computes the **output error**. For **quadratic cost**: $\delta^L = (a^L - y) \odot \sigma'(z^L)$. For **cross-entropy cost**: $\delta^L = a^L - y$ (the $\sigma'$ term cancels). |
| **BP2** | $\delta^l = ((w^{l+1})^T \delta^{l+1}) \odot \sigma'(z^l)$ — backpropagates the **error** from layer $l+1$ to layer $l$. |
| **BP3** | $\frac{\partial C}{\partial b^l_j} = \delta^l_j$ — the **gradient** of **C** with respect to a **bias** equals the **delta** of that neuron. |
| **BP4** | $\frac{\partial C}{\partial w^l_{jk}} = a^{l-1}_k \delta^l_j$ — the **gradient** of **C** with respect to a **weight** equals the **activation** feeding into it times the **delta** of the neuron it feeds. |
| **σ' (sigma prime)** | The derivative of **σ**: $\sigma'(z) = \sigma(z)(1 - \sigma(z))$. Appears in **BP1** and **BP2**. |

---

## 5. Improvement Terms (Chapter 3)

| Term | Definition |
|---|---|
| **regularization** | Techniques that reduce **overfitting** by penalizing complexity or adding noise. |
| **L2 regularization** | Adds $\frac{\lambda}{2n}\sum_w w^2$ to the **cost function**, discouraging large **weights**. Controlled by the **regularization parameter λ**. |
| **regularization parameter (λ)** | The scalar controlling the strength of **L2 regularization**. Larger λ → stronger penalty on **weights**. Nielsen uses λ = 0.1 as a starting point. |
| **weight decay** | The effect of **L2 regularization** on the weight update rule: $w \to \left(1 - \frac{\eta \lambda}{n}\right)w - \frac{\eta}{m}\sum_x \frac{\partial C_x}{\partial w}$. The factor $(1 - \eta\lambda/n)$ shrinks weights each step. |
| **early stopping** | Halting training when **validation accuracy** stops improving, to prevent **overfitting**. |
| **weight initialization** | How **weights** are set before training. Naive: $w \sim \mathcal{N}(0, 1)$ — causes saturation in large-fan-in neurons. Better (Ch. 3): $w \sim \mathcal{N}(0, 1/\sqrt{n_{in}})$ — keeps **weighted inputs z** from saturating **σ** at the start. |
| **dropout** | Briefly introduced in Ch. 3. Randomly deactivating neurons during training to reduce **overfitting**. |

---

## 6. Evaluation Terms

| Term | Definition |
|---|---|
| **accuracy** | The fraction of **samples** where `argmax(a^L)` matches the true **label**. `correct / total`. |
| **training accuracy** | **Accuracy** on the **training set**, reported after each **epoch**. |
| **validation accuracy** | **Accuracy** on the **validation set**, used to detect **overfitting** and guide hyperparameter choices. |
| **test accuracy** | **Accuracy** on the **test set**, reported only after training is finalized. Nielsen achieves ~95% with `[784, 30, 10]` + SGD, and ~98%+ with CNNs. |
| **cost curve** | A plot of **C** versus **epoch** for training and validation sets. Used to diagnose **overfitting** or **underfitting**. |
| **confusion matrix** | A 10×10 grid showing how often each true digit was predicted as each digit. Reveals systematic errors. |
| **misclassified sample** | A **sample** where `argmax(a^L)` does not match the true **label**. |

---

## 7. Code / Implementation Terms

| Term | Definition |
|---|---|
| **virtual environment** | An isolated Python installation for this project. |
| **checkpoint** | A saved `.pkl` or `.npy` snapshot of the network's **weights** and **biases** at a point in training. |
| **inference** | Using a trained **Network** to compute `feedforward(x)` on new data — without updating **weights**. |
| **pipeline** | The end-to-end sequence: load data → preprocess → train → evaluate → run **inference**. |
| **mnist_loader** | The data-loading module (`src/loader.py`) that returns `training_data`, `validation_data`, and `test_data` in the format the **Network** class expects: lists of `(x, y)` tuples where `x` is a 784×1 column vector and `y` is a 10×1 **one-hot encoding** vector. |

---

## LLM Guidance Summary

- Always introduce a concept with its **bolded term** before explaining it.
- Use Nielsen's notation consistently: **weighted input** `z`, **activation** `a`, **delta** `δ`, **learning rate** `η`, **regularization parameter** `λ`, **cost** `C`.
- When explaining **backpropagation**, always name and number the equation being used (**BP1** through **BP4**).
- Never say "loss" — always say **cost** or **cost function** (Nielsen's term).
- Never say "target" or "ground truth" — always say **label**.
- When writing a step from ROADMAP.md, explicitly name which **terms** from this file the step exercises.
- When two terms relate, show the relationship (e.g., "**BP2** backpropagates the **delta** using the **Hadamard product** with **σ'**").
- Match variable names in all code to Nielsen's notation: `w`, `b`, `z`, `a`, `delta`, `nabla_w`, `nabla_b`, `eta`, `lmbda`.
- Reference the book chapter when introducing a concept (e.g., "as Nielsen introduces in Chapter 2…").

