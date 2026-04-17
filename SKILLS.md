---
applyTo: "**"
---

# Number Recognition — Domain Skills & Vocabulary

> **LLM Instruction:** When assisting with this project, always use the exact terms defined in this file when explaining or discussing concepts. If a user asks about something covered here, anchor your explanation in the vocabulary below before expanding further. Cross-reference terms when they relate to each other (e.g., "a **neuron** applies an **activation function** to a weighted sum of its **inputs**").

---

## 1. Data Terms

| Term | Definition |
|---|---|
| **pixel** | A single numeric value in an image. In MNIST, each pixel is an integer 0–255 representing brightness (0 = black, 255 = white). |
| **grayscale** | An image format where each pixel has exactly one value representing intensity, rather than separate R, G, B channels. |
| **image** | A 2D grid of **pixels** with shape `(height, width)`. MNIST images are `(28, 28)`. |
| **label** | The correct answer for a given sample — an integer 0–9 identifying which digit the **image** shows. |
| **sample** | One (image, label) pair in the **dataset**. |
| **dataset** | The full collection of **samples** used in a project. MNIST contains 70 000 samples. |
| **training set** | The portion of the **dataset** the model sees and learns from (typically 60 000 samples in MNIST). |
| **validation set** | A held-out portion used to monitor performance during training without influencing **weights**. |
| **test set** | A completely separate portion used only for final evaluation after training is done (10 000 samples in MNIST). |
| **split** | The act of dividing the **dataset** into **training set**, **validation set**, and **test set**. |
| **normalization** | Rescaling **pixel** values from 0–255 to 0.0–1.0 so that no single feature dominates learning. |
| **flattening** | Reshaping a 2D **image** `(28, 28)` into a 1D array `(784,)` so it can be fed into a **dense layer**. |
| **one-hot encoding** | Converting an integer **label** (e.g., `3`) into a vector with a `1` at the target index and `0` everywhere else (e.g., `[0,0,0,1,0,0,0,0,0,0]`). |
| **batch** | A small subset of the **training set** processed together before updating **weights**. Typical sizes: 32, 64, 128. |
| **batch size** | The number of **samples** in one **batch**. |

---

## 2. Model Terms

| Term | Definition |
|---|---|
| **model** | A mathematical function with learnable **weights** that maps an **image** to a predicted **label**. |
| **weight** | A learnable numeric parameter inside a **neuron**. Updated during training to reduce **loss**. |
| **bias** | An additional learnable scalar added to a **neuron**'s output, independent of its **inputs**. |
| **neuron** | A computational unit that takes a vector of **inputs**, multiplies each by a **weight**, adds a **bias**, then passes the result through an **activation function**. |
| **layer** | A group of **neurons** that operate in parallel on the same **inputs**. |
| **dense layer** | A **layer** in which every **neuron** is connected to every input value. Also called a fully-connected layer. |
| **convolutional layer** | A **layer** that slides a small **filter** (kernel) across the **image**, learning spatial patterns like edges and curves. |
| **filter** | A small 2D matrix of **weights** in a **convolutional layer** that detects a specific spatial feature. |
| **feature map** | The output produced by applying a **filter** to an **image** or previous **feature map**. |
| **pooling layer** | A **layer** that reduces the spatial size of a **feature map** by taking the max or average over small regions. |
| **activation function** | A non-linear function applied after a **neuron**'s weighted sum to introduce non-linearity. Common choices: ReLU, softmax. |
| **ReLU** | Rectified Linear Unit — an **activation function** defined as `max(0, x)`. Used in hidden **layers**. |
| **softmax** | An **activation function** for the output **layer** that converts raw scores into a probability distribution over all 10 digit classes. |
| **input layer** | The first **layer** that receives the raw **image** data (flattened **pixels** or 2D grid). |
| **hidden layer** | Any **layer** between the **input layer** and the **output layer**. Learns intermediate representations. |
| **output layer** | The final **layer** that produces a 10-value probability vector, one per digit class. |
| **architecture** | The overall design of a **model**: number of **layers**, type of each **layer**, and their sizes. |

---

## 3. Training Terms

| Term | Definition |
|---|---|
| **forward pass** | Feeding an **image** through every **layer** from input to output to produce a **prediction**. |
| **prediction** | The **model**'s output — a probability distribution over 10 digit classes. The predicted **label** is the class with the highest probability. |
| **loss** | A scalar number measuring how wrong the **model**'s **prediction** is compared to the true **label**. Lower is better. |
| **loss function** | The formula used to calculate **loss**. For digit classification: **cross-entropy loss**. |
| **cross-entropy loss** | A **loss function** that penalizes confident wrong **predictions** heavily. Defined as `-sum(label * log(prediction))`. |
| **backward pass** | Computing the **gradient** of the **loss** with respect to every **weight** in the **model** using the chain rule. |
| **gradient** | The direction and magnitude in which a **weight** should change to reduce **loss**. |
| **gradient descent** | An optimization strategy: adjust each **weight** by a small step opposite to its **gradient**. |
| **learning rate** | A scalar (e.g., `0.001`) that controls how large each **gradient descent** step is. Too high → unstable training. Too low → slow convergence. |
| **optimizer** | The algorithm that applies **gradient descent** to update **weights**. Common choices: SGD, Adam. |
| **epoch** | One full pass through the entire **training set**. Training typically runs for 5–50 **epochs**. |
| **training step** | Processing one **batch**: **forward pass** → compute **loss** → **backward pass** → update **weights**. |
| **convergence** | The state where **loss** stops meaningfully decreasing, indicating the **model** has learned as much as it can. |
| **overfitting** | When the **model** memorizes the **training set** but performs poorly on the **validation set** or **test set**. |
| **underfitting** | When the **model** is too simple to capture the patterns in the **training set**, resulting in high **loss** on both sets. |

---

## 4. Evaluation Terms

| Term | Definition |
|---|---|
| **accuracy** | The fraction of **samples** where the predicted **label** matches the true **label**. `correct / total`. |
| **training accuracy** | **Accuracy** measured on the **training set**. |
| **validation accuracy** | **Accuracy** measured on the **validation set** after each **epoch**. Used to detect **overfitting**. |
| **test accuracy** | **Accuracy** measured on the **test set** after training is fully complete. |
| **confusion matrix** | A 10×10 grid showing how often each true digit class was predicted as each digit class. Reveals systematic errors. |
| **misclassified sample** | A **sample** where the **model**'s **prediction** does not match the true **label**. |
| **loss curve** | A plot of **loss** versus **epoch** for both the **training set** and **validation set**. Used to diagnose **overfitting** or **underfitting**. |

---

## 5. Code / Implementation Terms

| Term | Definition |
|---|---|
| **virtual environment** | An isolated Python installation for this project so dependencies don't conflict with other projects. |
| **checkpoint** | A saved snapshot of the **model**'s **weights** at a given point in training, allowing training to resume or the best **model** to be recovered. |
| **inference** | Using a trained **model** to make a **prediction** on new, unseen data — without updating **weights**. |
| **pipeline** | The end-to-end sequence: load data → preprocess → train **model** → evaluate → run **inference**. |

---

## LLM Guidance Summary

- Always introduce a new concept using its **bolded term** before explaining it.
- When two terms relate, show the relationship explicitly (e.g., "the **optimizer** uses the **gradient** from the **backward pass** to update **weights**").
- Match the vocabulary in code comments and variable names to the terms above (e.g., use `learning_rate`, `num_epochs`, `batch_size`).
- When writing a step from ROADMAP.md, state which **terms** from this file are exercised in that step.
- Do not use synonyms for defined terms (e.g., prefer **loss** over "error" or "cost"; prefer **label** over "target" or "ground truth").
