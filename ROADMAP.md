# Number Recognition — Implementation Roadmap

> **Book reference:** *Neural Networks and Deep Learning* by Michael Nielsen  
> Each phase maps to a chapter or section of the book. Every term in **bold** is defined in [SKILLS.md](SKILLS.md).

Mark each step `[x]` when done. Steps are intentionally small — complete one before moving to the next.

## How to use this roadmap

- **Read the book section first.** Each phase header cites the chapter. Read it *before* writing any code — the equations and intuition will make the implementation obvious rather than mysterious.
- **One checkpoint at a time.** Each numbered step (e.g. `6.3`) is tiny on purpose. Do it, run it, verify the output matches the expected value shown, then mark it `[x]` and move on.
- **Print and verify shapes.** After every new function, print array shapes. Shape mismatches are silent in NumPy — they won't crash but will produce wrong results. Catching them early saves hours.
- **Use SKILLS.md as your dictionary.** Every bolded term links to a precise definition and the Nielsen variable name (`w`, `b`, `z`, `a`, `δ`, `η`, `λ`). Match these names in your code exactly.
- **Don't skip to later phases.** Later phases build directly on earlier ones. Phase 10 (backprop) requires Phase 7 (feedforward) to be correct.
- **See README.md** for the key math equations (BP1–BP4, cost functions, SGD update rule) laid out before you start coding.

---

## Phase 1 — Environment Setup

> Chapter: Prerequisite. Goal: get a working Python environment.

- [ ] **1.1** Create a **virtual environment**: `python -m venv .venv`
- [ ] **1.2** Activate it: `.venv\Scripts\activate` (Windows)
- [ ] **1.3** Install dependencies: `pip install numpy matplotlib`
- [ ] **1.4** Install TensorFlow for MNIST loading only: `pip install tensorflow` *(the **Network** class itself will use only NumPy)*
- [ ] **1.5** Verify: `python -c "import numpy as np; import matplotlib; print('OK')"` — confirm `OK`
- [ ] **1.6** Create folders: `data/`, `src/`
- [ ] **1.7** Create empty source files: `src/loader.py`, `src/network.py`, `src/train.py`, `src/evaluate.py`

---

## Phase 2 — Load and Inspect the Dataset

> Chapter 1, Section: "Using neural nets to recognize handwritten digits"  
> Terms exercised: **dataset**, **sample**, **image**, **pixel**, **grayscale**, **label**, **training set**, **validation set**, **test set**, **split**

- [ ] **2.1** In `src/loader.py`, write `load_raw()` that calls `tensorflow.keras.datasets.mnist.load_data()` and returns the raw NumPy arrays
- [ ] **2.2** Print the shape of `x_train` — confirm `(60000, 28, 28)`
- [ ] **2.3** Print the dtype of `x_train` — confirm `uint8`
- [ ] **2.4** Print min and max **pixel** values — confirm `0` and `255`
- [ ] **2.5** Print the shape of `y_train` — confirm `(60000,)` with integer **labels** 0–9
- [ ] **2.6** Display a single **image** from `x_train` using `matplotlib.pyplot.imshow(..., cmap='gray')` and print its **label**
- [ ] **2.7** Display a 5×5 grid of the first 25 **images** with their **labels** as titles

---

## Phase 3 — Preprocess into Nielsen's Format

> Chapter 1. Nielsen's **Network** class expects specific input shapes.  
> Terms exercised: **normalization**, **flattening**, **one-hot encoding**, **split**, **mnist_loader**

- [ ] **3.1** In `src/loader.py`, add `normalize(x)` that converts `uint8` pixels to `float32` in [0.0, 1.0] by dividing by `255.0`
- [ ] **3.2** Add `flatten(x)` that reshapes `(N, 28, 28)` → `(N, 784, 1)` producing a column vector per **sample** (Nielsen's format)
- [ ] **3.3** Add `one_hot(y)` that converts an integer **label** (0–9) into a `(10, 1)` column vector with a `1.0` at the correct index
- [ ] **3.4** Perform the 50k/10k/10k **split**: first 50 000 → **training set**, next 10 000 → **validation set**, last 10 000 → **test set**
- [ ] **3.5** Add a `load_data()` function that returns:
  - `training_data`: list of 50 000 `(x, y)` tuples — `x` is `(784,1)` float, `y` is `(10,1)` **one-hot encoding**
  - `validation_data`: list of 10 000 `(x, label)` tuples — `y` is the raw integer **label** (used for accuracy counting)
  - `test_data`: same format as `validation_data`
- [ ] **3.6** Print one item from each list to confirm the correct shapes
- [ ] **3.7** Verify: `len(training_data) == 50000`, `len(validation_data) == 10000`, `len(test_data) == 10000`

---

## Phase 4 — Perceptrons (Historical Foundation)

> Chapter 1, Section: "Perceptrons"  
> Terms exercised: **perceptron**, **weight**, **bias**, **weighted input (z)**, **activation**

- [ ] **4.1** Read Nielsen's perceptron section — understand: output = 0 if $\sum_j w_j x_j \leq \text{threshold}$, else 1
- [ ] **4.2** In a scratch file or notebook cell, write a `perceptron(inputs, weights, threshold)` function
- [ ] **4.3** Test it: with inputs `[1, 0]`, weights `[0.6, 0.4]`, threshold `0.5` — confirm output is `1`
- [ ] **4.4** Rewrite the perceptron using **bias** notation: $w \cdot x + b > 0$ (equivalent to the threshold form)
- [ ] **4.5** Try to train it on MNIST — observe: a single **perceptron** with a step function cannot learn a 10-class problem
- [ ] **4.6** Write one sentence explaining *why* a **perceptron** cannot be trained with **gradient descent** (hint: the step function has zero **gradient** almost everywhere)

---

## Phase 5 — Sigmoid Neurons

> Chapter 1, Section: "Sigmoid neurons"  
> Terms exercised: **sigmoid neuron**, **sigmoid function (σ)**, **weighted input (z)**, **activation (a)**, **learning slowdown** (contrast with perceptron)

- [ ] **5.1** In a scratch file, implement `sigmoid(z)` as $\frac{1}{1+e^{-z}}$ using `numpy`
- [ ] **5.2** Plot σ(z) for z in [−6, 6] — confirm it is smooth, bounded in (0, 1), and S-shaped
- [ ] **5.3** Implement `sigmoid_prime(z)` as $\sigma(z)(1 - \sigma(z))$
- [ ] **5.4** Plot σ'(z) — observe that it is near-zero at both ends (this is the root cause of **learning slowdown**)
- [ ] **5.5** Compute the **activation** for a single **sigmoid neuron**: given inputs `[0.5, 0.8]`, weights `[0.3, −0.2]`, bias `0.1`, compute `z` then `a = σ(z)`
- [ ] **5.6** Verify: small changes in **weights** and **bias** produce small, proportional changes in the **activation** (perturb each by 0.001 and observe the output change)

---

## Phase 6 — Network Architecture

> Chapter 1, Section: "The architecture of neural networks"  
> Terms exercised: **architecture**, **Network class**, **input layer**, **hidden layer**, **output layer**, **weight (w)**, **bias (b)**

- [ ] **6.1** In `src/network.py`, create a `Network` class with `__init__(self, sizes)` where `sizes` is a list like `[784, 30, 10]`
- [ ] **6.2** In `__init__`, initialize **biases**: a list of column vectors, one per non-input layer — `[np.random.randn(y, 1) for y in sizes[1:]]`
- [ ] **6.3** Initialize **weights**: a list of matrices — `[np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]`
- [ ] **6.4** Print `net.biases[0].shape` for `net = Network([784, 30, 10])` — confirm `(30, 1)`
- [ ] **6.5** Print `net.weights[0].shape` — confirm `(30, 784)` — each row is the **weights** going *into* one **neuron**
- [ ] **6.6** Print `net.weights[1].shape` — confirm `(10, 30)`
- [ ] **6.7** Count total parameters: `sum(b.size + w.size for b, w in zip(net.biases, net.weights))` — confirm ~23 860 for `[784, 30, 10]`

---

## Phase 7 — Feedforward (Forward Pass)

> Chapter 1, Section: "A simple network to classify handwritten digits"  
> Terms exercised: **forward pass**, **weighted input (z)**, **activation (a)**, **sigmoid function (σ)**, **prediction**

- [ ] **7.1** In `Network`, implement `feedforward(self, a)` that iterates over each `(b, w)` pair and computes `a = sigmoid(np.dot(w, a) + b)`
- [ ] **7.2** Load one **sample** `x` from `test_data` and run `net.feedforward(x)` on the untrained **Network**
- [ ] **7.3** Confirm the output shape is `(10, 1)` — ten **activations**, one per digit class
- [ ] **7.4** The **prediction** is `np.argmax(net.feedforward(x))` — print it alongside the true **label** (it will likely be wrong on the untrained network)
- [ ] **7.5** Add an `evaluate(self, test_data)` method that counts how many **samples** the **Network** predicts correctly — `sum(int(np.argmax(net.feedforward(x)) == y) for x, y in test_data)`
- [ ] **7.6** Run `net.evaluate(test_data)` on the untrained **Network** — confirm accuracy is ~10% (random chance for 10 classes)

---

## Phase 8 — Quadratic Cost

> Chapter 1. The first **cost function** we minimize.  
> Terms exercised: **cost function (C)**, **quadratic cost**, **gradient**

- [ ] **8.1** Read the formula: $C(w, b) = \frac{1}{2n}\sum_x \|y(x) - a^L(x)\|^2$. Identify each symbol using SKILLS.md definitions.
- [ ] **8.2** Implement `total_cost(self, data)` on the **Network** that computes the **quadratic cost** averaged over all **samples** in `data`
- [ ] **8.3** Compute `net.total_cost(training_data)` on the untrained **Network** — record the value
- [ ] **8.4** Understand *why* we need the $\frac{1}{2}$ factor: it makes $\partial C / \partial a^L$ simplify to $(a^L - y)$ cleanly

---

## Phase 9 — Stochastic Gradient Descent (SGD)

> Chapter 1, Section: "Learning with gradient descent"  
> Terms exercised: **stochastic gradient descent (SGD)**, **mini-batch**, **mini-batch size**, **epoch**, **learning rate (η)**

- [ ] **9.1** In `Network`, implement `SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None)`
- [ ] **9.2** Inside the `epochs` loop: shuffle `training_data` using `random.shuffle`
- [ ] **9.3** Slice the shuffled data into **mini-batches** of size `mini_batch_size`: `[training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]`
- [ ] **9.4** Call `self.update_mini_batch(mini_batch, eta)` for each **mini-batch** (stub the method for now — just `pass`)
- [ ] **9.5** If `test_data` is provided, print `self.evaluate(test_data)` at the end of each **epoch**
- [ ] **9.6** Confirm the loop structure runs without error using the stub (accuracy will still be ~10%)

---

## Phase 10 — Backpropagation: The Four Equations

> Chapter 2. The core algorithm of the book.  
> Terms exercised: **backward pass**, **delta (δ)**, **output error (δ^L)**, **Hadamard product (⊙)**, **σ' (sigma prime)**, **BP1**, **BP2**, **BP3**, **BP4**

- [ ] **10.1** Read Chapter 2 of Nielsen's book. Identify all four equations by name: **BP1**, **BP2**, **BP3**, **BP4**
- [ ] **10.2** In `Network`, implement `backprop(self, x, y)` that returns `(nabla_b, nabla_w)` — lists of gradient arrays with the same shapes as `self.biases` and `self.weights`
- [ ] **10.3** Run the **forward pass**: store all **weighted inputs** `zs` and all **activations** `activations` layer by layer
- [ ] **10.4** Apply **BP1** to compute `delta` for the **output layer**: `delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])`
  - `cost_derivative` returns $(a^L - y)$ for **quadratic cost**
- [ ] **10.5** Use **BP3** and **BP4** to store output-layer gradients: `nabla_b[-1] = delta`, `nabla_w[-1] = np.dot(delta, activations[-2].T)`
- [ ] **10.6** Loop backwards through layers 2 to L−1, applying **BP2** to propagate `delta`: `delta = np.dot(self.weights[l+1].T, delta) * sigmoid_prime(zs[l])`
- [ ] **10.7** Store each layer's gradients using **BP3** and **BP4**: `nabla_b[l] = delta`, `nabla_w[l] = np.dot(delta, activations[l-1].T)`
- [ ] **10.8** Return `(nabla_b, nabla_w)`

---

## Phase 11 — Update Mini-Batch

> Chapter 1/2. Connects backpropagation to weight updates.  
> Terms exercised: **gradient descent**, **learning rate (η)**, **mini-batch**, **weight (w)**, **bias (b)**

- [ ] **11.1** Implement `update_mini_batch(self, mini_batch, eta)` in `Network`
- [ ] **11.2** Initialize `nabla_b` and `nabla_w` as zero arrays with the same shapes as `self.biases` and `self.weights`
- [ ] **11.3** For each `(x, y)` **sample** in the **mini-batch**, call `self.backprop(x, y)` and accumulate: `nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]`
- [ ] **11.4** Update **weights**: `self.weights = [w - (eta/len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]`
- [ ] **11.5** Update **biases**: `self.biases = [b - (eta/len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]`

---

## Phase 12 — Train and See First Results

> Chapter 1. The payoff — a working digit recognizer.  
> Terms exercised: **epoch**, **training accuracy**, **test accuracy**, **convergence**

- [ ] **12.1** In `src/train.py`, load data using `loader.load_data()` and create `net = Network([784, 30, 10])`
- [ ] **12.2** Train: `net.SGD(training_data, epochs=30, mini_batch_size=10, eta=3.0, test_data=test_data)`
- [ ] **12.3** Watch the correct-count printed per **epoch** — it should climb from ~1 000 to ~9 500 out of 10 000
- [ ] **12.4** Confirm final **test accuracy** is approximately 95% (Nielsen's baseline result)
- [ ] **12.5** Try `Network([784, 100, 10])` with same settings — observe whether accuracy improves
- [ ] **12.6** Try increasing **learning rate (η)** to `10.0` — observe instability (accuracy degrades or collapses)
- [ ] **12.7** Try reducing η to `0.1` — observe slower **convergence**

---

## Phase 13 — Visualize and Evaluate

> Chapter 1/3.  
> Terms exercised: **cost curve**, **validation accuracy**, **training accuracy**, **overfitting**, **misclassified sample**, **confusion matrix**

- [ ] **13.1** Modify `SGD` to record **test accuracy** per **epoch** into a list; return it at the end
- [ ] **13.2** Plot **test accuracy** vs **epoch** using `matplotlib`
- [ ] **13.3** Add **training accuracy** tracking too — plot both curves on the same graph
- [ ] **13.4** Observe: does **training accuracy** pull ahead of **validation accuracy** in later epochs? Name this phenomenon using the SKILLS.md term
- [ ] **13.5** Find 10 **misclassified samples** from `test_data` — display each **image**, true **label**, and network's **prediction**
- [ ] **13.6** Compute and display the **confusion matrix** — identify which digit pairs are most often confused

---

## Phase 14 — Cross-Entropy Cost (Chapter 3)

> Chapter 3, Section: "The cross-entropy cost function"  
> Terms exercised: **cross-entropy cost**, **learning slowdown**, **BP1** (updated), **cost function (C)**

- [ ] **14.1** Re-read the **learning slowdown** section in Chapter 3 — understand why $\sigma'(z)$ causes the problem
- [ ] **14.2** Read the **cross-entropy cost** formula: $C = -\frac{1}{n}\sum_x [y \ln a + (1-y)\ln(1-a)]$
- [ ] **14.3** Show algebraically (or just read Nielsen's derivation) that with **cross-entropy cost**, **BP1** simplifies to $\delta^L = a^L - y$ (no $\sigma'$ term)
- [ ] **14.4** Add a `CrossEntropyCost` class with a `fn(a, y)` static method for the cost value and a `delta(z, a, y)` static method returning $a - y$
- [ ] **14.5** Refactor `Network` to accept an optional `cost=QuadraticCost` parameter — use `cost.delta(z, a, y)` in **BP1** instead of hardcoding
- [ ] **14.6** Train `Network([784, 30, 10], cost=CrossEntropyCost)` and compare convergence speed to **quadratic cost** in the first 10 **epochs**

---

## Phase 15 — Overfitting and L2 Regularization (Chapter 3)

> Chapter 3, Section: "Overfitting and regularization"  
> Terms exercised: **overfitting**, **L2 regularization**, **regularization parameter (λ)**, **weight decay**, **early stopping**

- [ ] **15.1** Train a large network `[784, 100, 10]` for 400 **epochs** — plot **training accuracy** and **validation accuracy** side by side
- [ ] **15.2** Identify the **epoch** where **validation accuracy** peaks and begins declining — this is the onset of **overfitting**
- [ ] **15.3** Understand **L2 regularization**: the updated **cost** is $C + \frac{\lambda}{2n}\sum_w w^2$
- [ ] **15.4** Understand **weight decay**: the weight update becomes $w \to (1 - \frac{\eta\lambda}{n})w - \frac{\eta}{m}\sum_x \frac{\partial C_x}{\partial w}$
- [ ] **15.5** Add an optional `lmbda=0.0` parameter to `SGD` and `update_mini_batch`
- [ ] **15.6** Modify the **weight** update in `update_mini_batch` to apply **weight decay**: multiply existing **weights** by $(1 - \frac{\eta \cdot \text{lmbda}}{n})$ before subtracting the gradient term
- [ ] **15.7** Retrain with `lmbda=0.1` — observe reduced gap between **training accuracy** and **validation accuracy**
- [ ] **15.8** Implement **early stopping**: after each **epoch**, check if **validation accuracy** has improved; if it hasn't improved in 10 consecutive **epochs**, stop training

---

## Phase 16 — Better Weight Initialization (Chapter 3)

> Chapter 3, Section: "Weight initialization"  
> Terms exercised: **weight initialization**, **weighted input (z)**, **sigma prime (σ')**, **learning slowdown**

- [ ] **16.1** Understand the problem: with $w \sim \mathcal{N}(0, 1)$ and 784 inputs, the **weighted input z** has standard deviation ~28 → **σ** saturates → **σ'(z) ≈ 0** → slow start
- [ ] **16.2** Implement `default_weight_initializer` in `Network` (the original $\mathcal{N}(0,1)$)
- [ ] **16.3** Implement `improved_weight_initializer` that uses $w \sim \mathcal{N}(0, 1/\sqrt{n_{in}})$ for **weights** and $b \sim \mathcal{N}(0, 1)$ for **biases**
- [ ] **16.4** Make `improved_weight_initializer` the default in `Network.__init__`
- [ ] **16.5** Compare training curves: old initializer vs improved initializer for the first 15 **epochs** — improved init should converge faster

---

## Phase 17 — Convolutional Neural Network (Chapter 6)

> Chapter 6. Final phase — replace dense layers with spatial feature learning.  
> Terms exercised: **convolutional layer**, **filter**, **feature map**, **pooling layer**, **architecture**, **ReLU**

- [ ] **17.1** Read Chapter 6 introduction — understand why spatial structure in **images** is wasted by full **flattening**
- [ ] **17.2** Understand a **convolutional layer**: a 5×5 **filter** slides across the 28×28 **image** with shared **weights**, producing a `(24, 24)` **feature map**
- [ ] **17.3** Understand a **pooling layer**: 2×2 max-pooling shrinks `(24, 24)` → `(12, 12)` per **feature map**
- [ ] **17.4** Install and use `theano` or (preferably) switch to a minimal Keras/TF implementation *only for the CNN phase* — note that building a CNN from scratch in NumPy is advanced and outside the book's scope for beginners
- [ ] **17.5** Define a CNN **architecture**: `Conv(20 filters, 5×5)` → `MaxPool(2×2)` → `Flatten` → `Dense(100, ReLU)` → `Dense(10, softmax)`
- [ ] **17.6** Train for 10–20 **epochs** — compare **test accuracy** against the fully-connected `[784, 100, 10]` result from Phase 12
- [ ] **17.7** Experiment with adding a second **convolutional layer** — observe whether **test accuracy** approaches 98–99%

---

## Completion Criteria

You have completed the roadmap when:
- [ ] The `Network` class in `src/network.py` implements `feedforward`, `SGD`, `update_mini_batch`, and `backprop` entirely in NumPy
- [ ] You can explain each of **BP1**, **BP2**, **BP3**, **BP4** in your own words
- [ ] **Test accuracy** with `[784, 30, 10]` + **SGD** + **quadratic cost** reaches ~95%
- [ ] **Test accuracy** improves with **cross-entropy cost** and **L2 regularization**
- [ ] You can explain the difference between **quadratic cost** and **cross-entropy cost** and when each is preferred
- [ ] The CNN achieves **test accuracy** above 98%
- [ ] You can define every term in [SKILLS.md](SKILLS.md) without looking it up


---

## Phase 1 — Environment Setup

> Goal: Get a working Python environment before writing any ML code.

- [ ] **1.1** Create a **virtual environment** in the project root: `python -m venv .venv`
- [ ] **1.2** Activate the **virtual environment** (`.venv\Scripts\activate` on Windows)
- [ ] **1.3** Install core dependencies: `pip install tensorflow numpy matplotlib`
- [ ] **1.4** Verify the install by running `python -c "import tensorflow as tf; import numpy as np; import matplotlib; print('OK')"` — confirm it prints `OK` with no errors
- [ ] **1.5** Create the folder structure: `data/`, `notebooks/`, `src/`
- [ ] **1.6** Create empty files: `src/data.py`, `src/model.py`, `src/train.py`, `src/evaluate.py`

---

## Phase 2 — Load and Inspect the Dataset

> Goal: Understand the shape and content of the **dataset** before writing any model code.  
> Terms exercised: **dataset**, **sample**, **image**, **label**, **pixel**, **grayscale**, **training set**, **test set**

- [ ] **2.1** In `src/data.py`, write a `load_data()` function that calls `tf.keras.datasets.mnist.load_data()` and returns `(x_train, y_train), (x_test, y_test)`
- [ ] **2.2** Print the shape of `x_train` — confirm it is `(60000, 28, 28)`
- [ ] **2.3** Print the dtype of `x_train` — confirm it is `uint8`
- [ ] **2.4** Print the shape of `y_train` — confirm it is `(60000,)`
- [ ] **2.5** Print the minimum and maximum **pixel** values in `x_train` — confirm they are `0` and `255`
- [ ] **2.6** Print the unique values in `y_train` — confirm they are integers 0–9 (**labels**)
- [ ] **2.7** In a notebook or script, display a single **image** from `x_train` using `matplotlib.pyplot.imshow(..., cmap='gray')` and print its **label**
- [ ] **2.8** Display a 5×5 grid of **images** with their **labels** as titles — pick the first 25 **samples**

---

## Phase 3 — Preprocess the Data

> Goal: Transform raw **pixel** values into a form the **model** can learn from.  
> Terms exercised: **normalization**, **flattening**, **one-hot encoding**, **split**, **validation set**

- [ ] **3.1** In `src/data.py`, add a `normalize(x)` function that converts `x` from `uint8` (0–255) to `float32` (0.0–1.0) by dividing by `255.0`
- [ ] **3.2** Call `normalize` on both `x_train` and `x_test`. Print min/max again to confirm they are now `0.0` and `1.0`
- [ ] **3.3** Add a `flatten(x)` function that reshapes `(N, 28, 28)` to `(N, 784)` — needed for the **dense layer** model in Phase 4
- [ ] **3.4** Verify the shape after **flattening**: `x_train_flat.shape` should be `(60000, 784)`
- [ ] **3.5** Carve out a **validation set** from `x_train`: use the last 10 000 **samples** as validation, the first 50 000 as the new **training set**
- [ ] **3.6** Add a `one_hot(y, num_classes=10)` function using `tf.keras.utils.to_categorical` that converts integer **labels** to **one-hot encoding** vectors
- [ ] **3.7** Apply `one_hot` to `y_train`, `y_val`, and `y_test`. Print one result to confirm shape `(10,)` with exactly one `1`

---

## Phase 4 — Build a Simple Dense Model

> Goal: Define the first **model** using only **dense layers** — no convolutions yet.  
> Terms exercised: **model**, **architecture**, **input layer**, **hidden layer**, **output layer**, **dense layer**, **neuron**, **weight**, **bias**, **activation function**, **ReLU**, **softmax**

- [ ] **4.1** In `src/model.py`, write a `build_dense_model()` function using `tf.keras.Sequential`
- [ ] **4.2** Add an **input layer** that accepts shape `(784,)`
- [ ] **4.3** Add one **hidden layer** with 128 **neurons** and **ReLU** **activation function**
- [ ] **4.4** Add a second **hidden layer** with 64 **neurons** and **ReLU** **activation function**
- [ ] **4.5** Add the **output layer** with 10 **neurons** (one per digit class) and **softmax** **activation function**
- [ ] **4.6** Call `model.summary()` and read the output — identify the number of **weights** and **biases** in each **layer**

---

## Phase 5 — Configure Training

> Goal: Attach a **loss function** and **optimizer** to the **model** so it can learn.  
> Terms exercised: **loss function**, **cross-entropy loss**, **optimizer**, **learning rate**, **gradient descent**

- [ ] **5.1** In `src/train.py`, import the **model** from `src/model.py` and the preprocessed data from `src/data.py`
- [ ] **5.2** Compile the **model** with `loss='categorical_crossentropy'` (**cross-entropy loss**)
- [ ] **5.3** Set the **optimizer** to Adam with a **learning rate** of `0.001`
- [ ] **5.4** Set `metrics=['accuracy']` so Keras tracks **training accuracy** each **epoch**

---

## Phase 6 — Train the Model

> Goal: Run the training loop and observe **loss** and **accuracy** changing across **epochs**.  
> Terms exercised: **epoch**, **batch**, **batch size**, **training step**, **forward pass**, **backward pass**, **gradient**, **convergence**

- [ ] **6.1** Call `model.fit()` with: **training set**, **batch size** of `64`, `5` **epochs**, and the **validation set** passed as `validation_data`
- [ ] **6.2** Read the printed output — identify the columns: `loss`, `accuracy`, `val_loss`, `val_accuracy`
- [ ] **6.3** Increase **epochs** to `15` and retrain — observe whether **loss** keeps decreasing (**convergence**)
- [ ] **6.4** Save the trained **model** weights as a **checkpoint** using `model.save('data/dense_model.keras')`

---

## Phase 7 — Evaluate the Model

> Goal: Measure how well the **model** generalizes using metrics and visualizations.  
> Terms exercised: **test accuracy**, **validation accuracy**, **loss curve**, **confusion matrix**, **misclassified sample**, **overfitting**, **underfitting**

- [ ] **7.1** In `src/evaluate.py`, load the saved **checkpoint**: `tf.keras.models.load_model('data/dense_model.keras')`
- [ ] **7.2** Call `model.evaluate(x_test_flat, y_test_onehot)` and print the **test accuracy**
- [ ] **7.3** Compare **training accuracy** vs **validation accuracy** vs **test accuracy** — does a gap suggest **overfitting**?
- [ ] **7.4** Use the training `history` object to plot the **loss curve**: training **loss** and validation **loss** on the same graph over **epochs**
- [ ] **7.5** Plot the accuracy curve the same way
- [ ] **7.6** Get **predictions** on `x_test` using `model.predict()`
- [ ] **7.7** Find at least 10 **misclassified samples** — display each **image**, its true **label**, and the **model**'s **prediction**
- [ ] **7.8** Compute the **confusion matrix** using `sklearn.metrics.confusion_matrix` and display it as a heatmap

---

## Phase 8 — Run Inference

> Goal: Use the trained **model** to make a **prediction** on a single **image** (simulating real use).  
> Terms exercised: **inference**, **prediction**, **pipeline**, **normalization**

- [ ] **8.1** Write a `predict_digit(image_array, model)` function in `src/evaluate.py` that accepts a raw `(28, 28)` uint8 **image**, applies **normalization** and **flattening**, runs a **forward pass**, and returns the predicted **label**
- [ ] **8.2** Call `predict_digit` on 5 **samples** from `x_test` and print the results alongside the true **labels**
- [ ] **8.3** Display each **image** with the predicted **label** as the title

---

## Phase 9 — Build a Convolutional Model (CNN)

> Goal: Replace the **dense layer** **model** with a **convolutional layer** model and observe the improvement.  
> Terms exercised: **convolutional layer**, **filter**, **feature map**, **pooling layer**, **architecture**

- [ ] **9.1** In `src/data.py`, add a `reshape_for_cnn(x)` function that reshapes `(N, 28, 28)` to `(N, 28, 28, 1)` — adds a channel dimension required by **convolutional layers**
- [ ] **9.2** Apply `reshape_for_cnn` to `x_train`, `x_val`, and `x_test`
- [ ] **9.3** In `src/model.py`, write a `build_cnn_model()` function
- [ ] **9.4** Add a **convolutional layer** with 32 **filters**, kernel size `3×3`, and **ReLU** **activation function**
- [ ] **9.5** Add a **pooling layer** (MaxPooling) with pool size `2×2`
- [ ] **9.6** Add a second **convolutional layer** with 64 **filters**, kernel size `3×3`, and **ReLU** **activation function**
- [ ] **9.7** Add another **pooling layer** with pool size `2×2`
- [ ] **9.8** Add a `Flatten` **layer** to transition from **feature maps** to a 1D vector
- [ ] **9.9** Add one **dense layer** with 64 **neurons** and **ReLU**, then the **output layer** with **softmax**
- [ ] **9.10** Call `model.summary()` — compare total **weight** count to the dense **model** from Phase 4

---

## Phase 10 — Train and Compare the CNN

> Goal: Train the CNN and compare its **test accuracy** and **loss curve** against the dense **model**.  
> Terms exercised: **epoch**, **batch size**, **checkpoint**, **test accuracy**, **loss curve**, **overfitting**

- [ ] **10.1** Compile the CNN with the same **loss function** and **optimizer** settings as Phase 5
- [ ] **10.2** Train for `10` **epochs** with **batch size** `64` and the **validation set**
- [ ] **10.3** Save the CNN **checkpoint** to `data/cnn_model.keras`
- [ ] **10.4** Evaluate on the **test set** — record **test accuracy**
- [ ] **10.5** Plot the **loss curves** for both the dense **model** and the CNN side-by-side
- [ ] **10.6** Write a brief comparison (2–3 sentences) in a code comment or notebook cell: which **model** has higher **test accuracy**? Does the CNN show signs of **overfitting**?

---

## Completion Criteria

You have completed the roadmap when:
- [ ] Both models (dense and CNN) are trained and saved as **checkpoints**
- [ ] The **test accuracy** of the CNN exceeds 98%
- [ ] You can call `predict_digit()` on any raw **image** and get the correct **label**
- [ ] You can explain the role of each term in [SKILLS.md](SKILLS.md) in your own words
