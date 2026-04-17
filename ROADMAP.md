# Number Recognition — Implementation Roadmap

Each checkpoint is a single, self-contained task. Complete them in order. Every term in **bold** is defined in [SKILLS.md](SKILLS.md).

Mark each step with `[x]` when done.

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
