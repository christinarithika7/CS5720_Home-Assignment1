"""
Task 3: Train a Model with Different Optimizers
CS5720 - Neural Networks and Deep Learning - Home Assignment 1
Student Name: Vinay Kumar Christina Rithika Nethakani
Student ID: 700778781

Steps:
1. Load the MNIST dataset.
2. Train two models: one with Adam and another with SGD.
3. Compare training and validation accuracy trends.

CODE EXPLANATION
-----------------
1. MNIST is loaded as 60,000 training images and 10,000 test images of
   handwritten digits (28x28 grayscale). Dividing pixel values by 255
   scales them from [0,255] to [0,1], which helps the network train
   faster and more stably.

2. build_model() creates a simple feed-forward network:
   - Flatten turns each 28x28 image into a 784-length vector.
   - Dense(128, relu) is a hidden layer that learns feature combinations.
   - Dropout(0.2) randomly zeroes 20% of neurons during training to
     reduce overfitting.
   - Dense(10, softmax) outputs a probability distribution over the 10
     digit classes.

3. Two identical models are trained with different optimizers:
   - Adam adapts the learning rate for each parameter individually using
     running estimates of the gradient's mean and variance, so it
     typically converges faster with less manual tuning.
   - SGD (plain stochastic gradient descent) updates every parameter by
     the same fixed learning rate times the gradient, which is simpler
     but usually converges more slowly and can be more sensitive to the
     learning rate choice.

4. history.history['accuracy'] and ['val_accuracy'] store the accuracy
   after every epoch on the training set and validation (test) set,
   respectively. Plotting all four curves together lets us directly
   compare how quickly and how well each optimizer learns, and whether
   either one starts overfitting (train accuracy rising while validation
   accuracy stalls).
"""

import tensorflow as tf
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Step 1: Load and preprocess the MNIST dataset
# ---------------------------------------------------------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values from [0, 255] to [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


def build_model():
    """Builds a simple fully-connected neural network for MNIST digit classification."""
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    return model


# ---------------------------------------------------------------------------
# Step 2: Train one model with Adam and another with SGD
# ---------------------------------------------------------------------------
EPOCHS = 10
BATCH_SIZE = 32

# --- Model 1: Adam optimizer ---
model_adam = build_model()
model_adam.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
history_adam = model_adam.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=2,
)

# --- Model 2: SGD optimizer ---
model_sgd = build_model()
model_sgd.compile(
    optimizer="sgd",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
history_sgd = model_sgd.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=2,
)

# ---------------------------------------------------------------------------
# Step 3: Compare training and validation accuracy trends
# ---------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(history_adam.history["accuracy"], label="Adam - Train Accuracy")
plt.plot(history_adam.history["val_accuracy"], label="Adam - Val Accuracy")
plt.plot(history_sgd.history["accuracy"], label="SGD - Train Accuracy")
plt.plot(history_sgd.history["val_accuracy"], label="SGD - Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Adam vs. SGD: Training & Validation Accuracy on MNIST")
plt.legend()
plt.tight_layout()
plt.savefig("adam_vs_sgd_accuracy.png")
plt.show()

print("Final Adam val accuracy:", history_adam.history["val_accuracy"][-1])
print("Final SGD val accuracy:", history_sgd.history["val_accuracy"][-1])

print("""
Observation:
Adam typically converges faster and reaches higher accuracy in fewer
epochs than plain SGD, because it adapts the learning rate per-parameter
using estimates of first and second moments of the gradients. SGD (without
momentum) usually needs more epochs and/or a carefully tuned learning
rate to reach comparable accuracy, but can sometimes generalize slightly
better once fully converged.
""")
