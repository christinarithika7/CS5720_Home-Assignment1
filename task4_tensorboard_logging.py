"""
Task 4: Train a Neural Network and Log to TensorBoard
CS5720 - Neural Networks and Deep Learning - Home Assignment 1
Student Name: Vinay Kumar Christina Rithika Nethakani
Student ID: 700778781

Steps:
1. Load the MNIST dataset and preprocess it.
2. Train a simple neural network model and enable TensorBoard logging.
3. Launch TensorBoard and analyze loss and accuracy trends.

After running this script, launch TensorBoard from a terminal with:
    tensorboard --logdir logs/fit
Then open the URL it prints (usually http://localhost:6006) in a browser.

CODE EXPLANATION
-----------------
1. Same MNIST loading/normalization and model architecture as Task 3,
   but trained only once, with Adam, for 5 epochs.

2. tf.keras.callbacks.TensorBoard is a callback - a function Keras calls
   automatically after each batch/epoch during training. Here, it writes
   loss/accuracy values (and, because histogram_freq=1, weight and
   activation histograms) to the log_dir folder in a special format that
   TensorBoard can read.

   The log directory name includes a timestamp (datetime.now()) so each
   run gets its own folder and old runs aren't overwritten.

3. Passing callbacks=[tensorboard_callback] into model.fit() means these
   logs are written automatically during training - no extra code needed
   per epoch.

4. Running `tensorboard --logdir logs/fit` in a terminal starts a local
   web server that reads all the run folders under logs/fit and displays
   interactive plots of loss and accuracy over epochs (Scalars tab), plus
   weight/activation distributions over time (Histograms tab) - this is
   how you'd visually check for overfitting: if training loss keeps
   dropping while validation loss starts climbing, that gap is the
   overfitting signal.
"""

import datetime
import tensorflow as tf

# ---------------------------------------------------------------------------
# Step 1: Load and preprocess the MNIST dataset
# ---------------------------------------------------------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# ---------------------------------------------------------------------------
# Step 2: Build a simple neural network
# ---------------------------------------------------------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# ---------------------------------------------------------------------------
# Enable TensorBoard logging
# ---------------------------------------------------------------------------
# Logs are stored in a unique, timestamped subdirectory under logs/fit/
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,  # also logs weight/activation histograms each epoch
)

# ---------------------------------------------------------------------------
# Train the model for 5 epochs, logging to TensorBoard
# ---------------------------------------------------------------------------
history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=5,
    callbacks=[tensorboard_callback],
    verbose=2,
)

print(f"\nTraining complete. Logs written to: {log_dir}")
print("Launch TensorBoard with:  tensorboard --logdir logs/fit")

# ---------------------------------------------------------------------------
# 4.1 Questions to Answer
# ---------------------------------------------------------------------------
"""
Q: What patterns do you observe in the training and validation accuracy
   curves?
A: Training accuracy typically increases steadily and smoothly across
   epochs as the model fits the training data. Validation accuracy also
   increases, generally tracking close to (but usually slightly below)
   training accuracy for the first several epochs. If training continues
   too long, training accuracy keeps rising while validation accuracy
   plateaus or starts to dip slightly, indicating the model is beginning
   to overfit.

Q: How can you use TensorBoard to detect overfitting?
A: In TensorBoard's Scalars tab, plot training loss/accuracy alongside
   validation loss/accuracy over epochs. Overfitting is indicated when
   training loss keeps decreasing (and training accuracy keeps
   increasing) while validation loss starts increasing (and validation
   accuracy plateaus or decreases) -- i.e., a growing gap between the
   training and validation curves. TensorBoard's histogram/distribution
   tabs can also show weight distributions growing very large, another
   sign of overfitting.

Q: What happens when you increase the number of epochs?
A: With more epochs, the model continues to fit the training data more
   closely, so training accuracy keeps rising (and training loss keeps
   falling) toward very high/low values. However, beyond a certain point,
   validation accuracy stops improving and validation loss starts to
   increase again, because the model begins memorizing noise/specifics of
   the training set rather than learning generalizable patterns -- i.e.,
   overfitting. This is why techniques like early stopping, dropout, and
   monitoring validation metrics in TensorBoard are used to decide how
   many epochs to actually train for.
"""
