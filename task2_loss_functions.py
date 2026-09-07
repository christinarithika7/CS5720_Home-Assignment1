"""
Task 2: Loss Functions & Hyperparameter Tuning
CS5720 - Neural Networks and Deep Learning - Home Assignment 1
Student Name: Christina Rithika
Student ID: 700778781

Steps:
1. Define true values (y_true) and model predictions (y_pred).
2. Compute Mean Squared Error (MSE) and Categorical Cross-Entropy (CCE) losses.
3. Modify predictions slightly and check how loss values change.
4. Plot loss function values using Matplotlib.

CODE EXPLANATION
-----------------
1. y_true holds one-hot encoded ground-truth labels for 3 samples across
   3 classes (e.g., row 1 = [0,1,0] means the true class is index 1).
   y_pred holds the model's predicted class probabilities for each sample.

2. MeanSquaredError() computes the average squared difference between
   every predicted value and every true value: mean((y_true - y_pred)^2).
   It treats all classes equally and doesn't care that predictions should
   sum to 1.

   CategoricalCrossentropy() instead only looks at the predicted
   probability assigned to the TRUE class, and penalizes it with
   -log(predicted_probability). This means confident wrong predictions
   (predicting a low probability for the correct class) are punished much
   more heavily than with MSE, because -log() grows very large as the
   probability approaches 0.

3. By creating y_pred_better (predictions closer to the one-hot targets)
   and y_pred_worse (predictions closer to uniform/incorrect), we can see
   both loss functions decrease as predictions improve and increase as
   predictions get worse - confirming both are valid "distance from
   truth" measures, but CCE reacts more sharply to confident errors.

4. The bar chart plots MSE and CCE side by side for the worse/original/
   better cases so you can visually compare how much more steeply CCE
   changes compared to MSE.
"""

import tensorflow as tf
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Step 1: Define true values (one-hot labels) and model predictions
# ---------------------------------------------------------------------------
# y_true: one-hot encoded ground-truth labels for a 3-class classification example
y_true = tf.constant([[0, 1, 0],
                       [1, 0, 0],
                       [0, 0, 1]], dtype=tf.float32)

# y_pred: predicted probabilities from a model (rows sum to ~1)
y_pred = tf.constant([[0.1, 0.8, 0.1],
                       [0.7, 0.2, 0.1],
                       [0.2, 0.2, 0.6]], dtype=tf.float32)

# ---------------------------------------------------------------------------
# Step 2: Compute MSE and Categorical Cross-Entropy losses
# ---------------------------------------------------------------------------
mse = tf.keras.losses.MeanSquaredError()
cce = tf.keras.losses.CategoricalCrossentropy()

mse_loss = mse(y_true, y_pred).numpy()
cce_loss = cce(y_true, y_pred).numpy()

print("Original predictions:")
print("MSE Loss:", mse_loss)
print("Categorical Cross-Entropy Loss:", cce_loss)

# ---------------------------------------------------------------------------
# Step 3: Modify predictions slightly and check how loss values change
# ---------------------------------------------------------------------------
# Make the predictions slightly "better" (closer to the true one-hot labels)
y_pred_better = tf.constant([[0.05, 0.9, 0.05],
                              [0.85, 0.1, 0.05],
                              [0.1, 0.1, 0.8]], dtype=tf.float32)

# Make the predictions slightly "worse" (further from the true labels)
y_pred_worse = tf.constant([[0.3, 0.4, 0.3],
                             [0.4, 0.4, 0.2],
                             [0.4, 0.3, 0.3]], dtype=tf.float32)

mse_better = mse(y_true, y_pred_better).numpy()
cce_better = cce(y_true, y_pred_better).numpy()

mse_worse = mse(y_true, y_pred_worse).numpy()
cce_worse = cce(y_true, y_pred_worse).numpy()

print("\nBetter predictions (closer to y_true):")
print("MSE Loss:", mse_better)
print("Categorical Cross-Entropy Loss:", cce_better)

print("\nWorse predictions (further from y_true):")
print("MSE Loss:", mse_worse)
print("Categorical Cross-Entropy Loss:", cce_worse)

print("""
Observation:
As predictions move closer to the true one-hot labels, both MSE and CCE
decrease. As predictions move further away (more uniform/incorrect),
both losses increase. Categorical Cross-Entropy tends to penalize
confident wrong predictions more heavily than MSE does, since it uses a
logarithmic penalty on the predicted probability of the true class.
""")

# ---------------------------------------------------------------------------
# Step 4: Plot loss function values using Matplotlib
# ---------------------------------------------------------------------------
labels = ["Worse", "Original", "Better"]
mse_values = [mse_worse, mse_loss, mse_better]
cce_values = [cce_worse, cce_loss, cce_better]

x = range(len(labels))
width = 0.35

plt.figure(figsize=(7, 5))
plt.bar([i - width / 2 for i in x], mse_values, width=width, label="MSE")
plt.bar([i + width / 2 for i in x], cce_values, width=width, label="Categorical Cross-Entropy")
plt.xticks(list(x), labels)
plt.ylabel("Loss value")
plt.title("MSE vs. Categorical Cross-Entropy Loss for Different Predictions")
plt.legend()
plt.tight_layout()
plt.savefig("mse_vs_cce_loss.png")
plt.show()

print("Bar chart saved as 'mse_vs_cce_loss.png'")
