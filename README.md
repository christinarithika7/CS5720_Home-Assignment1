# CS5720 – Home Assignment 1
### Neural Networks and Deep Learning — Fall 2026

**Student Name:** Christina Rithika
**Student ID:** 700778781
**Course:** CS5720 Neural Network and Deep Learning
**University:** University of Central Missouri

---

## Overview

This repository contains my solutions for Home Assignment 1, covering:

- **Part I** — Short-answer theory questions on ML/DL fundamentals, neural network layers, perceptrons, and activation functions (see `Part1_ShortAnswers.md`).
- **Part II** — Four programming tasks implemented in TensorFlow/Keras:
  1. Tensor reshaping & broadcasting (`task1_tensor_reshaping.py`)
  2. Loss function comparison — MSE vs. Categorical Cross-Entropy (`task2_loss_functions.py`)
  3. Optimizer comparison — Adam vs. SGD on MNIST (`task3_optimizer_comparison.py`)
  4. Training with TensorBoard logging (`task4_tensorboard_logging.py`)

## Repository Structure

```
CS5720_HA1/
├── README.md
├── Part1_ShortAnswers.md
├── task1_tensor_reshaping.py
├── task2_loss_functions.py
├── task3_optimizer_comparison.py
└── task4_tensorboard_logging.py
```

## Requirements

```
pip install tensorflow matplotlib
```

## How to Run

Each task is a standalone script:

```bash
python task1_tensor_reshaping.py
python task2_loss_functions.py
python task3_optimizer_comparison.py
python task4_tensorboard_logging.py
```

For Task 4, after running the script, launch TensorBoard:

```bash
tensorboard --logdir logs/fit
```

Then open the printed URL (usually `http://localhost:6006`) in your browser to view training/validation loss and accuracy curves.

## Task Summaries & Results

### Task 1 — Tensor Reshaping
- Creates a random `(4, 6)` tensor, prints its rank/shape.
- Reshapes to `(2, 3, 4)`, transposes to `(3, 2, 4)`.
- Broadcasts a `(1, 4)` tensor against it and adds them.
- Includes a written explanation of how TensorFlow broadcasting works.

### Task 2 — Loss Functions
- Computes MSE and Categorical Cross-Entropy for a sample prediction.
- Shows how loss values shift for "better" vs. "worse" predictions.
- Produces `mse_vs_cce_loss.png`, a bar chart comparing the two losses.

### Task 3 — Optimizer Comparison
- Trains identical models on MNIST — one with Adam, one with SGD.
- Produces `adam_vs_sgd_accuracy.png` comparing training/validation accuracy curves.

### Task 4 — TensorBoard Logging
- Trains a simple neural network on MNIST for 5 epochs with a `TensorBoard` callback.
- Logs are saved under `logs/fit/`.
- Written answers to the reflection questions (overfitting patterns, using TensorBoard to detect overfitting, effect of increasing epochs) are included as comments at the bottom of the script.

## Video Demonstration

_[Add your 2–3 minute video link here — demonstrating the code running and briefly explaining key snippets, per the assignment's submission requirements.]_

## Notes

- Remember to **comment your code** appropriately (already done in each script, but review the code so you can explain it fluently in your video).
- Submit both your **GitHub repo link** and your **video** on Bright Space before the deadline.
