# Part I — Short Answer Questions

**Student Name:** Vinay kumar Christina Rithika Nethakani
**Student ID:** 700778781

## Question 1

**a. Main difference between traditional programming and machine learning**

In traditional programming, a human writes explicit rules (logic/algorithms), and the program applies those rules to input data to produce output. In machine learning, we instead provide the computer with input data *and* the corresponding output (or a reward signal), and the algorithm learns the rules/patterns on its own by finding statistical relationships in the data. In short: traditional programming = Rules + Data → Output, while machine learning = Data + Output → Rules (model).

**b. Relationship among AI, ML, and Deep Learning**

They are nested subsets of one another:
- **Artificial Intelligence (AI)** is the broadest field — any technique that enables machines to mimic intelligent human behavior (reasoning, planning, perception, language, etc.), including rule-based/expert systems that don't involve learning at all.
- **Machine Learning (ML)** is a subset of AI in which systems learn patterns from data rather than being explicitly programmed with rules.
- **Deep Learning (DL)** is a subset of ML that uses multi-layered (deep) artificial neural networks to automatically learn hierarchical representations of data, rather than relying on hand-crafted features.

So DL ⊂ ML ⊂ AI.

**c. Two reasons deep learning has become more successful in recent years**

1. **Availability of large datasets** — the growth of the internet, digitization, and sensors has produced massive labeled datasets (e.g., ImageNet, large text corpora) that deep networks need to learn effective representations.
2. **Advances in computing hardware** — GPUs/TPUs and parallel computing made it feasible to train very large networks in reasonable time, which was previously computationally prohibitive. (Other valid reasons: better algorithms/architectures such as ReLU, dropout, batch normalization, and the availability of open-source frameworks like TensorFlow/PyTorch that lowered the barrier to experimentation.)

---

## Question 2

**a. Roles of input, hidden, and output layers**

- **Input layer**: Receives the raw feature values of the data (e.g., pixel values, numerical features) and passes them into the network. It performs no computation itself — it just represents the data.
- **Hidden layer(s)**: Perform intermediate computations — each neuron takes a weighted sum of its inputs, adds a bias, and applies a non-linear activation function. Hidden layers progressively transform the input into more abstract, useful representations (features) for the task.
- **Output layer**: Produces the final prediction of the network (e.g., a class probability for classification, or a continuous value for regression), using an activation suited to the task (softmax for multi-class classification, sigmoid for binary classification, linear for regression).

**b. Weights and biases**

- **Weights** are learnable parameters that scale the importance of each input to a neuron — they determine how strongly an input influences the neuron's output.
- **Biases** are learnable parameters added to the weighted sum, allowing the neuron to shift its activation function left/right, so the neuron can still produce a useful (non-zero) output even when all inputs are zero.

An artificial neuron computes: `z = (w1*x1 + w2*x2 + ... + wn*xn) + b`, then passes `z` through an activation function to produce the neuron's output. During training, weights and biases are iteratively adjusted (via gradient descent/backpropagation) to minimize the loss.

**c. Why an activation function is needed**

Without a non-linear activation function, a neural network — no matter how many layers it has — would collapse mathematically into a single linear transformation of the input (since a composition of linear functions is still linear). This would make the network no more powerful than plain linear regression, unable to model complex, non-linear relationships in data. Non-linear activation functions (ReLU, sigmoid, tanh, etc.) allow the network to learn and approximate arbitrarily complex, non-linear functions.

---

## Question 3

**a. What is a perceptron?**

A perceptron is the simplest type of artificial neuron/model, used for binary classification. It takes a set of inputs, computes a weighted sum plus a bias (`z = w·x + b`), and passes this through a step (threshold) activation function. It produces a binary output — typically `1` if `z` is greater than or equal to some threshold (often 0), and `0` (or `-1`) otherwise.

**b. Why a single perceptron can solve AND and OR**

The AND and OR functions are **linearly separable** — meaning you can draw a single straight line (a linear decision boundary) in the input space that correctly separates the class-0 points from the class-1 points. Since a single perceptron computes exactly one linear decision boundary, it can find weights and a bias that correctly implement AND or OR.

**c. Why a single perceptron cannot solve XOR, and how a multilayer network solves this**

XOR is **not linearly separable** — no single straight line can separate the (0,0)/(1,1) outputs (0) from the (0,1)/(1,0) outputs (1); the true points lie in a pattern that requires at least two lines/boundaries to separate. Since a single perceptron can only represent one linear boundary, it cannot represent XOR.

A **multilayer neural network** (with at least one hidden layer and non-linear activations) solves this because the hidden layer neurons can each learn different linear boundaries, and the output layer combines them non-linearly. This lets the network carve out a non-linear (piecewise linear) decision region, which is sufficient to represent XOR (e.g., a 2-neuron hidden layer can each detect one of the two linear regions, and the output neuron combines/XORs their results).

---

## Question 4

**a. Sigmoid vs. Tanh vs. ReLU**

| Activation | Output range | Behavior |
|---|---|---|
| **Sigmoid** | (0, 1) | S-shaped curve; squashes inputs into a probability-like range; saturates (flattens) for large positive/negative inputs, causing very small gradients there. |
| **Tanh** | (-1, 1) | Similar S-shape but zero-centered, which often helps optimization compared to sigmoid; still saturates at extreme values. |
| **ReLU** | [0, ∞) | Linear (identity) for positive inputs and 0 for negative inputs; does not saturate for positive values, is computationally cheap, but can "die" (output 0 and stop learning) if a neuron's input is always negative. |

**b. Vanishing-gradient problem and why ReLU helps**

The vanishing-gradient problem occurs when gradients become extremely small as they are backpropagated through many layers — especially with saturating activations like sigmoid/tanh, whose derivatives approach 0 for large-magnitude inputs. When many such small derivatives are multiplied together across layers (chain rule), the gradient reaching the earlier layers shrinks toward zero, so those layers learn extremely slowly or not at all.

ReLU helps because its derivative is a constant 1 for all positive inputs (no saturation on that side), so gradients can flow backward through many layers without shrinking exponentially, as long as the neuron stays in its active (positive) region.

**c. Neural network training cycle**

1. **Forward propagation**: Input data is passed through the network layer by layer; each neuron computes a weighted sum + bias and applies its activation function, until the output layer produces a prediction.
2. **Error/loss calculation**: The prediction is compared to the true target value using a loss function (e.g., MSE for regression, cross-entropy for classification), producing a scalar loss value that measures how wrong the prediction is.
3. **Backpropagation**: The gradient of the loss with respect to every weight and bias in the network is computed by applying the chain rule backward from the output layer to the input layer.
4. **Weight update**: An optimizer (e.g., SGD, Adam) uses these gradients to adjust the weights and biases — typically by moving them a small step in the direction that reduces the loss (gradient descent), scaled by a learning rate.

This cycle repeats for many iterations/epochs over the training data until the loss converges to an acceptably low value.
