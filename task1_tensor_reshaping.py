"""
Task 1: Tensor Manipulations & Reshaping
CS5720 - Neural Networks and Deep Learning - Home Assignment 1
Student Name: Christina Rithika
Student ID: 700778781

Steps:
1. Create a random tensor of shape (4, 6).
2. Find its rank and shape using TensorFlow functions.
3. Reshape it into (2, 3, 4) and transpose it to (3, 2, 4).
4. Broadcast a smaller tensor (1, 4) to match the larger tensor and add them.
5. Explain how broadcasting works in TensorFlow.

CODE EXPLANATION
-----------------
1. tf.random.uniform(shape=(4,6)) creates a tensor with 4 rows and 6 columns
   (24 elements total), filled with random floats between 0 and 10.

2. tf.rank() tells us how many dimensions the tensor has (2, since it's a
   matrix). tensor.shape gives the size along each dimension: (4, 6).

3. tf.reshape() rearranges the same 24 elements into a new shape (2, 3, 4)
   without changing their order in memory - it just reinterprets how many
   elements go in each dimension. Reshaping only works if the total number
   of elements stays the same (2*3*4 = 24 = 4*6).

   tf.transpose(perm=[1,0,2]) then permutes the axes: axis 0 and axis 1 swap
   places, turning shape (2,3,4) into (3,2,4). Unlike reshape, transpose
   actually reorders the underlying data according to the new axis order.

4. Broadcasting: small_tensor has shape (1,4) and transposed_tensor has
   shape (3,2,4). Since they don't match, TensorFlow tries to broadcast:
   it aligns shapes from the right, treats missing left dimensions as size
   1, and stretches any dimension of size 1 to match the other tensor.
   So (1,4) becomes virtually (1,1,4), then stretched to (3,2,4) - meaning
   the same 4 values are added to every one of the 3*2 = 6 "rows" in the
   larger tensor, without actually duplicating that data in memory.
"""

import tensorflow as tf

# ---------------------------------------------------------------------------
# Step 1: Create a random tensor of shape (4, 6)
# ---------------------------------------------------------------------------
tensor = tf.random.uniform(shape=(4, 6), minval=0, maxval=10, dtype=tf.float32)
print("Original tensor (4, 6):\n", tensor.numpy())

# ---------------------------------------------------------------------------
# Step 2: Find its rank and shape BEFORE reshaping
# ---------------------------------------------------------------------------
print("\n--- Before reshaping/transposing ---")
print("Rank:", tf.rank(tensor).numpy())      # number of dimensions
print("Shape:", tensor.shape)                # (4, 6)

# ---------------------------------------------------------------------------
# Step 3: Reshape into (2, 3, 4) and transpose to (3, 2, 4)
# ---------------------------------------------------------------------------
# 4 * 6 = 24 elements total, and 2 * 3 * 4 = 24, so this reshape is valid.
reshaped_tensor = tf.reshape(tensor, (2, 3, 4))

# Transpose swaps axis 0 and axis 1 (perm defines the new axis order)
transposed_tensor = tf.transpose(reshaped_tensor, perm=[1, 0, 2])

print("\n--- After reshaping to (2, 3, 4) ---")
print("Rank:", tf.rank(reshaped_tensor).numpy())
print("Shape:", reshaped_tensor.shape)

print("\n--- After transposing to (3, 2, 4) ---")
print("Rank:", tf.rank(transposed_tensor).numpy())
print("Shape:", transposed_tensor.shape)

# ---------------------------------------------------------------------------
# Step 4: Broadcast a smaller tensor (1, 4) to match the larger tensor and add
# ---------------------------------------------------------------------------
small_tensor = tf.random.uniform(shape=(1, 4), minval=0, maxval=10, dtype=tf.float32)
print("\nSmall tensor (1, 4):\n", small_tensor.numpy())

broadcast_sum = transposed_tensor + small_tensor

print("\nResult of broadcasting add -> shape:", broadcast_sum.shape)
print(broadcast_sum.numpy())

# ---------------------------------------------------------------------------
# Step 5: Explanation of broadcasting in TensorFlow (also printed at runtime)
# ---------------------------------------------------------------------------
explanation = """
How broadcasting works in TensorFlow
-------------------------------------
Broadcasting lets TensorFlow perform element-wise operations (like +, -, *)
on tensors of different shapes without explicitly copying data to match
their shapes. TensorFlow compares the shapes of the two tensors starting
from the *rightmost* (last) dimension and moving left:

1. If two dimensions are equal, they are compatible as-is.
2. If one of the dimensions is 1, that dimension is "stretched" (virtually
   repeated) to match the other tensor's size in that dimension.
3. If a tensor has fewer dimensions than the other, it is treated as if it
   had size-1 dimensions prepended to it until the ranks match.
4. If dimensions are not equal and neither is 1, the operation is invalid
   and TensorFlow raises a shape-incompatibility error.

In this example, small_tensor has shape (1, 4) and transposed_tensor has
shape (3, 2, 4):
  - small_tensor is first treated as (1, 1, 4) by prepending a size-1 dim.
  - The last dimension (4 vs 4) already matches.
  - The middle dimension (1 vs 2) is stretched from 1 to 2.
  - The first dimension (1 vs 3) is stretched from 1 to 3.
  - Result shape: (3, 2, 4), matching transposed_tensor.

This avoids the memory cost of manually tiling small_tensor into a full
(3, 2, 4) array before adding it.
"""
print(explanation)
