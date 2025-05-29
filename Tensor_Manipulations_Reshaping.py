import tensorflow as tf

#1: Create a random tensor of shape (4, 6)
original_tensor = tf.random.uniform(shape=(4, 6))
print("Original Tensor:\n", original_tensor.numpy())

#2: Find its rank and shape
rank = tf.rank(original_tensor)
shape = tf.shape(original_tensor)
print("\nOriginal Rank:", rank.numpy())
print("Original Shape:", shape.numpy())

#3: Reshape it to (2, 3, 4)
reshaped_tensor = tf.reshape(original_tensor, shape=(2, 3, 4))
print("\nReshaped Tensor Shape (2, 3, 4):", reshaped_tensor.shape)

# Transpose it to (3, 2, 4)
transposed_tensor = tf.transpose(reshaped_tensor, perm=[1, 0, 2])
print("Transposed Tensor Shape (3, 2, 4):", transposed_tensor.shape)

# 4: Broadcasting a smaller tensor (1, 4) and add to the transposed tensor
small_tensor = tf.constant([[1, 2, 3, 4]], dtype=tf.float32)
# Broadcast small_tensor to shape (3, 2, 4) and add
broadcasted_sum = transposed_tensor + small_tensor
print("\nBroadcasted Sum Shape:", broadcasted_sum.shape)

# Optional: print the result of broadcasted addition
print("Broadcasted Tensor Addition Result:\n", broadcasted_sum.numpy())
