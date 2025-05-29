import matplotlib.pyplot as plt
import tensorflow as tf

# Define true values and predictions
y_true = tf.constant([[0., 1., 0.]], dtype=tf.float32)
y_pred1 = tf.constant([[0.2, 0.7, 0.1]])
y_pred2 = tf.constant([[0.3, 0.4, 0.3]])

# Compute MSE & CCE losses
mse1 = tf.keras.losses.MeanSquaredError()(y_true, y_pred1).numpy()
mse2 = tf.keras.losses.MeanSquaredError()(y_true, y_pred2).numpy()
cce1 = tf.keras.losses.CategoricalCrossentropy()(y_true, y_pred1).numpy()
cce2 = tf.keras.losses.CategoricalCrossentropy()(y_true, y_pred2).numpy()

print(f"MSE1: {mse1}, MSE2: {mse2}")
print(f"CCE1: {cce1}, CCE2: {cce2}")

# Plotting
labels = ['Pred1', 'Pred2']
mse_vals = [mse1, mse2]
cce_vals = [cce1, cce2]

x = range(len(labels))
plt.bar(x, mse_vals, width=0.3, label='MSE', align='center')
plt.bar([p + 0.3 for p in x], cce_vals, width=0.3, label='CCE', align='center')
plt.xticks([p + 0.15 for p in x], labels)
plt.ylabel('Loss Value')
plt.title('MSE vs CCE for Predictions')
plt.legend()
plt.tight_layout()
plt.show()