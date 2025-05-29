# Home Assignment 1 – Summer 2025  
Student Name: Sowmya Laveti  
Student ID: 700771347  
University: University of Central Missouri  
Course: CS5720 – Neural Networks and Deep Learning  

---

## Assignment Overview  
This assignment is divided into three key parts:

1. Tensor Manipulations & Reshaping
2. Loss Functions & Hyperparameter Tuning
3. Neural Network Training with TensorBoard

---

## Part 1: Tensor Manipulations & Reshaping

### Tasks Completed:
1. Created a random tensor with shape (4, 6).
2. Found its rank and shape using TensorFlow functions.
3. Reshaped it to (2, 3, 4) and transposed it to (3, 2, 4).
4. Broadcasted a smaller tensor of shape (1, 4) and added it to the larger tensor.

### Outputs:
- Original Shape: (4, 6)  
- Rank: 2  
- Reshaped Shape: (2, 3, 4)  
- Transposed Shape: (3, 2, 4)  

### Broadcasting Explanation:
Broadcasting in TensorFlow is a technique that automatically expands the shape of a smaller tensor to match a larger one along compatible dimensions. This allows element-wise operations between tensors of different shapes without needing explicit reshaping or copying data, making computations more efficient and memory-friendly.

---

## Part 2: Loss Functions & Hyperparameter Tuning

### Tasks Completed:
1. Defined sample true labels and model predictions.
2. Calculated:
   - Mean Squared Error (MSE)
   - Categorical Cross-Entropy (CCE)
3. Modified predictions and recalculated the losses.
4. Plotted a bar chart comparing MSE and CCE.

### Results:

| Prediction Version | MSE Loss | CCE Loss |
|--------------------|----------|----------|
| Initial            | 0.0167   | 0.3562   |
| Modified           | 0.0102   | 0.2594   |

### Observations:
1. MSE is more sensitive to small differences across all predicted values.
2. CCE is more appropriate for classification tasks as it heavily penalizes incorrect class probabilities.

---

## Part 3: Neural Network Training with TensorBoard

### Tasks Completed:
1. Loaded and preprocessed the MNIST dataset.
2. Built a simple neural network model.
3. Trained the model for 5 epochs with TensorBoard logging enabled.
4. Analyzed training and validation metrics using TensorBoard.

### Model Configuration:

| Metric           | Value                          |
|------------------|---------------------------------|
| Loss Function     | Sparse Categorical Crossentropy |
| Optimizer         | Adam                            |
| Learning Rate     | Default (0.001)                 |
| Epochs            | 5                               |
| Batch Size        | 32                              |

### Accuracy & Loss Summary:

| Epoch | Accuracy | Loss   | Val Accuracy | Val Loss |
|-------|----------|--------|--------------|----------|
| 1     | 0.8567   | 0.4879 | 0.9565       | 0.1448   |
| 2     | 0.9548   | 0.1518 | 0.9702       | 0.1023   |
| 3     | 0.9675   | 0.1058 | 0.9742       | 0.0832   |
| 4     | 0.9736   | 0.0876 | 0.9747       | 0.0779   |
| 5     | 0.9774   | 0.0730 | 0.9779       | 0.0746   |

### TensorBoard Logs:
- Logs saved in: logs/fit/
- To launch TensorBoard:
  ```bash
  tensorboard --logdir=logs/fit
## Questions & Answers:

### 1. What patterns do you observe in the training and validation accuracy curves?

- Both training and validation accuracy **consistently increase** over the 5 epochs.
- Validation accuracy closely follows training accuracy, with **minimal gap**, indicating the model is generalizing well.
- The **loss values** for both training and validation decrease steadily, showing effective learning without signs of divergence.

### 2. How can you use TensorBoard to detect overfitting?

- TensorBoard lets you visualize **training vs. validation accuracy/loss** in real time.
- **Overfitting indicators**:
  - Training accuracy keeps increasing, but validation accuracy **plateaus or decreases**.
  - Validation loss starts to **increase** while training loss continues to decrease.
- If such divergence is observed, it suggests the model is memorizing the training data and not generalizing well.

### 3. What happens when you increase the number of epochs?

- If the model is well-regularized, increasing epochs may lead to **higher accuracy** and **lower loss**.
- However, training for too long without proper validation can lead to **overfitting**, where validation performance worsens despite improving training performance.
- Using **early stopping** or **monitoring validation loss** through TensorBoard helps decide the optimal number of epochs.
## How to Run
### 1. Clone the Repository
git clone <https://github.com/SowmyaLaveti0304/Sowmya-Laveti.git>
-cd home_assignment_1

### 2. Install Dependencies
pip install -r requirements.txt

### 3. Run the Python Scripts
1. Tensor_Manipulations_Reshaping.py
2. Loss_Functions_Hyperparameter_Tuning.py
3. Logto_TensorBoard.py

### 4. Launch TensorBoard
tensorboard --logdir=logs/fit

#### After launching, open your browser and go to:
#### http://localhost:6006
