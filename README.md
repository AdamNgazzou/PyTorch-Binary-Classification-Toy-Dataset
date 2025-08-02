# Neural Network Classification with PyTorch

This project demonstrates how to build and train a neural network for binary classification using PyTorch. The goal is to classify data points that form two concentric circles.

## 1. Data Preparation

- **Dataset:** The project uses the `make_circles` function from scikit-learn to generate a toy dataset of 1000 samples with two distinct classes.
- **Visualization:** The initial data is visualized to show the circular distribution of the two classes.
- **Data Conversion:** The generated NumPy arrays are converted into PyTorch tensors to be used with the neural network.
- **Train-Test Split:** The data is split into training and testing sets to evaluate the model's performance on unseen data.

## 2. Model Building

Two models are constructed in this project:

### Model 0: A Simple Linear Model

- **Architecture:** A simple model with two linear layers.
- **Activation Function:** No activation function is used between the layers.
- **Loss Function:** `BCEWithLogitsLoss` which combines a Sigmoid layer and the BCELoss in one single class.
- **Optimizer:** Stochastic Gradient Descent (SGD).
- **Result:** This model fails to learn the non-linear boundary of the circular data, resulting in poor accuracy.

### Model 1: A Deeper Model with Non-Linearity

- **Architecture:** A more complex model with multiple linear layers and ReLU activation functions.
- **Activation Function:** ReLU is used to introduce non-linearity, allowing the model to learn more complex patterns.
- **Loss Function:** `BCEWithLogitsLoss`.
- **Optimizer:** Adam optimizer.
- **Result:** This model successfully learns the non-linear decision boundary, achieving high accuracy on both the training and testing sets.

## 3. Training and Evaluation

- **Training Loop:** A standard PyTorch training loop is implemented to train the models. This includes:
    - Forward pass
    - Loss calculation
    - Backpropagation
    - Optimizer step
- **Evaluation:** The models are evaluated using accuracy as the primary metric. The decision boundaries of the trained models are visualized to understand their performance.

## 4. Key Takeaways

- **Importance of Non-Linearity:** The project highlights the importance of using non-linear activation functions (like ReLU) in neural networks to learn complex, non-linear patterns in data.
- **Hyperparameter Tuning:** The improvement from `model_0` to `model_1` demonstrates how changing the model's architecture, activation functions, and optimizer can significantly impact performance.
- **PyTorch Workflow:** The project provides a clear example of the end-to-end workflow for a classification task in PyTorch, from data preparation to model training and evaluation.
