"""
Utility functions for HW4: Two-Layer Neural Network
"""

import numpy as np
import torch
import os

# Fix OpenMP conflict issue
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def load_data(filename):
    """
    Load the dataset from .pth file
    """
    print("Loading data file...")
    data = torch.load(filename, weights_only=False)

    # Extract features and labels based on the actual structure
    x = data['features']  # Shape: (20000, 256)
    y = data['labels']    # Shape: (20000,)

    # Convert to numpy arrays for manual implementation
    if torch.is_tensor(x):
        x = x.numpy()
    if torch.is_tensor(y):
        y = y.numpy()

    print(f"Data loaded successfully:")
    print(f"  Features shape: {x.shape}")
    print(f"  Labels shape: {y.shape}")

    return x, y

def initialize_parameters(input_dim=256, hidden_dim=512, output_dim=1):
    """
    Initialize network parameters according to the assignment requirements
    """
    np.random.seed(42)  # For reproducibility

    # Fixed parameters (W1, b1) - remain unchanged during training
    W1 = np.random.randn(hidden_dim, input_dim)  # From N(0, 1)
    b1 = np.random.randn(hidden_dim)             # From N(0, 1)

    # Trainable parameters (W2, b2) - will be updated during training
    W2 = np.random.randn(output_dim, hidden_dim) # From N(0, 1)
    b2 = np.random.randn(output_dim)             # From N(0, 1)

    print(f"Network parameters initialized:")
    print(f"  W1 shape: {W1.shape}")
    print(f"  b1 shape: {b1.shape}")
    print(f"  W2 shape: {W2.shape}")
    print(f"  b2 shape: {b2.shape}")

    return W1, b1, W2, b2

def forward(x, W1, b1, W2, b2):
    """
    Forward propagation through the two-layer network
    """
    # First layer: z1 = W1x + b1
    z1 = np.dot(W1, x) + b1  # (hidden_dim,)

    # Activation: a1 = ReLU(z1) = max(0, z1)
    a1 = np.maximum(0, z1)   # ReLU activation

    # Second layer: y_pred = W2 a1 + b2
    y_pred = np.dot(W2, a1) + b2  # (output_dim,)

    return y_pred[0], a1, z1

def compute_loss(y_true, y_pred):
    """
    Compute Mean Squared Error loss for a single sample
    """
    return (y_true - y_pred) ** 2

def compute_gradients(y_true, y_pred, a1, batch_size):
    """
    Manually compute gradients for W2 and b2 using chain rule
    """
    # Gradient of loss with respect to y_pred
    dL_dy_pred = -2 * (y_true - y_pred) / batch_size

    # Gradient for W2: dL/dW2 = dL/dy_pred * dy_pred/dW2 = dL/dy_pred * a1^T
    dL_dW2 = dL_dy_pred * a1.reshape(1, -1)

    # Gradient for b2: dL/db2 = dL/dy_pred * dy_pred/db2 = dL/dy_pred
    dL_db2 = np.array([dL_dy_pred])

    return dL_dW2, dL_db2

def line_search(W2, b2, grad_W2, grad_b2, x_batch, y_batch, W1, b1, lr_candidates=None):
    """
    Perform line search to find the optimal learning rate
    """
    if lr_candidates is None:
        lr_candidates = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

    best_lr = lr_candidates[0]
    best_loss = float('inf')

    # Save current parameters
    W2_current = W2.copy()
    b2_current = b2.copy()

    for lr in lr_candidates:
        # Temporarily update parameters
        W2_temp = W2_current - lr * grad_W2
        b2_temp = b2_current - lr * grad_b2

        # Compute loss with temporary parameters
        temp_loss = 0
        batch_size = len(x_batch)

        for i in range(batch_size):
            y_pred, _, _ = forward(x_batch[i], W1, b1, W2_temp, b2_temp)
            temp_loss += compute_loss(y_batch[i], y_pred)

        avg_temp_loss = temp_loss / batch_size

        # Update best learning rate if this one gives lower loss
        if avg_temp_loss < best_loss:
            best_loss = avg_temp_loss
            best_lr = lr

    return best_lr