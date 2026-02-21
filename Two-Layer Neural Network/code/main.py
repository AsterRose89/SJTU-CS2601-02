"""
HW4: Two-Layer Neural Network
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Fix OpenMP conflict issue
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from utils import load_data, initialize_parameters, forward, compute_loss, compute_gradients, line_search

def train_model(x, y, epochs=200, batch_size=64, validation_ratio=0.1):
    """
    Main training function for the two-layer neural network
    """
    # Get actual data dimensions
    n_samples, input_dim = x.shape
    print(f"Training with {n_samples} samples, input dimension: {input_dim}")

    # Initialize network parameters
    hidden_dim = 512
    W1, b1, W2, b2 = initialize_parameters(input_dim=input_dim, hidden_dim=hidden_dim)

    # Data splitting
    n_val = int(n_samples * validation_ratio)
    n_train = n_samples - n_val

    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    x_train, y_train = x[train_indices], y[train_indices]
    x_val, y_val = x[val_indices], y[val_indices]

    print(f"Training samples: {n_train}, Validation samples: {n_val}")

    # Training history
    train_losses = []
    val_losses = []
    learning_rates = []

    print("\nStarting training...")
    print("Epoch\tTrain Loss\tVal Loss\tLearning Rate")
    print("-" * 50)

    for epoch in range(epochs):
        # Shuffle training data
        train_indices_shuffled = np.random.permutation(n_train)
        x_train_shuffled = x_train[train_indices_shuffled]
        y_train_shuffled = y_train[train_indices_shuffled]

        epoch_train_loss = 0
        grad_W2_accum = np.zeros_like(W2)
        grad_b2_accum = np.zeros_like(b2)

        # Mini-batch training
        for i in range(0, n_train, batch_size):
            batch_end = min(i + batch_size, n_train)
            x_batch = x_train_shuffled[i:batch_end]
            y_batch = y_train_shuffled[i:batch_end]
            actual_batch_size = len(x_batch)

            batch_loss = 0
            batch_grad_W2 = np.zeros_like(W2)
            batch_grad_b2 = np.zeros_like(b2)

            for j in range(actual_batch_size):
                # Forward pass
                y_pred, a1, _ = forward(x_batch[j], W1, b1, W2, b2)

                # Compute loss and gradients
                sample_loss = compute_loss(y_batch[j], y_pred)
                batch_loss += sample_loss

                grad_W2, grad_b2 = compute_gradients(y_batch[j], y_pred, a1, actual_batch_size)
                batch_grad_W2 += grad_W2
                batch_grad_b2 += grad_b2

            epoch_train_loss += batch_loss
            grad_W2_accum += batch_grad_W2
            grad_b2_accum += batch_grad_b2

        # Average gradients and loss
        avg_grad_W2 = grad_W2_accum / n_train
        avg_grad_b2 = grad_b2_accum / n_train
        avg_train_loss = epoch_train_loss / n_train

        # Line search for learning rate
        best_lr = line_search(W2, b2, avg_grad_W2, avg_grad_b2,
                             x_train_shuffled[:min(50, n_train)],
                             y_train_shuffled[:min(50, n_train)],
                             W1, b1)
        learning_rates.append(best_lr)

        # Update parameters
        W2 = W2 - best_lr * avg_grad_W2
        b2 = b2 - best_lr * avg_grad_b2

        # Validation loss
        val_loss = 0
        for i in range(n_val):
            y_pred_val, _, _ = forward(x_val[i], W1, b1, W2, b2)
            val_loss += compute_loss(y_val[i], y_pred_val)
        avg_val_loss = val_loss / n_val

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # Print progress
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"{epoch+1:4d}\t{avg_train_loss:.6f}\t{avg_val_loss:.6f}\t{best_lr:.6f}")

    return W1, b1, W2, b2, train_losses, val_losses, learning_rates

def plot_results(train_losses, val_losses, learning_rates):
    """
    Plot training results including loss curves and learning rate history
    """
    plt.style.use('seaborn-v0_8')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot loss curves
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2, alpha=0.8)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('MSE Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss Curves', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # Log scale for better visualization

    # Plot learning rate history
    ax2.plot(epochs, learning_rates, 'g-', linewidth=2, alpha=0.8)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Learning Rate', fontsize=12)
    ax2.set_title('Learning Rate History (Line Search)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig('loss_curve.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("Loss curve saved as 'loss_curve.png'")

def main():
    """
    Main function to run the complete training process
    """
    print("HW4: Two-Layer Neural Network Implementation")
    print("=" * 50)

    try:
        # Load data
        x, y = load_data('homework_features_256_20000.pth')
        print(f"\nData successfully loaded!")
        print(f"Features shape: {x.shape}")
        print(f"Targets shape: {y.shape}")
        print(f"Data range - Features: [{x.min():.3f}, {x.max():.3f}]")
        print(f"Data range - Targets: [{y.min():.3f}, {y.max():.3f}]")

        # Display data statistics
        print(f"\nData statistics:")
        print(f"Features - Min: {x.min():.3f}, Max: {x.max():.3f}, Mean: {x.mean():.3f}, Std: {x.std():.3f}")
        print(f"Labels - Min: {y.min():.3f}, Max: {y.max():.3f}, Mean: {y.mean():.3f}, Std: {y.std():.3f}")

        # Train model
        W1, b1, W2, b2, train_losses, val_losses, learning_rates = train_model(x, y)

        # Plot results
        print("\nGenerating loss curve...")
        plot_results(train_losses, val_losses, learning_rates)

        # Final evaluation
        print("\n" + "=" * 50)
        print("Training Completed Successfully!")
        print(f"Final Training Loss: {train_losses[-1]:.6f}")
        print(f"Final Validation Loss: {val_losses[-1]:.6f}")
        print(f"Final Learning Rate: {learning_rates[-1]:.6f}")
        print(f"Training converged after {len(train_losses)} epochs")

        # Calculate improvement
        improvement = (train_losses[0] - train_losses[-1]) / train_losses[0] * 100
        print(f"Training loss improved by {improvement:.1f}%")

        # Save model parameters
        np.savez('model_parameters.npz', W1=W1, b1=b1, W2=W2, b2=b2)
        print("Model parameters saved to 'model_parameters.npz'")

    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()