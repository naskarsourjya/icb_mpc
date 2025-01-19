import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def pinball_loss(predictions, targets, quantile):
    """
    Compute the pinball loss for a given quantile.

    Args:
        predictions: Predicted values (torch.Tensor).
        targets: True values (torch.Tensor).
        quantile: Quantile value (float, e.g., 0.1 for 10%).

    Returns:
        Pinball loss (torch.Tensor).
    """
    diff = targets - predictions
    loss = torch.maximum(quantile * diff, (quantile - 1) * diff)
    return loss.mean()

def train_cqr_model(dataset, model, optimizer, quantiles, epochs):
    """
    Train the model for upper and lower quantile regression using pinball loss.

    Args:
        dataset: Torch Dataset containing inputs and outputs.
        model: PyTorch model.
        optimizer: Optimizer for training.
        quantiles: List of quantiles to train on (e.g., [0.1, 0.9]).
        epochs: Number of training epochs.

    Returns:
        Trained model.
    """
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, targets in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = 0
            for i, q in enumerate(quantiles):
                loss += pinball_loss(outputs[:, i], targets, q)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")
    return model

def plot_results(real_outputs, lower_bound, upper_bound):
    """
    Plot the real outputs along with the lower and upper bounds.

    Args:
        real_outputs: Numpy array of true outputs.
        lower_bound: Numpy array of lower bound predictions.
        upper_bound: Numpy array of upper bound predictions.
    """
    x = np.arange(len(real_outputs))
    plt.figure(figsize=(10, 6))
    plt.plot(x, real_outputs, label="Real Outputs", color="blue")
    plt.plot(x, lower_bound, label="Lower Bound", color="red", linestyle="--")
    plt.plot(x, upper_bound, label="Upper Bound", color="green", linestyle="--")
    plt.fill_between(x, lower_bound, upper_bound, color="gray", alpha=0.2, label="Prediction Interval")
    plt.xlabel("Sample Index")
    plt.ylabel("Value")
    plt.title("Conformalized Quantile Regression Results")
    plt.legend()
    plt.grid()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    torch.manual_seed(42)
    num_samples = 100

    inputs = np.linspace(0, 10, num_samples).reshape(-1, 1)
    outputs = np.sin(inputs) + 0.1 * np.random.normal(size=inputs.shape)

    # Convert to PyTorch tensors
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
    outputs_tensor = torch.tensor(outputs, dtype=torch.float32)

    # Create dataset
    dataset = TensorDataset(inputs_tensor, outputs_tensor)

    # Define model
    class QuantileModel(nn.Module):
        def __init__(self, input_dim, quantiles):
            super(QuantileModel, self).__init__()
            self.quantiles = quantiles
            self.network = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.Tanh(),
                nn.Linear(64, len(quantiles))
            )

        def forward(self, x):
            return self.network(x)

    quantiles = [0.1, 0.9]
    model = QuantileModel(input_dim=1, quantiles=quantiles)

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Train model
    epochs = 100
    model = train_cqr_model(dataset, model, optimizer, quantiles, epochs)

    # Evaluate model
    model.eval()
    with torch.no_grad():
        predictions = model(inputs_tensor).numpy()
        lower_bound = predictions[:, 0]
        upper_bound = predictions[:, 1]

    # Plot results
    plot_results(outputs.flatten(), outputs.flatten()+lower_bound, outputs.flatten()+upper_bound)
