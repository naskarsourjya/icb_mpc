import numpy as np
import matplotlib.pyplot as plt

def conformal_quantile_regression(real_inputs, real_outputs, real_errors, alpha):
    """
    Perform Conformalized Quantile Regression (CQR) using real model data.

    Args:
        real_inputs: Numpy array of input features (not used directly in CQR).
        real_outputs: Numpy array of true outputs.
        real_errors: Numpy array of errors from the model.
        alpha: Miscoverage rate (e.g., 0.1 for 90% coverage).

    Returns:
        lower_bound: Numpy array of lower bound predictions.
        upper_bound: Numpy array of upper bound predictions.
    """
    # Calculate the quantile of residuals (errors)
    quantile = np.quantile(real_errors, 1 - alpha)

    # Compute prediction intervals
    lower_bound = real_outputs - quantile
    upper_bound = real_outputs + quantile

    return lower_bound, upper_bound

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
    # Assume these are your real model's outputs and errors
    np.random.seed(42)
    num_samples = 100

    # Example data
    real_inputs = np.random.rand(num_samples, 3)  # Input features (not used in this example)
    real_outputs = np.sin(np.linspace(0, 10, num_samples))  # True outputs
    real_errors = np.random.normal(0, 0.1, num_samples)  # Errors from the model

    # Perform CQR
    alpha = 0.1  # 90% prediction interval
    lower_bound, upper_bound = conformal_quantile_regression(real_inputs, real_outputs, real_errors, alpha)

    # Display results
    print("Lower Bounds:", lower_bound[:5])
    print("Upper Bounds:", upper_bound[:5])
    print("True Outputs:", real_outputs[:5])

    # Plot results
    plot_results(real_outputs, lower_bound, upper_bound)
