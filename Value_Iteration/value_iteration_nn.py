import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import sys

sys.path.append("..")  # Adjust path to import GridWorld
from Envs.GridWorld import GridWorld
from value_iteration_algorithm import ValueIterationSolver
from DQN.DQN import state_size


class SimpleNeuralNetwork:
    """
    Simple Neural Network for Value Function Approximation
    Architecture: Input(2) → Hidden(64) → Output(1)

    This will learn to approximate V(s) from our value iteration solution
    """

    def __init__(
        self,
        input_size: int = 2,
        hidden_size: int = 64,
        output_size: int = 1,
        learning_rate: float = 0.01,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights randomly (Xavier/Glorot initialization)
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))

        # Track training history
        self.loss_history = []

        print(f"🧠 Neural Network Created!")
        print(f"Architecture: {input_size} → {hidden_size} → {output_size}")
        print(f"Learning rate: {learning_rate}")
        print(f"Total parameters: {self.count_parameters()}")
        print()

    def count_parameters(self) -> int:
        """Count total number of trainable parameters"""
        return self.W1.size + self.b1.size + self.W2.size + self.b2.size

    def relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function"""
        return np.maximum(0, x)

    def relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of ReLU"""
        return (x > 0).astype(float)

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Forward propagation

        Args:
            X: Input data (batch_size, input_size)

        Returns:
            output: Network predictions
            cache: Intermediate values for backprop
        """
        # Layer 1: Input → Hidden
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self.relu(z1)

        # Layer 2: Hidden → Output
        z2 = np.dot(a1, self.W2) + self.b2
        output = z2  # Linear output for regression

        # Cache for backpropagation
        cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "output": output}

        return output, cache

    def backward(self, cache: dict, y_true: np.ndarray) -> dict:
        """
        Backward propagation (compute gradients)

        Args:
            cache: Forward pass intermediate values
            y_true: True target values

        Returns:
            gradients: Dictionary of gradients for each parameter
        """
        m = cache["X"].shape[0]  # batch size

        # Output layer gradients
        dz2 = cache["output"] - y_true  # derivative of MSE loss
        dW2 = (1 / m) * np.dot(cache["a1"].T, dz2)
        db2 = (1 / m) * np.sum(dz2, axis=0, keepdims=True)

        # Hidden layer gradients
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.relu_derivative(cache["z1"])
        dW1 = (1 / m) * np.dot(cache["X"].T, dz1)
        db1 = (1 / m) * np.sum(dz1, axis=0, keepdims=True)

        gradients = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

        return gradients

    def update_parameters(self, gradients: dict):
        """Update parameters using gradients"""
        self.W1 -= self.learning_rate * gradients["dW1"]
        self.b1 -= self.learning_rate * gradients["db1"]
        self.W2 -= self.learning_rate * gradients["dW2"]
        self.b2 -= self.learning_rate * gradients["db2"]

    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute Mean Squared Error loss"""
        return np.mean((y_pred - y_true) ** 2)

    def train_step(self, X: np.ndarray, y: np.ndarray) -> float:
        """Single training step"""
        # Forward pass
        y_pred, cache = self.forward(X)

        # Compute loss
        loss = self.compute_loss(y_pred, y)

        # Backward pass
        gradients = self.backward(cache, y)

        # Update parameters
        self.update_parameters(gradients)

        return loss

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        output, _ = self.forward(X)
        return output

    def train(
        self, X: np.ndarray, y: np.ndarray, epochs: int = 1000, verbose: bool = True
    ) -> List[float]:
        """
        Train the neural network

        Args:
            X: Input features (state coordinates)
            y: Target values (value function from value iteration)
            epochs: Number of training iterations
            verbose: Print training progress

        Returns:
            loss_history: Training loss over epochs
        """
        print(f"🚀 Training Neural Network for {epochs} epochs...")
        print("Epoch | Loss     | Progress")
        print("-" * 25)

        self.loss_history = []

        for epoch in range(epochs):
            loss = self.train_step(X, y)
            self.loss_history.append(loss)

            if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
                progress = "█" * int(20 * epoch / epochs) + "░" * (
                    20 - int(20 * epoch / epochs)
                )
                print(f"{epoch:5d} | {loss:.6f} | {progress}")

        print(f"\n🎯 Training Complete! Final loss: {self.loss_history[-1]:.6f}")
        return self.loss_history


def prepare_training_data(env: GridWorld, solver: ValueIterationSolver):
    """
    Prepare training data from value iteration solution

    Returns:
        X: State coordinates (input features)
        y: State values (target values)
    """
    X = []
    y = []

    for state in env.states:
        # Convert state coordinates to features
        r, c = state
        X.append([r, c])

        # Get true value from value iteration
        y.append([solver.values[state]])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def evaluate_network(
    network: SimpleNeuralNetwork, env: GridWorld, solver: ValueIterationSolver
):
    """Evaluate neural network performance against true values"""
    print("\n📊 NEURAL NETWORK EVALUATION")
    print("=" * 50)

    # Prepare test data
    X_test, y_true = prepare_training_data(env, solver)

    # Get predictions
    y_pred = network.predict(X_test)

    # Calculate metrics
    mse = np.mean((y_pred - y_true) ** 2)
    mae = np.mean(np.abs(y_pred - y_true))
    max_error = np.max(np.abs(y_pred - y_true))

    print(f"📈 Performance Metrics:")
    print(f"  Mean Squared Error:  {mse:.6f}")
    print(f"  Mean Absolute Error: {mae:.6f}")
    print(f"  Maximum Error:       {max_error:.6f}")

    # Print state-by-state comparison
    print(f"\n🎯 State-by-State Comparison:")
    print("State    | True Value | NN Prediction | Error")
    print("-" * 45)

    for i, state in enumerate(env.states):
        true_val = y_true[i, 0]
        pred_val = y_pred[i, 0]
        error = abs(true_val - pred_val)

        marker = ""
        if state == env.start_state:
            marker = " (S)"
        elif state == env.goal_state:
            marker = " (G)"
        elif state == env.pit_state:
            marker = " (#)"

        print(
            f"{state}{marker:>4} | {true_val:10.3f} | {pred_val:13.3f} | {error:5.3f}"
        )

    return mse, mae, max_error


def visualize_results(
    network: SimpleNeuralNetwork, env: GridWorld, solver: ValueIterationSolver
):
    """Create visualizations comparing true values vs NN predictions"""
    print("\n🎨 Creating Visualizations...")

    # Prepare data
    X, y_true = prepare_training_data(env, solver)
    y_pred = network.predict(X)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Neural Network Value Function Approximation", fontsize=16)

    # Plot 1: True values heatmap
    true_values = y_true.reshape(env.rows, env.cols)
    im1 = axes[0, 0].imshow(true_values, cmap="viridis", interpolation="nearest")
    axes[0, 0].set_title("True Values (Value Iteration)")
    axes[0, 0].set_xlabel("Column")
    axes[0, 0].set_ylabel("Row")
    plt.colorbar(im1, ax=axes[0, 0])

    # Plot 2: NN predictions heatmap
    pred_values = y_pred.reshape(env.rows, env.cols)
    im2 = axes[0, 1].imshow(pred_values, cmap="viridis", interpolation="nearest")
    axes[0, 1].set_title("NN Predictions")
    axes[0, 1].set_xlabel("Column")
    axes[0, 1].set_ylabel("Row")
    plt.colorbar(im2, ax=axes[0, 1])

    # Plot 3: Error heatmap
    error_values = np.abs(true_values - pred_values)
    im3 = axes[1, 0].imshow(error_values, cmap="Reds", interpolation="nearest")
    axes[1, 0].set_title("Absolute Error")
    axes[1, 0].set_xlabel("Column")
    axes[1, 0].set_ylabel("Row")
    plt.colorbar(im3, ax=axes[1, 0])

    # Plot 4: Training loss
    axes[1, 1].plot(network.loss_history)
    axes[1, 1].set_title("Training Loss")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("MSE Loss")
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

    return fig


# Main execution and testing
if __name__ == "__main__":
    print("🔥 NEURAL NETWORK FROM SCRATCH TEST")
    print("=" * 50)

    # Create environment and solve with value iteration
    print("🎮 Setting up Grid World and Value Iteration...")
    env = GridWorld()
    solver = ValueIterationSolver(env, gamma=0.9)
    solver.value_iteration()
    solver.extract_policy()

    print("✅ Value Iteration Complete!")

    # Prepare training data
    print("\n📊 Preparing Training Data...")
    X_train, y_train = prepare_training_data(env, solver)

    print(f"Training data shape: {X_train.shape} → {y_train.shape}")
    print(f"Sample input (state): {X_train[0]} → target: {y_train[0, 0]:.3f}")

    # Create and train neural network
    print("\n🧠 Creating Neural Network...")
    network = SimpleNeuralNetwork(
        input_size=2, hidden_size=64, output_size=1, learning_rate=0.01
    )

    # Train the network
    loss_history = network.train(X_train, y_train, epochs=1000, verbose=True)

    # Evaluate performance
    mse, mae, max_error = evaluate_network(network, env, solver)

    # Test generalization with new states
    print("\n🧪 Testing Generalization...")
    test_states = [(0.5, 0.5), (1.5, 1.5), (0.2, 1.8)]  # Non-grid positions

    for test_state in test_states:
        X_test = np.array([[test_state[0], test_state[1]]])
        prediction = network.predict(X_test)[0, 0]
        print(f"  State {test_state}: Predicted value = {prediction:.3f}")

    # Compare with tabular lookup
    print("\n⚡ Speed Comparison:")
    import time

    # Time neural network prediction
    start_time = time.time()
    for _ in range(1000):
        network.predict(X_train)
    nn_time = time.time() - start_time

    print(f"  Neural Network (1000 predictions): {nn_time:.4f} seconds")
    print(f"  Tabular Lookup: Instant (dictionary access)")

    print("\n🎯 Key Insights:")
    print(f"  • NN learned to approximate V(s) with {max_error:.3f} max error")
    print(f"  • Can generalize to non-grid positions")
    print(f"  • Scales to larger state spaces (millions of states)")
    print(f"  • Foundation for Deep Q-Networks!")

    # Create visualizations (optional - requires matplotlib)
    try:
        visualize_results(network, env, solver)
        print("📈 Visualizations created!")
    except ImportError:
        print("📈 Install matplotlib for visualizations: pip install matplotlib")

    print("\n🔥 NEURAL NETWORK MISSION COMPLETE!")
    print("💪 You've built the foundation for Deep RL!")
    print("🚀 Tomorrow: Q-Learning and Deep Q-Networks!")
