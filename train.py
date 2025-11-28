import numpy as np
import matplotlib.pyplot as plt
from classes.network import SelfModifyingNetwork
from setup.gpu_config import get_device_config


def generate_nonlinear_data(n_samples: int, noise: float = 0.1):
    """
    Generate a nonlinear regression dataset.
    y = sin(2*x) + 0.5*x^2 + noise
    
    Args:
        n_samples: Number of samples
        noise: Noise level
        
    Returns:
        X, y arrays
    """
    X = np.random.uniform(-3, 3, (n_samples, 1))
    y = np.sin(2 * X) + 0.5 * X**2 + np.random.randn(n_samples, 1) * noise
    return X, y


def train_regression():
    """Train the self-modifying network on a regression task."""
    # Initialize GPU/CPU device
    device_config = get_device_config(force_cpu=False)
    
    print("\nInitializing Self-Modifying Neural Network for Regression")
    print("="*60)
    
    # Create network
    network = SelfModifyingNetwork(
        input_dim=1,
        output_dim=1,
        initial_layers=2,
        initial_neurons_per_layer=4,
        learning_rate=0.01,
        device_config=device_config
    )
    
    # Generate training data
    n_train = 1000
    X_train, y_train = generate_nonlinear_data(n_train, noise=0.1)
    
    # Generate test data
    n_test = 200
    X_test, y_test = generate_nonlinear_data(n_test, noise=0.05)
    
    # Training loop
    n_epochs = 20
    samples_per_epoch = 100
    
    # Track metrics
    epoch_rewards = []
    epoch_errors = []
    network_sizes = []
    
    print(f"\nTraining for {n_epochs} epochs, {samples_per_epoch} samples per epoch")
    print("-"*60)
    
    for epoch in range(n_epochs):
        # Shuffle training data
        indices = np.random.permutation(n_train)
        
        epoch_reward_sum = 0
        
        for i in range(samples_per_epoch):
            idx = indices[i % n_train]
            x = X_train[idx]
            target = y_train[idx]
            
            # Train step (unified training/prediction)
            reward = network.train_step(x, target)
            epoch_reward_sum += reward
        
        # Epoch statistics
        avg_epoch_reward = epoch_reward_sum / samples_per_epoch
        avg_reward_50 = network.reward_function.get_average_reward(50)
        avg_error_50 = network.reward_function.get_average_error(50)
        
        epoch_rewards.append(avg_reward_50)
        epoch_errors.append(avg_error_50)
        
        # Network size
        stats = network.get_network_stats()
        network_sizes.append(stats['total_neurons'])
        
        # Print progress
        print(f"Epoch {epoch+1:2d}: Reward={avg_epoch_reward:6.2f}, "
              f"Avg50={avg_reward_50:6.2f}, Error={avg_error_50:.4f}, "
              f"Layers={stats['total_layers']}, Neurons={stats['total_neurons']}")
    
    print("\n" + "="*60)
    print("Training Complete!")
    network.print_network_summary()
    
    # Test the network
    print("\nTesting on held-out data...")
    test_predictions = []
    test_rewards = []
    
    for i in range(n_test):
        x = X_test[i]
        target = y_test[i]
        
        prediction = network.predict(x)
        reward = network.reward_function.compute_reward(prediction, target)
        
        test_predictions.append(prediction[0])
        test_rewards.append(reward)
    
    avg_test_reward = np.mean(test_rewards)
    print(f"Average Test Reward: {avg_test_reward:.3f}")
    
    # Visualize results
    visualize_training(
        epoch_rewards, 
        epoch_errors, 
        network_sizes,
        X_test, 
        y_test, 
        test_predictions
    )


def visualize_training(epoch_rewards, epoch_errors, network_sizes, X_test, y_test, predictions):
    """Visualize training progress and results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Reward over epochs
    axes[0, 0].plot(epoch_rewards, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Average Reward')
    axes[0, 0].set_title('Reward Progress')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Error over epochs
    axes[0, 1].plot(epoch_errors, 'r-', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Average Error')
    axes[0, 1].set_title('Error Progress')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Network size over epochs
    axes[1, 0].plot(network_sizes, 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Total Neurons')
    axes[1, 0].set_title('Network Size Evolution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Predictions vs Actual
    sorted_idx = np.argsort(X_test.flatten())
    axes[1, 1].scatter(X_test[sorted_idx], y_test[sorted_idx], alpha=0.5, label='Actual', s=20)
    axes[1, 1].plot(X_test[sorted_idx], np.array(predictions)[sorted_idx], 'r-', 
                    linewidth=2, label='Predicted')
    axes[1, 1].set_xlabel('Input')
    axes[1, 1].set_ylabel('Output')
    axes[1, 1].set_title('Predictions vs Actual (Test Set)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved as 'training_results.png'")
    plt.show()


def train_classification():
    """Train the network on a simple classification task."""
    print("\nInitializing Self-Modifying Neural Network for Classification")
    print("="*60)
    
    # Create network
    network = SelfModifyingNetwork(
        input_dim=2,
        output_dim=1,
        initial_layers=2,
        initial_neurons_per_layer=5,
        learning_rate=0.02
    )
    
    # Generate XOR-like dataset
    n_samples = 500
    X = np.random.randn(n_samples, 2)
    # XOR pattern: positive if signs differ
    y = ((X[:, 0] > 0) != (X[:, 1] > 0)).astype(float).reshape(-1, 1)
    
    print(f"\nTraining on XOR classification ({n_samples} samples)")
    print("-"*60)
    
    # Training loop
    for step in range(500):
        idx = np.random.randint(0, n_samples)
        x = X[idx]
        target = y[idx]
        
        reward = network.train_step(x, target)
        
        if (step + 1) % 100 == 0:
            stats = network.get_network_stats()
            print(f"Step {step+1:3d}: Reward={reward:6.2f}, "
                  f"Avg50={stats['avg_reward_50']:6.2f}, "
                  f"Layers={stats['total_layers']}, "
                  f"Neurons={stats['total_neurons']}")
    
    print("\n" + "="*60)
    network.print_network_summary()


def demonstrate_dynamic_growth():
    """Demonstrate the network's ability to grow and shrink."""
    print("\nDemonstrating Dynamic Network Growth/Shrinking")
    print("="*60)
    
    network = SelfModifyingNetwork(
        input_dim=3,
        output_dim=2,
        initial_layers=2,
        initial_neurons_per_layer=3,
        learning_rate=0.01
    )
    
    print("Initial network:")
    print(network)
    
    # Phase 1: Easy task (network should shrink)
    print("\nPhase 1: Easy task (linear)")
    for i in range(100):
        x = np.random.randn(3)
        target = np.array([x[0] + x[1], x[2]])
        network.train_step(x, target)
    
    print(f"After easy task: {network}")
    
    # Phase 2: Hard task (network should grow)
    print("\nPhase 2: Hard task (nonlinear)")
    for i in range(200):
        x = np.random.randn(3)
        target = np.array([np.sin(x[0]) * np.cos(x[1]), x[2]**2])
        network.train_step(x, target)
    
    print(f"After hard task: {network}")
    
    network.print_network_summary()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("SELF-MODIFYING NEURAL NETWORK - TRAINING DEMO")
    print("="*60)
    
    # Run regression task
    print("\n### TASK 1: NONLINEAR REGRESSION ###")
    train_regression()
    
    # Run classification task
    print("\n### TASK 2: CLASSIFICATION (XOR) ###")
    train_classification()
    
    # Demonstrate dynamic behavior
    print("\n### TASK 3: DYNAMIC GROWTH DEMONSTRATION ###")
    demonstrate_dynamic_growth()
    
    print("\n" + "="*60)
    print("All demonstrations complete!")
    print("="*60)
