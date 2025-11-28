import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pickle
import time
import threading
from classes.network import SelfModifyingNetwork
from setup.gpu_config import get_device_config


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

# Continuous training parameters
SAMPLES_PER_BATCH = 50  # Train on this many samples before allowing predictions
SAVE_INTERVAL_BATCHES = 100  # Auto-save every N batches
MAX_DIGIT = 9  # Maximum digit for operations

# Task types
TASK_ADDITION = 0
TASK_DIVISION = 1


# ============================================================================
# SAVE/LOAD FUNCTIONS
# ============================================================================

CHECKPOINT_DIR = "checkpoints"
NETWORK_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "network_state.pkl")

def save_network(network, filepath=NETWORK_CHECKPOINT):
    """Save the entire network state to disk."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    state = {
        'network': network,
        'training_steps': network.training_steps,
        'modification_tracker': network.modification_tracker if hasattr(network, 'modification_tracker') else None,
        'meta_learner': network.meta_learner if hasattr(network, 'meta_learner') else None
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(state, f)
    print(f"✓ Network saved to {filepath}")

def load_network(filepath=NETWORK_CHECKPOINT):
    """Load network state from disk."""
    if not os.path.exists(filepath):
        return None
    
    try:
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        print(f"✓ Network loaded from {filepath} (steps: {state['training_steps']})")
        return state['network']
    except Exception as e:
        print(f"✗ Failed to load network: {e}")
        return None


# ============================================================================
# DATA GENERATION
# ============================================================================

def generate_addition_data(n_samples: int, max_digit: int = 9):
    """
    Generate addition dataset with 4-element input: [num1, num2, 0, 0].
    The last two zeros indicate this is an addition operation (not division).
    
    Args:
        n_samples: Number of samples
        max_digit: Maximum digit value (default 9 for single digits)
        
    Returns:
        X, y arrays where X is [num1, num2, 0, 0] and y is [sum]
    """
    X = np.zeros((n_samples, 4))
    y = np.zeros((n_samples, 1))
    
    for i in range(n_samples):
        num1 = np.random.randint(0, max_digit + 1)
        num2 = np.random.randint(0, max_digit + 1)
        
        # Format: [num1, num2, 0, 0] - last two zeros signal addition
        X[i] = [num1, num2, 0, 0]
        y[i] = num1 + num2
    
    return X, y


def generate_division_data(n_samples: int, max_digit: int = 9):
    """
    Generate division dataset with 4-element input: [0, 0, dividend, divisor].
    The first two zeros indicate this is a division operation (not addition).
    Ensures no division by zero.
    
    Args:
        n_samples: Number of samples
        max_digit: Maximum digit value (default 9 for single digits)
        
    Returns:
        X, y arrays where X is [0, 0, dividend, divisor] and y is [quotient]
    """
    X = np.zeros((n_samples, 4))
    y = np.zeros((n_samples, 1))
    
    for i in range(n_samples):
        dividend = np.random.randint(1, max_digit + 1)
        divisor = np.random.randint(1, max_digit + 1)
        
        # Format: [0, 0, dividend, divisor] - first two zeros signal division
        X[i] = [0, 0, dividend, divisor]
        y[i] = dividend / divisor
    
    return X, y


def train_level_1_addition(network=None):
    """Level 1: Train network to add single digit numbers."""
    print("\n" + "="*70)
    print("LEVEL 1: ADDITION OF SINGLE DIGIT NUMBERS")
    print("="*70)
    print("Task: Learn to add two single digit numbers (0-9)")
    print("Example: [3, 8, 0, 0] -> 11, [5, 2, 0, 0] -> 7, [9, 9, 0, 0] -> 18")
    print("Format: [num1, num2, 0, 0] where last two zeros signal addition")
    print("-"*70)
    
    # Create or use existing network
    if network is None:
        # Initialize GPU/CPU device
        device_config = get_device_config(force_cpu=False)
        
        network = SelfModifyingNetwork(
            input_dim=4,      # Four elements: [num1, num2, 0, 0]
            output_dim=1,     # One result output
            initial_layers=2,  # Start with 2 layers for better capacity
            initial_neurons_per_layer=8,  # More neurons to start
            learning_rate=0.1,  # Higher learning rate for faster convergence
            device_config=device_config
        )
        
        # More lenient thresholds for learning
        network.reward_function.thresholds['excellent'] = 0.10  # Within 10%
        network.reward_function.thresholds['good'] = 0.20       # Within 20%
        network.reward_function.thresholds['poor'] = 0.40       # Within 40%
    else:
        print(f"Resuming with existing network: {len(network.layers)} layers, {sum(l.get_neuron_count() for l in network.layers)} neurons")
    network.reward_function.thresholds['bad'] = 0.60        # Within 60%
    
    # Generate training data
    n_train = ADDITION_TRAIN_SAMPLES
    X_train, y_train = generate_addition_data(n_train, max_digit=ADDITION_MAX_DIGIT)
    
    # Generate test data
    n_test = ADDITION_TEST_SAMPLES
    X_test, y_test = generate_addition_data(n_test, max_digit=ADDITION_MAX_DIGIT)
    
    print(f"\nTraining on {n_train} addition problems")
    print(f"Sample problems:")
    for i in range(5):
        print(f"  [{int(X_train[i, 0])}, {int(X_train[i, 1])}, {int(X_train[i, 2])}, {int(X_train[i, 3])}] -> {int(y_train[i, 0])} ({int(X_train[i, 0])} + {int(X_train[i, 1])})")
    print()
    
    # Training loop with adaptive sampling
    n_epochs = ADDITION_EPOCHS
    base_samples_per_epoch = BASE_SAMPLES_PER_EPOCH
    min_samples = MIN_SAMPLES_PER_EPOCH
    max_samples = MAX_SAMPLES_PER_EPOCH
    
    epoch_rewards = []
    epoch_errors = []
    network_sizes = []
    layer_counts = []
    samples_used = []
    
    print("Training Progress:")
    print("-"*70)
    print("Using adaptive sampling: more samples when struggling, fewer when doing well")
    print("-"*70)
    
    for epoch in range(n_epochs):
        indices = np.random.permutation(n_train)
        
        # Adaptive sampling based on recent performance
        avg_reward_recent = network.reward_function.get_average_reward(50)
        
        if epoch == 0:
            samples_per_epoch = base_samples_per_epoch
        elif avg_reward_recent < -5.0:  # Struggling badly
            samples_per_epoch = min(max_samples, int(base_samples_per_epoch * 2.5))
        elif avg_reward_recent < 0:  # Struggling
            samples_per_epoch = min(max_samples, int(base_samples_per_epoch * 1.5))
        elif avg_reward_recent > 7.0:  # Doing excellently
            samples_per_epoch = max(min_samples, int(base_samples_per_epoch * 0.5))
        elif avg_reward_recent > 3.0:  # Doing well
            samples_per_epoch = max(min_samples, int(base_samples_per_epoch * 0.75))
        else:  # Average performance
            samples_per_epoch = base_samples_per_epoch
        
        samples_used.append(samples_per_epoch)
        
        epoch_reward_sum = 0
        
        for i in range(samples_per_epoch):
            idx = indices[i % n_train]
            x = X_train[idx]
            target = y_train[idx]
            
            reward = network.train_step(x, target)
            epoch_reward_sum += reward
        
        # Epoch statistics
        avg_epoch_reward = epoch_reward_sum / samples_per_epoch
        avg_reward_50 = network.reward_function.get_average_reward(50)
        avg_error_50 = network.reward_function.get_average_error(50)
        
        epoch_rewards.append(avg_reward_50)
        epoch_errors.append(avg_error_50)
        
        stats = network.get_network_stats()
        network_sizes.append(stats['total_neurons'])
        layer_counts.append(stats['total_layers'])
        
        # Print every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d}: Reward={avg_reward_50:6.2f}, "
                  f"Error={avg_error_50:.4f}, "
                  f"Samples={samples_per_epoch:3d}, "
                  f"Layers={stats['total_layers']}, Neurons={stats['total_neurons']}")
        
        # Auto-save every 100 epochs
        if (epoch + 1) % 100 == 0:
            save_network(network)
    
    print("\n" + "="*70)
    print("LEVEL 1 TRAINING COMPLETE!")
    print("="*70)
    
    # Test the network
    print("\nTesting on 20 new addition problems:")
    print("-"*70)
    
    test_correct = 0
    test_close = 0
    
    for i in range(min(20, n_test)):
        x = X_test[i]
        target = y_test[i]
        prediction = network.predict(x)
        
        actual = int(target[0])
        pred = prediction[0]
        error_pct = abs(pred - actual) / max(actual, 1) * 100
        
        # Count as correct if within 10% or rounds to correct answer
        is_correct = abs(round(pred) - actual) == 0
        is_close = error_pct < 10
        
        if is_correct:
            test_correct += 1
            status = "✓ CORRECT"
        elif is_close:
            test_close += 1
            status = "~ CLOSE"
        else:
            status = "✗ WRONG"
        
        print(f"  [{int(x[0])}, {int(x[1])}, {int(x[2])}, {int(x[3])}] -> {actual} | "
              f"Predicted: {pred:.2f} (rounded: {round(pred)}) | {status}")
    
    accuracy = (test_correct / 20) * 100
    close_accuracy = ((test_correct + test_close) / 20) * 100
    
    print("-"*70)
    print(f"Exact Accuracy: {test_correct}/20 ({accuracy:.1f}%)")
    print(f"Close Accuracy: {test_correct + test_close}/20 ({close_accuracy:.1f}%)")
    print("="*70)
    
    # Show final network stats
    network.print_network_summary()
    
    return network, epoch_rewards, epoch_errors, network_sizes, layer_counts


def train_level_2_division(network):
    """Level 2: Train network to divide single digit numbers using the same network from Level 1."""
    print("\n" + "="*70)
    print("LEVEL 2: DIVISION OF SINGLE DIGIT NUMBERS")
    print("="*70)
    print("Task: Learn to divide two single digit numbers (1-9)")
    print("Example: [0, 0, 8, 4] -> 2, [0, 0, 9, 3] -> 3, [0, 0, 7, 4] -> 1.75")
    print("Format: [0, 0, dividend, divisor] where first two zeros signal division")
    print("-"*70)
    
    print("\nContinuing with the same network from Level 1...")
    print(f"Starting with {len(network.layers)} layers and "
          f"{sum(l.get_neuron_count() for l in network.layers)} neurons")
    print("The network must learn to distinguish between:")
    print("  - Addition: [num1, num2, 0, 0] -> num1 + num2")
    print("  - Division: [0, 0, dividend, divisor] -> dividend / divisor")
    
    # Adjust thresholds for division task
    network.reward_function.thresholds['excellent'] = 0.05
    network.reward_function.thresholds['good'] = 0.10
    network.reward_function.thresholds['poor'] = 0.20
    network.reward_function.thresholds['bad'] = 0.35
    
    # Reset reward statistics for new task
    network.reward_function.reset_statistics()
    
    # Generate training data
    n_train = DIVISION_TRAIN_SAMPLES
    X_train, y_train = generate_division_data(n_train, max_digit=DIVISION_MAX_DIGIT)
    
    # Generate test data
    n_test = DIVISION_TEST_SAMPLES
    X_test, y_test = generate_division_data(n_test, max_digit=DIVISION_MAX_DIGIT)
    
    print(f"\nTraining on {n_train} division problems")
    print(f"Sample problems:")
    for i in range(5):
        print(f"  [{int(X_train[i, 0])}, {int(X_train[i, 1])}, {int(X_train[i, 2])}, {int(X_train[i, 3])}] -> {y_train[i, 0]:.3f} ({int(X_train[i, 2])} / {int(X_train[i, 3])})")
    print()
    
    # Training loop with adaptive sampling
    n_epochs = DIVISION_EPOCHS
    base_samples_per_epoch = BASE_SAMPLES_PER_EPOCH
    min_samples = MIN_SAMPLES_PER_EPOCH
    max_samples = MAX_SAMPLES_PER_EPOCH
    
    epoch_rewards = []
    epoch_errors = []
    network_sizes = []
    layer_counts = []
    samples_used = []
    
    print("Training Progress:")
    print("-"*70)
    print("Using adaptive sampling: more samples when struggling, fewer when doing well")
    print("-"*70)
    
    initial_steps = network.training_steps
    
    for epoch in range(n_epochs):
        indices = np.random.permutation(n_train)
        
        # Adaptive sampling based on recent performance
        avg_reward_recent = network.reward_function.get_average_reward(50)
        
        if epoch == 0:
            samples_per_epoch = base_samples_per_epoch
        elif avg_reward_recent < -5.0:  # Struggling badly
            samples_per_epoch = min(max_samples, int(base_samples_per_epoch * 2.5))
        elif avg_reward_recent < 0:  # Struggling
            samples_per_epoch = min(max_samples, int(base_samples_per_epoch * 1.5))
        elif avg_reward_recent > 7.0:  # Doing excellently
            samples_per_epoch = max(min_samples, int(base_samples_per_epoch * 0.5))
        elif avg_reward_recent > 3.0:  # Doing well
            samples_per_epoch = max(min_samples, int(base_samples_per_epoch * 0.75))
        else:  # Average performance
            samples_per_epoch = base_samples_per_epoch
        
        samples_used.append(samples_per_epoch)
        epoch_reward_sum = 0
        
        for i in range(samples_per_epoch):
            idx = indices[i % n_train]
            x = X_train[idx]
            target = y_train[idx]
            
            reward = network.train_step(x, target)
            epoch_reward_sum += reward
        
        # Epoch statistics
        avg_epoch_reward = epoch_reward_sum / samples_per_epoch
        avg_reward_50 = network.reward_function.get_average_reward(50)
        avg_error_50 = network.reward_function.get_average_error(50)
        
        epoch_rewards.append(avg_reward_50)
        epoch_errors.append(avg_error_50)
        
        stats = network.get_network_stats()
        network_sizes.append(stats['total_neurons'])
        layer_counts.append(stats['total_layers'])
        
        # Print every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d}: Reward={avg_reward_50:6.2f}, "
                  f"Error={avg_error_50:.4f}, "
                  f"Samples={samples_per_epoch:3d}, "
                  f"Layers={stats['total_layers']}, Neurons={stats['total_neurons']}")
        
        # Auto-save every 100 epochs
        if (epoch + 1) % 100 == 0:
            save_network(network)
    
    print("\n" + "="*70)
    print("LEVEL 2 TRAINING COMPLETE!")
    print("="*70)
    
    # Test the network
    print("\nTesting on 20 new division problems:")
    print("-"*70)
    
    test_correct = 0
    test_close = 0
    
    for i in range(min(20, n_test)):
        x = X_test[i]
        target = y_test[i]
        prediction = network.predict(x)
        
        actual = target[0]
        pred = prediction[0]
        error_pct = abs(pred - actual) / max(actual, 0.01) * 100
        
        # Count as close if within 15% (division is harder)
        is_very_close = error_pct < 5
        is_close = error_pct < 15
        
        if is_very_close:
            test_correct += 1
            status = "✓ EXCELLENT"
        elif is_close:
            test_close += 1
            status = "~ CLOSE"
        else:
            status = "✗ WRONG"
        
        print(f"  [{int(x[0])}, {int(x[1])}, {int(x[2])}, {int(x[3])}] -> {actual:.3f} | "
              f"Predicted: {pred:.3f} | Error: {error_pct:.1f}% | {status}")
    
    accuracy = (test_correct / 20) * 100
    close_accuracy = ((test_correct + test_close) / 20) * 100
    
    print("-"*70)
    print(f"Excellent Accuracy (<5% error): {test_correct}/20 ({accuracy:.1f}%)")
    print(f"Close Accuracy (<15% error): {test_correct + test_close}/20 ({close_accuracy:.1f}%)")
    print("="*70)
    
    # Show final network stats
    network.print_network_summary()
    
    return network, epoch_rewards, epoch_errors, network_sizes, layer_counts


def visualize_multi_level_training(level1_data, level2_data, network=None):
    """Visualize training progress across both levels with modification tracking."""
    l1_rewards, l1_errors, l1_sizes, l1_layers = level1_data
    l2_rewards, l2_errors, l2_sizes, l2_layers = level2_data
    
    # Create larger figure with more subplots for modification tracking
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Self-Modifying Network: Multi-Level Learning with Intelligent Modifications', 
                 fontsize=16, fontweight='bold')
    
    epochs_l1 = range(1, len(l1_rewards) + 1)
    epochs_l2 = range(len(l1_rewards) + 1, len(l1_rewards) + len(l2_rewards) + 1)
    
    # Plot 1: Rewards over time
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs_l1, l1_rewards, 'b-', linewidth=2, label='Level 1 (Addition)')
    ax1.plot(epochs_l2, l2_rewards, 'r-', linewidth=2, label='Level 2 (Division)')
    ax1.axvline(x=len(l1_rewards), color='gray', linestyle='--', alpha=0.5, label='Task Switch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Reward Progress Across Levels')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Error over time (Loss Curve)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs_l1, l1_errors, 'b-', linewidth=2, label='Level 1 (Addition)')
    ax2.plot(epochs_l2, l2_errors, 'r-', linewidth=2, label='Level 2 (Division)')
    ax2.axvline(x=len(l1_rewards), color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Average Error (MSE)')
    ax2.set_title('Loss Curves Across Levels')
    ax2.set_yscale('log')  # Log scale for better visualization
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Network size over time
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(epochs_l1, l1_sizes, 'b-', linewidth=2, label='Level 1 (Addition)')
    ax3.plot(epochs_l2, l2_sizes, 'r-', linewidth=2, label='Level 2 (Division)')
    ax3.axvline(x=len(l1_rewards), color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Total Neurons')
    ax3.set_title('Network Size Evolution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Layer count over time
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(epochs_l1, l1_layers, 'b-', linewidth=2, marker='o', markersize=4, label='Level 1')
    ax4.plot(epochs_l2, l2_layers, 'r-', linewidth=2, marker='o', markersize=4, label='Level 2')
    ax4.axvline(x=len(l1_rewards), color='gray', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Number of Layers')
    ax4.set_title('Layer Count Evolution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Modification Timeline (if network provided)
    ax5 = fig.add_subplot(gs[1, 1])
    if network is not None and hasattr(network, 'modification_tracker'):
        mod_history = network.modification_tracker.get_recent_modifications(100)
        if len(mod_history) > 0:
            # Color map for modification types
            color_map = {
                'add_neuron': 'blue',
                'remove_neuron': 'red',
                'add_layer': 'green',
                'remove_layer': 'orange',
                'rewire_connections': 'purple',
                'adjust_thresholds': 'brown',
                'none': 'gray'
            }
            
            for record in mod_history:
                mod_type = record.modification_type.value
                color = color_map.get(mod_type, 'gray')
                marker = 'x' if record.rolled_back else 'o'
                alpha = 0.3 if record.rolled_back else 0.8
                
                ax5.scatter(record.timestamp, record.reward_improvement, 
                           c=color, marker=marker, s=50, alpha=alpha,
                           label=mod_type if mod_type not in ax5.get_legend_handles_labels()[1] else "")
            
            ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax5.set_xlabel('Training Step')
            ax5.set_ylabel('Reward Improvement')
            ax5.set_title('Modification Timeline (x=rolled back)')
            ax5.legend(loc='best', fontsize=8)
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'No modifications yet', ha='center', va='center')
            ax5.set_title('Modification Timeline')
    else:
        ax5.text(0.5, 0.5, 'Network not provided', ha='center', va='center')
        ax5.set_title('Modification Timeline')
    
    # Plot 6: Strategy Distribution (Pie Chart)
    ax6 = fig.add_subplot(gs[1, 2])
    if network is not None and hasattr(network, 'modification_tracker'):
        stats = network.modification_tracker.get_statistics()
        if 'success_rate_by_type' in stats and len(stats['success_rate_by_type']) > 0:
            types = list(stats['success_rate_by_type'].keys())
            # Get attempt counts
            attempts = [network.modification_tracker.attempt_count_by_type.get(
                next(mt for mt in network.modification_tracker.attempt_count_by_type.keys() 
                     if mt.value == t), 0) for t in types]
            
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'gray']
            ax6.pie(attempts, labels=types, autopct='%1.1f%%', colors=colors[:len(types)])
            ax6.set_title('Modification Strategy Distribution')
        else:
            ax6.text(0.5, 0.5, 'No modifications yet', ha='center', va='center')
            ax6.set_title('Modification Strategy Distribution')
    else:
        ax6.text(0.5, 0.5, 'Network not provided', ha='center', va='center')
        ax6.set_title('Modification Strategy Distribution')
    
    # Plot 7: Meta-Learner Accuracy
    ax7 = fig.add_subplot(gs[2, 0])
    if network is not None and hasattr(network, 'meta_learner'):
        ml_stats = network.meta_learner.get_statistics()
        if ml_stats['total_updates'] > 0 and len(network.meta_learner.training_accuracies) > 0:
            ax7.plot(network.meta_learner.training_accuracies, 'g-', linewidth=1, alpha=0.3)
            # Plot moving average
            window = min(20, len(network.meta_learner.training_accuracies))
            if window > 0:
                moving_avg = np.convolve(network.meta_learner.training_accuracies, 
                                        np.ones(window)/window, mode='valid')
                ax7.plot(range(window-1, len(network.meta_learner.training_accuracies)), 
                        moving_avg, 'g-', linewidth=2, label=f'MA({window})')
            ax7.axhline(y=0.5, color='red', linestyle='--', linewidth=1, label='Random')
            ax7.set_xlabel('Training Batch')
            ax7.set_ylabel('Accuracy')
            ax7.set_title('Meta-Learner Training Accuracy')
            ax7.legend()
            ax7.grid(True, alpha=0.3)
            ax7.set_ylim([0, 1])
        else:
            ax7.text(0.5, 0.5, 'No meta-learner training yet', ha='center', va='center')
            ax7.set_title('Meta-Learner Training Accuracy')
    else:
        ax7.text(0.5, 0.5, 'Network not provided', ha='center', va='center')
        ax7.set_title('Meta-Learner Training Accuracy')
    
    # Plot 8: Performance comparison
    ax8 = fig.add_subplot(gs[2, 1])
    l1_final_reward = l1_rewards[-1] if l1_rewards else 0
    l2_final_reward = l2_rewards[-1] if l2_rewards else 0
    
    ax8.bar(['Level 1\n(Addition)', 'Level 2\n(Division)'], 
            [l1_final_reward, l2_final_reward],
            color=['blue', 'red'], alpha=0.7)
    ax8.set_ylabel('Final Average Reward')
    ax8.set_title('Final Performance Comparison')
    ax8.grid(True, alpha=0.3, axis='y')
    
    # Plot 9: Modification Success Rates
    ax9 = fig.add_subplot(gs[2, 2])
    if network is not None and hasattr(network, 'modification_tracker'):
        stats = network.modification_tracker.get_statistics()
        if 'success_rate_by_type' in stats and len(stats['success_rate_by_type']) > 0:
            types = list(stats['success_rate_by_type'].keys())
            rates = [stats['success_rate_by_type'][t] * 100 for t in types]
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'gray']
            
            ax9.barh(types, rates, color=colors[:len(types)], alpha=0.7)
            ax9.axvline(x=50, color='gray', linestyle='--', linewidth=1, label='Random')
            ax9.set_xlabel('Success Rate (%)')
            ax9.set_title('Modification Success Rates')
            ax9.set_xlim([0, 100])
            ax9.legend()
            ax9.grid(True, alpha=0.3, axis='x')
        else:
            ax9.text(0.5, 0.5, 'No modifications yet', ha='center', va='center')
            ax9.set_title('Modification Success Rates')
    else:
        ax9.text(0.5, 0.5, 'Network not provided', ha='center', va='center')
        ax9.set_title('Modification Success Rates')
    
    plt.savefig('multi_level_training_results.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved as 'multi_level_training_results.png'")
    plt.show()


if __name__ == "__main__":
    # Parse command-line arguments
    use_existing_brain = True
    if "--new" in sys.argv:
        use_existing_brain = False
        print("\n⚠ Starting with NEW brain (ignoring saved checkpoint)")
    
    print("\n" + "="*70)
    print("SELF-MODIFYING NEURAL NETWORK - CONTINUOUS LEARNING")
    print("="*70)
    print("Like a human brain, this network continuously learns and improves")
    print("Training progress is automatically saved and restored")
    print("Use --new flag to start fresh")
    print("="*70)
    
    # Try to load existing network
    network = None
    if use_existing_brain:
        network = load_network()
        if network is None:
            print("No saved network found, starting fresh...")
    
    try:
        # Level 1: Addition
        network, l1_rewards, l1_errors, l1_sizes, l1_layers = train_level_1_addition(network)
        
        # Save checkpoint after Level 1
        save_network(network)
        
        # Level 2: Division (using the SAME network - it must learn both operations)
        network, l2_rewards, l2_errors, l2_sizes, l2_layers = train_level_2_division(network)
        
        # Save final checkpoint
        save_network(network)
        
        # Visualize results with network for modification tracking
        visualize_multi_level_training(
            (l1_rewards, l1_errors, l1_sizes, l1_layers),
            (l2_rewards, l2_errors, l2_sizes, l2_layers),
            network=network
        )
        
        # Print modification tracker statistics
        if hasattr(network, 'modification_tracker'):
            network.modification_tracker.print_statistics()
        
        print("\n" + "="*70)
        print("TRAINING SESSION COMPLETE!")
        print("="*70)
        print("\nKey Observations:")
        print("1. Network structure adapted between addition and division tasks")
        print("2. Same network learned two different mathematical operations")
        print("3. Tag-based routing allowed neurons to specialize for each operation")
        print("4. Network size self-adjusted based on task complexity")
        print(f"5. Progress saved - next run will continue from {network.training_steps} steps")
        print("="*70)
    
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
        if network is not None:
            save_network(network)
            print("Network state saved before exit")
        sys.exit(0)
