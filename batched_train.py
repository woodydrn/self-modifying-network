"""
Batched GPU Training for Self-Modifying Network

This script uses the BatchedSelfModifyingNetwork which processes multiple
samples simultaneously for efficient GPU utilization.
"""

import numpy as np
import sys
import os
import pickle
import time
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter

from classes.batched_network import BatchedSelfModifyingNetwork
from setup.gpu_config import get_device_config

# ============================================================================
# CONFIGURATION
# ============================================================================

CHECKPOINT_DIR = "checkpoints"
NETWORK_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "batched_network_state.pkl")
TENSORBOARD_LOG_DIR = "runs"

BATCH_SIZE = 256  # Process this many samples at once (GPU sweet spot)
BATCHES_PER_EPOCH = 50  # Train this many batches before printing stats
SAVE_INTERVAL_EPOCHS = 10  # Auto-save every N epochs
MAX_DIGIT = 3  # Maximum digit for operations

# Plateau detection
PLATEAU_WINDOW = 50
PLATEAU_THRESHOLD = 0.005
STUCK_THRESHOLD = 3


# ============================================================================
# SAVE/LOAD FUNCTIONS
# ============================================================================

def save_network(network, filepath=NETWORK_CHECKPOINT):
    """Save the network state to disk."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Save network state (move to CPU for portability)
    state = {
        'training_steps': network.training_steps,
        'batch_count': network.batch_count,
        'layer_count': len(network.layers),
        'layer_configs': [],
        'reward_function': network.reward_function,
        'modification_tracker': network.modification_tracker,
        'meta_learner': network.meta_learner,
    }
    
    # Save each layer's weights
    for layer in network.layers:
        layer_state = {
            'neuron_count': layer.neuron_count,
            'input_dim': layer.input_dim,
            'output_dim': layer.output_dim,
            'weights': layer.weights.data.cpu().numpy(),
            'bias': layer.bias.data.cpu().numpy(),
            'functional_tags': layer.functional_tags.cpu().numpy(),
            'activation_threshold': layer.activation_threshold,
            'is_output_layer': layer.is_output_layer,
        }
        state['layer_configs'].append(layer_state)
    
    with open(filepath, 'wb') as f:
        pickle.dump(state, f)
    print(f"  💾 Network saved (steps: {network.training_steps})")


def load_network(filepath=NETWORK_CHECKPOINT, device=None):
    """Load network state from disk."""
    if not os.path.exists(filepath):
        return None
    
    try:
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        # Recreate network
        first_layer = state['layer_configs'][0]
        last_layer = state['layer_configs'][-1]
        
        network = BatchedSelfModifyingNetwork(
            input_dim=first_layer['input_dim'],
            output_dim=last_layer['output_dim'],
            initial_layers=1,  # Will be replaced
            initial_neurons_per_layer=1,
            device=device
        )
        
        # Restore layers
        network.layers = []
        for layer_state in state['layer_configs']:
            from classes.batched_network import BatchedLayer
            layer = BatchedLayer(
                neuron_count=layer_state['neuron_count'],
                input_dim=layer_state['input_dim'],
                output_dim=layer_state['output_dim'],
                device=device,
                is_output_layer=layer_state['is_output_layer']
            )
            
            # Restore weights
            layer.weights = torch.nn.Parameter(
                torch.tensor(layer_state['weights'], device=device)
            )
            layer.bias = torch.nn.Parameter(
                torch.tensor(layer_state['bias'], device=device)
            )
            layer.functional_tags = torch.tensor(
                layer_state['functional_tags'], device=device
            )
            layer.activation_threshold = layer_state['activation_threshold']
            
            network.layers.append(layer)
        
        # Restore state
        network.training_steps = state['training_steps']
        network.batch_count = state['batch_count']
        network.reward_function = state['reward_function']
        network.modification_tracker = state['modification_tracker']
        network.meta_learner = state['meta_learner']
        
        print(f"  ✓ Network loaded (steps: {network.training_steps})")
        return network
    except Exception as e:
        print(f"  ✗ Failed to load network: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# DATA GENERATION
# ============================================================================

def generate_batch(batch_size, max_digit=9):
    """Generate a batch of random math problems."""
    x_batch = np.zeros((batch_size, 4), dtype=np.float32)
    y_batch = np.zeros((batch_size, 1), dtype=np.float32)
    
    for i in range(batch_size):
        task_type = np.random.choice([0, 1])  # 0=addition, 1=division
        
        if task_type == 0:  # Addition
            num1 = np.random.randint(0, max_digit + 1)
            num2 = np.random.randint(0, max_digit + 1)
            x_batch[i] = [num1, num2, 0, 0]
            y_batch[i] = num1 + num2
        else:  # Division
            dividend = np.random.randint(1, max_digit + 1)
            divisor = np.random.randint(1, max_digit + 1)
            x_batch[i] = [0, 0, dividend, divisor]
            y_batch[i] = dividend / divisor
    
    return x_batch, y_batch


# ============================================================================
# BATCHED TRAINER
# ============================================================================

class BatchedTrainer:
    """Manages batched training with GPU."""
    
    def __init__(self, network, use_tensorboard=True):
        self.network = network
        self.running = False
        self.epoch_count = 0
        
        # Statistics
        self.recent_rewards = []
        self.recent_errors = []
        self.reward_history = []
        self.plateau_counter = 0
        self.is_stuck = False
        
        # Timing
        self.samples_per_second = 0
        
        # TensorBoard
        self.use_tensorboard = use_tensorboard
        if use_tensorboard:
            run_name = f"batched_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            log_path = os.path.join(TENSORBOARD_LOG_DIR, run_name)
            self.writer = SummaryWriter(log_path)
            print(f"  📊 TensorBoard logging to: {log_path}")
        else:
            self.writer = None
    
    def train_epoch(self, batches_per_epoch=BATCHES_PER_EPOCH, batch_size=BATCH_SIZE):
        """Train for one epoch (multiple batches)."""
        epoch_rewards = []
        epoch_errors = []
        start_time = time.time()
        
        for _ in range(batches_per_epoch):
            x_batch, y_batch = generate_batch(batch_size, max_digit=MAX_DIGIT)
            avg_reward, avg_error = self.network.train_batch(x_batch, y_batch)
            epoch_rewards.append(avg_reward)
            epoch_errors.append(avg_error)
        
        elapsed = time.time() - start_time
        samples_processed = batches_per_epoch * batch_size
        self.samples_per_second = samples_processed / elapsed
        
        self.epoch_count += 1
        
        # Update statistics
        avg_reward = np.mean(epoch_rewards)
        avg_error = np.mean(epoch_errors)
        self.recent_rewards.append(avg_reward)
        self.recent_errors.append(avg_error)
        self.reward_history.append(avg_reward)
        
        # Keep history bounded
        if len(self.reward_history) > PLATEAU_WINDOW * 2:
            self.reward_history = self.reward_history[-PLATEAU_WINDOW * 2:]
        
        # Detect plateau
        if len(self.reward_history) >= PLATEAU_WINDOW:
            variance = np.var(self.reward_history[-PLATEAU_WINDOW:])
            if variance < PLATEAU_THRESHOLD:
                self.plateau_counter += 1
                if self.plateau_counter >= STUCK_THRESHOLD and not self.is_stuck:
                    print(f"\n  ⚠️  PLATEAU DETECTED")
                    self.is_stuck = True
                    self.network.plateau_detected = True
            else:
                if self.is_stuck:
                    print(f"\n  ✓ Escaped plateau")
                self.plateau_counter = 0
                self.is_stuck = False
                self.network.plateau_detected = False
        
        # Log to TensorBoard
        if self.writer:
            step = self.network.training_steps
            self.writer.add_scalar('Training/Reward', avg_reward, step)
            self.writer.add_scalar('Training/Error', avg_error, step)
            self.writer.add_scalar('Training/Samples_Per_Second', self.samples_per_second, step)
            
            stats = self.network.get_network_stats()
            self.writer.add_scalar('Network/Total_Neurons', stats['total_neurons'], step)
            self.writer.add_scalar('Network/Total_Layers', stats['total_layers'], step)
            
            # Log meta-learner stats
            if hasattr(self.network, 'last_meta_stats') and self.network.last_meta_stats:
                meta = self.network.last_meta_stats
                self.writer.add_scalar('MetaLearner/Loss', meta['loss'], step)
                self.writer.add_scalar('MetaLearner/Accuracy', meta['accuracy'], step)
        
        return avg_reward, avg_error
    
    def print_status(self):
        """Print current training status."""
        stats = self.network.get_network_stats()
        avg_reward = stats['avg_reward_50']
        avg_error = stats['avg_error_50']
        
        layer_info = ", ".join([f"L{i}: {l['total_neurons']}" 
                               for i, l in enumerate(stats['layer_stats'])])
        
        output = (f"[Epoch {self.epoch_count:5d}] "
                 f"Steps={self.network.training_steps:7d} | "
                 f"Reward={avg_reward:+.2f} | "
                 f"Error={avg_error:.4f} | ")
        
        # Add meta-learner stats if available
        if hasattr(self.network, 'last_meta_stats') and self.network.last_meta_stats:
            meta = self.network.last_meta_stats
            output += f"Meta[L={meta['loss']:.3f} A={meta['accuracy']:.2f}] | "
        
        output += (f"Speed={self.samples_per_second:.0f}/s | "
                  f"Layers={stats['total_layers']} | "
                  f"Neurons={stats['total_neurons']} | "
                  f"{layer_info}")
        
        print(output)
    
    def run_continuous(self):
        """Main continuous training loop."""
        print("\n" + "="*70)
        print("BATCHED GPU TRAINING - Press Ctrl+C to stop")
        print(f"Batch size: {BATCH_SIZE} | Batches per epoch: {BATCHES_PER_EPOCH}")
        print(f"Device: {self.network.device}")
        print("="*70)
        
        self.running = True
        
        try:
            while self.running:
                # Train one epoch
                self.train_epoch()
                
                # Print status
                self.print_status()
                
                # Auto-save
                if self.epoch_count % SAVE_INTERVAL_EPOCHS == 0:
                    save_network(self.network)
        
        except KeyboardInterrupt:
            print("\n\n⚠ Training stopped by user")
            self.running = False
            save_network(self.network)
            if self.writer:
                self.writer.close()
    
    def test_network(self, n_tests=20):
        """Test the network."""
        print("\n" + "="*70)
        print(f"TESTING NETWORK (after {self.network.training_steps} steps)")
        print("="*70)
        
        addition_correct = 0
        division_correct = 0
        
        print("\nAddition Tests:")
        for i in range(n_tests // 2):
            num1 = np.random.randint(0, MAX_DIGIT + 1)
            num2 = np.random.randint(0, MAX_DIGIT + 1)
            x = np.array([num1, num2, 0, 0], dtype=np.float32)
            expected = num1 + num2
            
            pred = self.network.predict(x)[0]
            error = abs(pred - expected) / max(expected, 1) * 100
            
            status = "✓" if error < 10 else "✗"
            if error < 10:
                addition_correct += 1
            
            print(f"  {num1} + {num2} = {expected} | Predicted: {pred:.2f} | {status}")
        
        print("\nDivision Tests:")
        for i in range(n_tests // 2):
            dividend = np.random.randint(1, MAX_DIGIT + 1)
            divisor = np.random.randint(1, MAX_DIGIT + 1)
            x = np.array([0, 0, dividend, divisor], dtype=np.float32)
            expected = dividend / divisor
            
            pred = self.network.predict(x)[0]
            error = abs(pred - expected) / max(expected, 0.01) * 100
            
            status = "✓" if error < 15 else "✗"
            if error < 15:
                division_correct += 1
            
            print(f"  {dividend} / {divisor} = {expected:.2f} | Predicted: {pred:.2f} | {status}")
        
        print("\n" + "-"*70)
        print(f"Addition Accuracy: {addition_correct}/{n_tests//2} ({addition_correct*100/(n_tests//2):.1f}%)")
        print(f"Division Accuracy: {division_correct}/{n_tests//2} ({division_correct*100/(n_tests//2):.1f}%)")
        print("="*70 + "\n")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("BATCHED SELF-MODIFYING NETWORK - GPU OPTIMIZED")
    print("="*70)
    
    # Parse arguments
    use_existing = "--new" not in sys.argv
    test_only = "--test" in sys.argv
    force_cpu = "--cpu" in sys.argv
    
    if "--new" in sys.argv:
        print("⚠ Starting with NEW network")
    
    print("\nUsage:")
    print("  python batched_train.py         - Continue training")
    print("  python batched_train.py --new   - Start fresh")
    print("  python batched_train.py --test  - Test only")
    print("  python batched_train.py --cpu   - Force CPU mode")
    print("="*70)
    
    # Initialize device
    device_config = get_device_config(force_cpu=force_cpu)
    device = device_config.device
    
    # Load or create network
    network = None
    if use_existing:
        network = load_network(device=device)
    
    if network is None:
        print("\n  Creating new batched network...")
        network = BatchedSelfModifyingNetwork(
            input_dim=4,
            output_dim=1,
            initial_layers=3,
            initial_neurons_per_layer=16,
            learning_rate=0.05,
            device=device
        )
        
        # Configure reward thresholds
        network.reward_function.thresholds['excellent'] = 0.10
        network.reward_function.thresholds['good'] = 0.20
        network.reward_function.thresholds['poor'] = 0.40
        network.reward_function.thresholds['bad'] = 0.60
    
    # Create trainer
    trainer = BatchedTrainer(network)
    
    if test_only:
        trainer.test_network(n_tests=20)
    else:
        trainer.run_continuous()
        trainer.test_network(n_tests=20)
