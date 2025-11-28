import numpy as np
import sys
import os
import pickle
from classes.network import SelfModifyingNetwork
from setup.gpu_config import get_device_config

# Custom unpickler to handle module path changes
class ModuleRenameUnpickler(pickle.Unpickler):
    """Unpickler that remaps old module names to new classes. prefix"""
    def find_class(self, module, name):
        # Remap old direct imports to classes. package
        if module in ['network', 'layer', 'neuron', 'reward', 'backward', 
                      'modification_tracker', 'meta_learner']:
            module = f'classes.{module}'
        # Also handle old 'class.' prefix
        elif module.startswith('class.'):
            module = module.replace('class.', 'classes.')
        # Handle gpu_config moved to setup folder
        elif module == 'gpu_config':
            module = 'setup.gpu_config'
        return super().find_class(module, name)


# ============================================================================
# CONFIGURATION
# ============================================================================

CHECKPOINT_DIR = "checkpoints"
NETWORK_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "network_state.pkl")

SAMPLES_PER_BATCH = 50  # Train on this many samples before stats update
SAVE_INTERVAL_BATCHES = 100  # Auto-save every N batches
MAX_DIGIT = 3  # Maximum digit for operations
PRINT_INTERVAL = 10  # Print stats every N batches


# ============================================================================
# SAVE/LOAD FUNCTIONS
# ============================================================================

def save_network(network, filepath=NETWORK_CHECKPOINT):
    """Save the entire network state to disk."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    state = {
        'network': network,
        'training_steps': network.training_steps,
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(state, f)
    print(f"  💾 Network saved (steps: {network.training_steps})")

def load_network(filepath=NETWORK_CHECKPOINT):
    """Load network state from disk."""
    if not os.path.exists(filepath):
        return None
    
    try:
        with open(filepath, 'rb') as f:
            # Use custom unpickler to handle module path changes
            unpickler = ModuleRenameUnpickler(f)
            state = unpickler.load()
        print(f"  ✓ Network loaded from checkpoint (steps: {state['training_steps']})")
        return state['network']
    except Exception as e:
        print(f"  ✗ Failed to load network: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# DATA GENERATION
# ============================================================================

def generate_random_sample(max_digit=9):
    """Generate a random math problem (either addition or division)."""
    task_type = np.random.choice([0, 1])  # 0=addition, 1=division
    
    if task_type == 0:  # Addition
        num1 = np.random.randint(0, max_digit + 1)
        num2 = np.random.randint(0, max_digit + 1)
        x = np.array([num1, num2, 0, 0], dtype=float)
        y = np.array([num1 + num2], dtype=float)
    else:  # Division
        dividend = np.random.randint(1, max_digit + 1)
        divisor = np.random.randint(1, max_digit + 1)
        x = np.array([0, 0, dividend, divisor], dtype=float)
        y = np.array([dividend / divisor], dtype=float)
    
    return x, y


# ============================================================================
# CONTINUOUS TRAINING LOOP
# ============================================================================

class ContinuousTrainer:
    """Manages continuous training in background thread."""
    
    def __init__(self, network):
        self.network = network
        self.running = False
        self.batch_count = 0
        self.total_samples = 0
        
        # Statistics
        self.recent_rewards = []
        self.recent_errors = []
        
    def train_batch(self, batch_size=SAMPLES_PER_BATCH):
        """Train on a batch of random samples."""
        batch_rewards = []
        
        for _ in range(batch_size):
            x, y = generate_random_sample(max_digit=MAX_DIGIT)
            reward = self.network.train_step(x, y)
            batch_rewards.append(reward)
            self.total_samples += 1
        
        self.batch_count += 1
        
        # Update statistics
        avg_reward = self.network.reward_function.get_average_reward(50)
        avg_error = self.network.reward_function.get_average_error(50)
        self.recent_rewards.append(avg_reward)
        self.recent_errors.append(avg_error)
        
        return avg_reward, avg_error
    
    def print_status(self):
        """Print current training status."""
        stats = self.network.get_network_stats()
        avg_reward = self.network.reward_function.get_average_reward(50)
        avg_error = self.network.reward_function.get_average_error(50)
        
        # Build layer neuron count string
        layer_info = ", ".join([f"L{i}: {layer['total_neurons']}" 
                                for i, layer in enumerate(stats['layer_stats'])])
        
        print(f"[Batch {self.batch_count:5d}] "
              f"Steps={self.network.training_steps:6d} | "
              f"Reward={avg_reward:+.2f} | "
              f"Error={avg_error:.4f} | "
              f"Layers={stats['total_layers']} | "
              f"Neurons={stats['total_neurons']} | "
              f"{layer_info}")
    
    def run_continuous(self):
        """Main continuous training loop."""
        print("\n" + "="*70)
        print("CONTINUOUS TRAINING MODE - Press Ctrl+C to stop")
        print("="*70)
        
        self.running = True
        
        try:
            while self.running:
                # Train one batch
                avg_reward, avg_error = self.train_batch()
                
                # Print status
                if self.batch_count % PRINT_INTERVAL == 0:
                    self.print_status()
                
                # Auto-save
                if self.batch_count % SAVE_INTERVAL_BATCHES == 0:
                    save_network(self.network)
        
        except KeyboardInterrupt:
            print("\n\n⚠ Training stopped by user")
            self.running = False
            save_network(self.network)
    
    def test_network(self, n_tests=20):
        """Test the network on random problems."""
        print("\n" + "="*70)
        print(f"TESTING NETWORK (after {self.network.training_steps} training steps)")
        print("="*70)
        
        addition_correct = 0
        division_correct = 0
        
        print("\nAddition Tests:")
        for i in range(n_tests // 2):
            num1 = np.random.randint(0, MAX_DIGIT + 1)
            num2 = np.random.randint(0, MAX_DIGIT + 1)
            x = np.array([num1, num2, 0, 0], dtype=float)
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
            x = np.array([0, 0, dividend, divisor], dtype=float)
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
    # Parse command-line arguments
    use_existing_brain = True
    if "--new" in sys.argv:
        use_existing_brain = False
        print("\n⚠ Starting with NEW brain (ignoring saved checkpoint)")
    
    if "--test" in sys.argv:
        # Load network and test only
        network = load_network()
        if network is None:
            print("No saved network found!")
            sys.exit(1)
        
        trainer = ContinuousTrainer(network)
        trainer.test_network(n_tests=20)
        sys.exit(0)
    
    print("\n" + "="*70)
    print("SELF-MODIFYING NEURAL NETWORK - CONTINUOUS LEARNING")
    print("="*70)
    print("Like a human brain, this network continuously learns")
    print("Training runs forever until you stop it (Ctrl+C)")
    print("Progress is auto-saved every", SAVE_INTERVAL_BATCHES, "batches")
    print("\nUsage:")
    print("  python continuous_train.py         - Continue training")
    print("  python continuous_train.py --new   - Start fresh")
    print("  python continuous_train.py --test  - Test only (no training)")
    print("="*70)
    
    # Try to load existing network
    network = None
    if use_existing_brain:
        network = load_network()
    
    # Create new network if needed
    if network is None:
        print("\n  Creating new network...")
        # Initialize GPU/CPU device
        device_config = get_device_config(force_cpu=False)
        
        network = SelfModifyingNetwork(
            input_dim=4,
            output_dim=1,
            initial_layers=2,
            initial_neurons_per_layer=8,
            learning_rate=0.1,
            device_config=device_config
        )
        
        # Configure reward thresholds
        network.reward_function.thresholds['excellent'] = 0.10
        network.reward_function.thresholds['good'] = 0.20
        network.reward_function.thresholds['poor'] = 0.40
        network.reward_function.thresholds['bad'] = 0.60
    
    # Start continuous training
    trainer = ContinuousTrainer(network)
    trainer.run_continuous()
    
    # Test after training stops
    print("\n")
    trainer.test_network(n_tests=20)
    
    # Print final stats
    if hasattr(network, 'modification_tracker'):
        network.modification_tracker.print_statistics()
