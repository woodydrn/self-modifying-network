import numpy as np
from typing import List, Optional, Tuple, Dict
from .layer import AdaptiveLayer
from .reward import GradedRewardFunction
from .backward import IntelligentBackward
from .modification_tracker import ModificationTracker, ModificationType, ModificationRecord
from .meta_learner import MetaLearner

try:
    from setup.gpu_config import DeviceConfig
except ImportError:
    DeviceConfig = None


class SelfModifyingNetwork:
    """
    A self-modifying neural network that dynamically adjusts its structure.
    
    Features:
    - Tag-based selective activation
    - Dynamic layer and neuron addition/removal
    - Intelligent backpropagation with role assignment
    - Memory of poor configurations
    - Unified training/prediction loop
    """
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 initial_layers: int = 2,
                 initial_neurons_per_layer: int = 5,
                 tag_dim: int = 4,
                 learning_rate: float = 0.01,
                 device_config = None):
        """
        Initialize self-modifying network.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            initial_layers: Number of layers to start with
            initial_neurons_per_layer: Neurons per layer initially
            tag_dim: Dimension of tag vectors
            learning_rate: Base learning rate
            device_config: GPU/CPU device configuration (optional)
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.tag_dim = tag_dim
        self.learning_rate = learning_rate
        self.device_config = device_config
        
        # Network structure
        self.layers: List[AdaptiveLayer] = []
        self._initialize_layers(initial_layers, initial_neurons_per_layer)
        
        # Components
        self.reward_function = GradedRewardFunction()
        self.backward_engine = IntelligentBackward(tag_dim=tag_dim)
        
        # Meta-learning components
        self.modification_tracker = ModificationTracker(max_history=1000)
        self.meta_learner = MetaLearner(input_dim=15, hidden_dim=32, learning_rate=0.01)
        
        # Training state
        self.training_steps = 0
        self.last_reward = 0.0
        self.active_indices_history = []  # Track which neurons activated
        self.last_modification_step = 0
        self.network_snapshot = None  # For rollback
        self.pre_modification_reward = 0.0
        self.steps_since_modification = 0
        self.plateau_detected = False  # Set by trainer when stuck in local minimum
        self.last_meta_stats = None  # Last meta-learner training stats
        
        # Structure modification parameters
        self.min_layers = 1
        self.max_layers = 15  # Reduce max layers to prevent too deep networks
        self.min_neurons_per_layer = 2
        self.max_neurons_per_layer = 256  # Reduce max neurons per layer
        self.growth_threshold = -0.5  # Add structure if reward below this (scaled for -1/0/+1)
        self.pruning_threshold = 0.5  # Remove structure if reward above this (scaled for -1/0/+1)
        self.check_structure_every = 5000  # Wait much longer before modifying (was 300)
        
    def _save_network_snapshot(self) -> Dict:
        """Save current network state for potential rollback."""
        snapshot = {
            'layers': [],
            'training_steps': self.training_steps,
            'avg_reward': self.reward_function.get_average_reward(50)
        }
        
        for layer in self.layers:
            layer_state = {
                'neurons': [],
                'layer_tag': layer.layer_tag.copy()
            }
            for neuron in layer.neurons:
                neuron_state = {
                    'weights': neuron.weights.copy(),
                    'bias': neuron.bias.copy(),
                    'functional_tag': neuron.functional_tag.copy(),
                    'input_tags': [tag.copy() for tag in neuron.input_tags],
                    'output_tags': [tag.copy() for tag in neuron.output_tags],
                    'is_broadcast': neuron.is_broadcast,
                    'target_neuron_indices': neuron.target_neuron_indices.copy(),
                    'target_layer_id': neuron.target_layer_id
                }
                layer_state['neurons'].append(neuron_state)
            snapshot['layers'].append(layer_state)
        
        return snapshot
    
    def _restore_network_snapshot(self, snapshot: Dict):
        """Restore network from snapshot."""
        # Restore layers
        self.layers = []
        for layer_state in snapshot['layers']:
            # Reconstruct layer
            layer = AdaptiveLayer(
                initial_neuron_count=0,  # Will add neurons manually
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                tag_dim=self.tag_dim
            )
            layer.layer_tag = layer_state['layer_tag'].copy()
            layer.neurons = []
            
            for neuron_state in layer_state['neurons']:
                from .neuron import Neuron
                neuron = Neuron(
                    input_dim=neuron_state['weights'].shape[0],
                    output_dim=neuron_state['weights'].shape[1],
                    functional_tag=neuron_state['functional_tag'].copy(),
                    is_broadcast=neuron_state['is_broadcast']
                )
                neuron.weights = neuron_state['weights'].copy()
                neuron.bias = neuron_state['bias'].copy()
                neuron.input_tags = [tag.copy() for tag in neuron_state['input_tags']]
                neuron.output_tags = [tag.copy() for tag in neuron_state['output_tags']]
                neuron.target_neuron_indices = neuron_state['target_neuron_indices'].copy()
                neuron.target_layer_id = neuron_state['target_layer_id']
                layer.neurons.append(neuron)
            
            self.layers.append(layer)
    
    def _initialize_layers(self, num_layers: int, neurons_per_layer: int):
        """Initialize the network layers."""
        current_dim = self.input_dim
        
        for i in range(num_layers):
            # Last layer outputs to final dimension
            if i == num_layers - 1:
                layer_output_dim = self.output_dim
            else:
                layer_output_dim = current_dim  # Keep dimension for now
            
            layer = AdaptiveLayer(
                initial_neuron_count=neurons_per_layer,
                input_dim=current_dim,
                output_dim=layer_output_dim,
                tag_dim=self.tag_dim
            )
            self.layers.append(layer)
            
            # Mark output neurons if this is the last layer
            if i == num_layers - 1:
                for neuron in layer.neurons:
                    neuron.is_output_neuron = True
                    # Initialize output neuron bias to middle of expected range
                    neuron.bias = np.ones_like(neuron.bias) * 9.0  # Middle of 0-18 range
            
            current_dim = layer_output_dim
    
    def input_to_tag(self, x: np.ndarray) -> np.ndarray:
        """
        Convert input to a tag vector for routing.
        
        Args:
            x: Input vector
            
        Returns:
            Tag vector
        """
        # Simple approach: hash input to tag space using projection
        if not hasattr(self, '_tag_projection'):
            # Create random projection matrix
            self._tag_projection = np.random.randn(self.input_dim, self.tag_dim) * 0.1
        
        # Project and normalize
        tag = np.dot(x, self._tag_projection)
        tag = tag / (np.linalg.norm(tag) + 1e-8)
        
        return tag
    
    def forward(self, x: np.ndarray, input_tag: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Forward pass through the network.
        
        Args:
            x: Input vector
            input_tag: Optional pre-computed input tag
            
        Returns:
            Network output
        """
        # Convert input to tag
        if input_tag is None:
            input_tag = self.input_to_tag(x)
        
        # Track active neurons per layer
        self.active_indices_history = []
        
        # Forward through layers with selective routing
        current_input = x
        for i, layer in enumerate(self.layers):
            # Pass next layer for selective routing
            next_layer = self.layers[i + 1] if i < len(self.layers) - 1 else None
            output, active_indices = layer.forward(current_input, input_tag, next_layer)
            self.active_indices_history.append(active_indices)
            current_input = output
        
        return current_input
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Make a prediction (same as forward, for clarity).
        
        Args:
            x: Input vector
            
        Returns:
            Predicted output (clipped to reasonable range)
        """
        output = self.forward(x)
        # Clip output to prevent unreasonable predictions
        return np.clip(output, -100.0, 100.0)
    
    def train_step(self, x: np.ndarray, target: np.ndarray) -> float:
        """
        Single training step (forward + backward + structure update + rollback check).
        
        Args:
            x: Input vector
            target: Target output
            
        Returns:
            Reward for this step
        """
        # Forward pass
        prediction = self.forward(x)
        
        # Clip prediction to prevent explosion
        prediction = np.clip(prediction, -100.0, 100.0)
        
        # Compute reward and error
        reward, level = self.reward_function.compute_reward(prediction, target, return_level=True)
        self.last_reward = reward
        error = float(np.mean((prediction - target) ** 2))
        
        # Update modification tracker trajectory if active modification
        self.modification_tracker.update_trajectory(reward, error)
        
        # Record performance in active neurons/layers
        for layer in self.layers:
            layer.record_layer_performance(reward)
        
        # Compute gradient (simple MSE-based for now)
        grad_output = 2 * (prediction - target)
        grad_output = np.clip(grad_output, -10.0, 10.0)  # Clip gradient at source
        
        # Intelligent backward pass with role assignment
        self.backward_engine.backward_with_role_assignment(
            layers=self.layers,
            grad_output=grad_output,
            active_indices_per_layer=self.active_indices_history,
            learning_rate=self.learning_rate
        )
        
        # Check for bad configurations and perturb if needed
        for layer in self.layers:
            layer.check_and_avoid_bad_configs()
        
        # Track steps since last modification
        self.steps_since_modification += 1
        
        # Check for rollback (if modification was made and performance dropped)
        if self.network_snapshot is not None and self.steps_since_modification == 50:
            current_reward = self.reward_function.get_average_reward(50)
            # Disabled rollback during initial learning phase - let network explore
            # if current_reward < self.pre_modification_reward - 0.5:
            #     self._restore_network_snapshot(self.network_snapshot)
            #     self.modification_tracker.record_rollback()
            #     print(f"[Step {self.training_steps}] Rolled back modification")
            self.network_snapshot = None  # Clear snapshot
        
        # Periodically train meta-learner on modification history
        if self.training_steps % 500 == 0:
            self._train_meta_learner()
        
        # Periodically check and modify structure
        self.training_steps += 1
        if self.training_steps % self.check_structure_every == 0:
            self._modify_structure_intelligent()
        
        return reward
    
    def _get_network_state(self) -> Dict:
        """Get current network state for meta-learner."""
        # Calculate reward trend (linear regression slope)
        recent_rewards = self.reward_function.get_recent_rewards(100)
        if len(recent_rewards) > 10:
            x = np.arange(len(recent_rewards))
            slope = np.polyfit(x, recent_rewards, 1)[0]
            reward_trend = float(slope)
        else:
            reward_trend = 0.0
        
        return {
            'avg_reward': self.reward_function.get_average_reward(50),
            'avg_error': self.reward_function.get_average_error(50),
            'reward_trend': reward_trend,
            'neuron_count': sum(layer.get_neuron_count() for layer in self.layers),
            'layer_count': len(self.layers),
            'steps_since_last_mod': self.steps_since_modification,
            'network_age': self.training_steps,
            'plateau_detected': float(self.plateau_detected)  # Signal to meta-learner when stuck
        }
    
    def _train_meta_learner(self):
        """Train meta-learner on modification history."""
        X, y = self.modification_tracker.get_training_data()
        if X.shape[0] >= 20:  # Need at least 20 examples
            stats = self.meta_learner.train(X, y, epochs=5, batch_size=16)
            # Stats will be displayed in the main batch output
            self.last_meta_stats = stats
        else:
            self.last_meta_stats = None
    
    def _modify_structure_intelligent(self):
        """
        Intelligently modify network structure using meta-learner.
        
        Instead of blindly adding neurons, evaluate multiple strategies
        and choose the one most likely to succeed based on past experience.
        """
        # Get current network state
        network_state = self._get_network_state()
        
        # Get strategy scores from meta-learner
        strategy_scores = self.meta_learner.get_strategy_scores(network_state)
        
        # Add exploration noise (epsilon-greedy with epsilon=0.1)
        if np.random.random() < 0.1:
            # Random exploration
            chosen_strategy = np.random.choice(list(ModificationType))
            if chosen_strategy == ModificationType.NONE:
                return
        else:
            # Exploit best strategy
            # Filter out invalid strategies
            valid_strategies = {}
            for mod_type, score in strategy_scores.items():
                if self._is_strategy_valid(mod_type):
                    valid_strategies[mod_type] = score
            
            if len(valid_strategies) == 0:
                return  # No valid strategies
            
            # Choose best strategy
            chosen_strategy = max(valid_strategies, key=valid_strategies.get)
        
        # Save snapshot before modification
        self.network_snapshot = self._save_network_snapshot()
        self.pre_modification_reward = network_state['avg_reward']
        self.steps_since_modification = 0
        
        # Start tracking modification
        record = self.modification_tracker.start_modification(
            modification_type=chosen_strategy,
            timestamp=self.training_steps,
            network_state=network_state,
            context={'scores': strategy_scores}
        )
        
        # Execute chosen strategy
        success = self._execute_strategy(chosen_strategy)
        
        # Record network state after modification
        if success:
            self.modification_tracker.complete_modification(self._get_network_state())
        else:
            # Failed to execute - mark as rolled back immediately
            self.modification_tracker.record_rollback()
            self.network_snapshot = None
    
    def _is_strategy_valid(self, strategy: ModificationType) -> bool:
        """Check if a strategy can be applied to current network."""
        if strategy == ModificationType.ADD_NEURON:
            # Need at least one hidden layer and space to grow
            if len(self.layers) < 3:  # Need input + hidden + output
                return False
            return any(layer.get_neuron_count() < self.max_neurons_per_layer 
                      for i, layer in enumerate(self.layers) 
                      if i > 0 and i < len(self.layers) - 1)  # Only check hidden layers
        elif strategy == ModificationType.REMOVE_NEURON:
            # Need at least one hidden layer with removable neurons
            if len(self.layers) < 3:
                return False
            return any(layer.get_neuron_count() > self.min_neurons_per_layer 
                      for i, layer in enumerate(self.layers) 
                      if i > 0 and i < len(self.layers) - 1)  # Only check hidden layers
        elif strategy == ModificationType.ADD_LAYER:
            return len(self.layers) < self.max_layers
        elif strategy == ModificationType.REMOVE_LAYER:
            return len(self.layers) > self.min_layers
        elif strategy == ModificationType.REWIRE_CONNECTIONS:
            return len(self.layers) >= 2  # Need at least 2 layers to rewire
        elif strategy == ModificationType.ADJUST_THRESHOLDS:
            return True
        return False
    
    def _execute_strategy(self, strategy: ModificationType) -> bool:
        """Execute a modification strategy. Returns True if successful."""
        try:
            if strategy == ModificationType.ADD_NEURON:
                return self._strategy_add_neuron()
            elif strategy == ModificationType.REMOVE_NEURON:
                return self._strategy_remove_neuron()
            elif strategy == ModificationType.ADD_LAYER:
                return self._strategy_add_layer()
            elif strategy == ModificationType.REMOVE_LAYER:
                return self._strategy_remove_layer()
            elif strategy == ModificationType.REWIRE_CONNECTIONS:
                return self._strategy_rewire_connections()
            elif strategy == ModificationType.ADJUST_THRESHOLDS:
                return self._strategy_adjust_thresholds()
        except Exception as e:
            print(f"[Error] Failed to execute {strategy.value}: {e}")
            return False
        return False
    
    def _strategy_add_neuron(self) -> bool:
        """Strategy: Add neuron to a struggling layer using intelligent selection."""
        # Build list of candidate layers with their scores
        candidates = []
        
        for layer_id, layer in enumerate(self.layers):
            # Skip first and last layer (input/output layers should be stable)
            if layer_id == 0 or layer_id == len(self.layers) - 1:
                continue
                
            if layer.get_neuron_count() >= self.max_neurons_per_layer:
                continue
            
            # Calculate layer score based on multiple factors
            struggling = layer.get_struggling_neurons()
            struggling_ratio = len(struggling) / max(1, layer.get_neuron_count())
            
            # Get average layer performance
            layer_perf = np.mean([n.get_average_performance() for n in layer.neurons]) if layer.neurons else 0.0
            
            # Calculate activation rate (how often this layer is used)
            activation_rate = np.mean([n.activation_count for n in layer.neurons]) / max(1, self.training_steps) if layer.neurons else 0.0
            
            # Calculate layer "need" score
            # Higher score = more need for additional neurons
            need_score = (
                struggling_ratio * 2.0 +  # Prioritize layers with struggling neurons
                (1.0 - layer_perf) * 1.5 +  # Prioritize low-performing layers
                (1.0 - activation_rate) * 1.0 +  # Slightly prioritize underutilized layers
                (1.0 if layer.get_neuron_count() < self.min_neurons_per_layer else 0.0) * 3.0  # Strongly prioritize undersized layers
            )
            
            # Add randomness to encourage exploration
            need_score += np.random.normal(0, 0.3)
            
            candidates.append((layer_id, layer, need_score))
        
        if not candidates:
            return False
        
        # Select layer with highest need score
        candidates.sort(key=lambda x: x[2], reverse=True)
        chosen_layer_id, chosen_layer, score = candidates[0]
        
        # Select best performing neuron as parent
        parent = None
        if len(chosen_layer.neurons) > 0:
            # Find neuron with best average performance
            best_perf = float('-inf')
            for neuron in chosen_layer.neurons:
                perf = neuron.get_average_performance()
                if perf > best_perf:
                    best_perf = perf
                    parent = neuron
        
        chosen_layer.add_neuron(parent_neuron=parent)
        print(f"[Step {self.training_steps}] Added neuron to layer {chosen_layer_id} (need_score={score:.2f}, Strategy: ADD_NEURON)")
        return True
    
    def _strategy_remove_neuron(self) -> bool:
        """Strategy: Remove underperforming neuron."""
        for layer_id, layer in enumerate(self.layers):
            # Skip first and last layer (input/output layers should be stable)
            if layer_id == 0 or layer_id == len(self.layers) - 1:
                continue
                
            if layer.get_neuron_count() <= self.min_neurons_per_layer:
                continue
            
            removed = layer.remove_underperforming_neurons(performance_threshold=-2.0)
            if removed == 0:
                removed = layer.remove_rarely_active_neurons(min_activations=3)
            
            if removed > 0:
                print(f"[Step {self.training_steps}] Removed {removed} neurons from layer {layer_id} (Strategy: REMOVE_NEURON)")
                return True
        return False
    
    def _strategy_add_layer(self) -> bool:
        """Strategy: Add a new layer to the network."""
        if len(self.layers) >= self.max_layers:
            return False
        
        insert_pos = max(1, len(self.layers) // 2)
        layer_input_dim = self.layers[insert_pos - 1].output_dim
        layer_output_dim = layer_input_dim
        
        new_layer = AdaptiveLayer(
            initial_neuron_count=self.min_neurons_per_layer,
            input_dim=layer_input_dim,
            output_dim=layer_output_dim,
            tag_dim=self.tag_dim
        )
        
        self.layers.insert(insert_pos, new_layer)
        print(f"[Step {self.training_steps}] Added layer at position {insert_pos} (Strategy: ADD_LAYER)")
        return True
    
    def _strategy_remove_layer(self) -> bool:
        """Strategy: Remove a poorly performing layer."""
        if len(self.layers) <= self.min_layers:
            return False
        
        # Don't remove first or last layer
        if len(self.layers) <= 2:
            return False
        
        # Find layer with lowest average performance (excluding first/last)
        min_perf = float('inf')
        worst_layer_idx = 1
        
        for i in range(1, len(self.layers) - 1):
            layer = self.layers[i]
            # Calculate average performance from neurons in this layer
            if len(layer.neurons) == 0:
                avg_perf = -1.0
            else:
                avg_perf = np.mean([n.get_average_performance() for n in layer.neurons])
            
            if avg_perf < min_perf:
                min_perf = avg_perf
                worst_layer_idx = i
        
        self.layers.pop(worst_layer_idx)
        print(f"[Step {self.training_steps}] Removed layer {worst_layer_idx} (Strategy: REMOVE_LAYER)")
        return True
    
    def _strategy_rewire_connections(self) -> bool:
        """Strategy: Rewire neurons to connect to different layers."""
        if len(self.layers) < 2:
            return False
        
        # Pick a random middle layer to rewire from
        source_layer_idx = np.random.randint(0, len(self.layers) - 1)
        source_layer = self.layers[source_layer_idx]
        
        if source_layer.get_neuron_count() == 0:
            return False
        
        # Pick a random neuron to rewire
        neuron_idx = np.random.randint(0, source_layer.get_neuron_count())
        neuron = source_layer.neurons[neuron_idx]
        
        if neuron.is_broadcast:
            return False  # Don't rewire broadcast neurons
        
        # Pick a new target layer (consider physical proximity)
        possible_targets = []
        for target_idx in range(source_layer_idx + 1, min(source_layer_idx + 4, len(self.layers))):
            possible_targets.append(target_idx)
        
        if len(possible_targets) == 0:
            return False
        
        new_target_layer_idx = np.random.choice(possible_targets)
        new_target_layer = self.layers[new_target_layer_idx]
        
        # Get proximity threshold
        threshold = source_layer.get_layer_proximity_threshold(source_layer_idx, new_target_layer_idx)
        
        # Find neurons in target layer with matching tags
        target_indices = []
        for idx, target_neuron in enumerate(new_target_layer.neurons):
            # Check tag similarity
            similarities = []
            for out_tag in neuron.output_tags:
                for in_tag in target_neuron.input_tags:
                    sim = np.dot(out_tag, in_tag)
                    similarities.append(sim)
            max_sim = max(similarities) if similarities else 0
            
            if max_sim >= threshold:
                target_indices.append(idx)
        
        # If no matches, pick a random target
        if len(target_indices) == 0:
            target_indices = [np.random.randint(0, new_target_layer.get_neuron_count())]
        
        # Rewire the neuron
        source_layer.rewire_neuron(neuron_idx, new_target_layer_idx, target_indices)
        print(f"[Step {self.training_steps}] Rewired neuron {neuron_idx} in layer {source_layer_idx} -> layer {new_target_layer_idx} (Strategy: REWIRE)")
        return True
    
    def _strategy_adjust_thresholds(self) -> bool:
        """Strategy: Adjust activation thresholds for all neurons."""
        # Randomly adjust thresholds slightly up or down
        adjustment = np.random.choice([-0.05, 0.05])
        
        adjusted = 0
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.activation_threshold = np.clip(
                    neuron.activation_threshold + adjustment,
                    0.1,  # Min threshold
                    0.7   # Max threshold
                )
                adjusted += 1
        
        direction = "increased" if adjustment > 0 else "decreased"
        print(f"[Step {self.training_steps}] {direction.capitalize()} activation thresholds for {adjusted} neurons (Strategy: ADJUST_THRESHOLDS)")
        return True
    
    def _modify_structure(self):
        """Old blind modification (kept for compatibility)."""
        avg_reward = self.reward_function.get_average_reward(last_n=50)
        is_struggling = self.reward_function.is_struggling()
        is_performing_well = self.reward_function.is_performing_well()
        
        # GROWTH: Add structure if struggling
        if is_struggling:
            self._grow_network()
        
        # PRUNING: Remove structure if performing well and network is large
        if is_performing_well and self._is_network_large():
            self._prune_network()
        
        # LAYER ADJUSTMENT: Add or remove layers based on performance trends
        # Check every 4 structure checks (more conservative)
        if self.training_steps % (self.check_structure_every * 4) == 0:
            if is_struggling and len(self.layers) < self.max_layers:
                self._add_layer()
                print(f"[Step {self.training_steps}] Added layer due to poor performance. Total layers: {len(self.layers)}")
            elif is_performing_well and len(self.layers) > self.min_layers:
                self._remove_layer()
    
    def _grow_network(self):
        """Add neurons to struggling areas."""
        neurons_added = 0
        for layer_id, layer in enumerate(self.layers):
            # Find struggling neurons
            struggling = layer.get_struggling_neurons()
            
            if len(struggling) > 0:
                # Add neuron near a struggling neuron's tag
                idx, neuron = struggling[0]
                layer.add_neuron(near_tag=neuron.functional_tag)
                neurons_added += 1
            elif layer.get_neuron_count() < self.min_neurons_per_layer:
                # Ensure minimum neurons
                layer.add_neuron()
                neurons_added += 1
        
        if neurons_added > 0:
            print(f"[Step {self.training_steps}] Grew network: +{neurons_added} neurons")
    
    def _prune_network(self):
        """Remove underperforming neurons from the network."""
        total_removed = 0
        for layer in self.layers:
            if layer.get_neuron_count() > self.min_neurons_per_layer:
                # Remove rarely active neurons
                removed = layer.remove_rarely_active_neurons(min_activations=3)
                total_removed += removed
                
                # Also remove underperforming neurons
                if removed == 0:
                    removed = layer.remove_underperforming_neurons(performance_threshold=-2.0)
                    total_removed += removed
        
        if total_removed > 0:
            print(f"[Step {self.training_steps}] Pruned network: -{total_removed} neurons")
        
        # Reset activation counts for next evaluation period
        for layer in self.layers:
            layer.reset_activation_counts()
    
    def _is_network_large(self) -> bool:
        """Check if network is larger than necessary."""
        total_neurons = sum(layer.get_neuron_count() for layer in self.layers)
        avg_neurons_per_layer = total_neurons / len(self.layers)
        
        return len(self.layers) > 3 or avg_neurons_per_layer > 10
    
    def _add_layer(self):
        """Add a new layer to the network."""
        if len(self.layers) >= self.max_layers:
            return
        
        # Insert layer in middle of network
        insert_position = len(self.layers) // 2
        
        if insert_position == 0:
            input_dim = self.input_dim
        else:
            input_dim = self.layers[insert_position - 1].output_dim
        
        output_dim = input_dim  # Keep dimension consistent
        
        new_layer = AdaptiveLayer(
            initial_neuron_count=self.min_neurons_per_layer,
            input_dim=input_dim,
            output_dim=output_dim,
            tag_dim=self.tag_dim
        )
        
        self.layers.insert(insert_position, new_layer)
        print(f"Added layer at position {insert_position}. Total layers: {len(self.layers)}")
    
    def _remove_layer(self):
        """Remove the least useful layer."""
        if len(self.layers) <= self.min_layers:
            return
        
        # Remove layer with lowest average performance
        worst_layer_idx = 0
        worst_performance = float('inf')
        
        for i, layer in enumerate(self.layers):
            stats = layer.get_layer_stats()
            if stats['avg_performance'] < worst_performance:
                worst_performance = stats['avg_performance']
                worst_layer_idx = i
        
        # Don't remove first or last layer (input/output critical)
        if 0 < worst_layer_idx < len(self.layers) - 1:
            self.layers.pop(worst_layer_idx)
            print(f"Removed layer at position {worst_layer_idx}. Total layers: {len(self.layers)}")
    
    def get_network_stats(self) -> Dict:
        """Get comprehensive network statistics."""
        layer_stats = []
        total_neurons = 0
        total_broadcast = 0
        
        for i, layer in enumerate(self.layers):
            stats = layer.get_layer_stats()
            layer_stats.append(stats)
            total_neurons += stats['total_neurons']
            total_broadcast += stats['broadcast_neurons']
        
        return {
            'total_layers': len(self.layers),
            'total_neurons': total_neurons,
            'total_broadcast_neurons': total_broadcast,
            'training_steps': self.training_steps,
            'last_reward': self.last_reward,
            'avg_reward_50': self.reward_function.get_average_reward(50),
            'avg_error_50': self.reward_function.get_average_error(50),
            'layer_stats': layer_stats,
            'backward_stats': self.backward_engine.get_statistics(),
            'reward_stats': self.reward_function.get_statistics()
        }
    
    def print_network_summary(self):
        """Print a summary of the network structure and performance."""
        stats = self.get_network_stats()
        
        print("\n" + "="*60)
        print("SELF-MODIFYING NETWORK SUMMARY")
        print("="*60)
        print(f"Training Steps: {stats['training_steps']}")
        print(f"Total Layers: {stats['total_layers']}")
        print(f"Total Neurons: {stats['total_neurons']} ({stats['total_broadcast_neurons']} broadcast)")
        print(f"Last Reward: {stats['last_reward']:.3f}")
        print(f"Avg Reward (50): {stats['avg_reward_50']:.3f}")
        print(f"Avg Error (50): {stats['avg_error_50']:.4f}")
        print()
        
        print("Layer Details:")
        for i, layer_stat in enumerate(stats['layer_stats']):
            print(f"  Layer {i}: {layer_stat['total_neurons']} neurons, "
                  f"avg perf={layer_stat['avg_performance']:.3f}, "
                  f"activations={layer_stat['layer_activations']}")
        print()
        
        print("Backward Engine:")
        print(f"  Neurons classified: {stats['backward_stats']['total_neurons_classified']}")
        print(f"  Bad updates remembered: {stats['backward_stats']['bad_updates_remembered']}")
        print()
        
        print("Reward Distribution:")
        for level, pct in stats['reward_stats']['level_distribution'].items():
            print(f"  {level}: {pct:.1f}%")
        print("="*60 + "\n")
    
    def save_state(self, filepath: str):
        """Save network state to file."""
        import pickle
        state = {
            'layers': [layer.get_layer_stats() for layer in self.layers],
            'reward_stats': self.reward_function.get_statistics(),
            'training_steps': self.training_steps
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    
    def __repr__(self):
        return f"SelfModifyingNetwork(layers={len(self.layers)}, neurons={sum(l.get_neuron_count() for l in self.layers)}, steps={self.training_steps})"
