"""
Batched version of the Self-Modifying Network for GPU efficiency.

This module provides a batched training approach that processes multiple samples
simultaneously to leverage GPU parallelism while preserving the self-modifying
network architecture.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict
from .layer import AdaptiveLayer
from .reward import GradedRewardFunction
from .backward import IntelligentBackward
from .modification_tracker import ModificationTracker, ModificationType
from .meta_learner import MetaLearner
from .neuron import DEVICE


class BatchedSelfModifyingNetwork:
    """
    A batched version of SelfModifyingNetwork that processes multiple samples
    at once for GPU efficiency.
    
    Key differences from SelfModifyingNetwork:
    - forward_batch: Processes entire batches through the network
    - train_batch: Updates weights once per batch (not per sample)
    - Structure modifications happen at batch boundaries
    - Tags are computed for each sample but routing is vectorized
    """
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 initial_layers: int = 2,
                 initial_neurons_per_layer: int = 5,
                 tag_dim: int = 4,
                 learning_rate: float = 0.01,
                 device: torch.device = None):
        """
        Initialize batched self-modifying network.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            initial_layers: Number of layers to start with
            initial_neurons_per_layer: Neurons per layer initially
            tag_dim: Dimension of tag vectors
            learning_rate: Base learning rate
            device: PyTorch device (defaults to CUDA if available)
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.tag_dim = tag_dim
        self.learning_rate = learning_rate
        self.device = device if device is not None else DEVICE
        
        # Network structure - store as batched weight matrices
        self.layers: List[BatchedLayer] = []
        self._initialize_layers(initial_layers, initial_neurons_per_layer)
        
        # Components (reuse from original)
        self.reward_function = GradedRewardFunction()
        self.backward_engine = IntelligentBackward(tag_dim=tag_dim)
        
        # Meta-learning components
        self.modification_tracker = ModificationTracker(max_history=1000)
        self.meta_learner = MetaLearner(input_dim=15, hidden_dim=32, learning_rate=0.01)
        
        # Training state
        self.training_steps = 0
        self.batch_count = 0
        self.last_reward = 0.0
        self.last_modification_step = 0
        self.network_snapshot = None
        self.pre_modification_reward = 0.0
        self.steps_since_modification = 0
        self.plateau_detected = False
        self.last_meta_stats = None
        
        # Structure modification parameters
        self.min_layers = 1
        self.max_layers = 15
        self.min_neurons_per_layer = 2
        self.max_neurons_per_layer = 256
        self.check_structure_every = 100  # Check every N batches
        
        # Tag projection matrix (for converting inputs to tags)
        self._tag_projection = torch.randn(
            input_dim, tag_dim, device=self.device
        ) * 0.1
        
    def _initialize_layers(self, num_layers: int, neurons_per_layer: int):
        """Initialize the network layers with batched operations."""
        current_dim = self.input_dim
        
        for i in range(num_layers):
            if i == num_layers - 1:
                layer_output_dim = self.output_dim
            else:
                layer_output_dim = current_dim
            
            layer = BatchedLayer(
                neuron_count=neurons_per_layer,
                input_dim=current_dim,
                output_dim=layer_output_dim,
                tag_dim=self.tag_dim,
                device=self.device,
                is_output_layer=(i == num_layers - 1)
            )
            self.layers.append(layer)
            current_dim = layer_output_dim
    
    def input_to_tag_batch(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert batched inputs to tag vectors.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Tag tensor of shape (batch_size, tag_dim)
        """
        tags = x @ self._tag_projection
        tags = F.normalize(tags, p=2, dim=-1)
        return tags
    
    def forward_batch(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Batched forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Tuple of (output tensor, list of activation masks per layer)
        """
        batch_size = x.shape[0]
        
        # Compute tags for batch
        tags = self.input_to_tag_batch(x)
        
        # Track activations for backward pass
        activation_masks = []
        
        # Forward through layers
        current_input = x
        for layer in self.layers:
            current_input, mask = layer.forward_batch(current_input, tags)
            activation_masks.append(mask)
        
        return current_input, activation_masks
    
    def train_batch(self, 
                    x_batch: np.ndarray, 
                    y_batch: np.ndarray) -> Tuple[float, float]:
        """
        Train on a batch of samples.
        
        Args:
            x_batch: Input batch of shape (batch_size, input_dim)
            y_batch: Target batch of shape (batch_size, output_dim)
            
        Returns:
            Tuple of (average reward, average error)
        """
        batch_size = x_batch.shape[0]
        
        # Convert to tensors on device
        x = torch.tensor(x_batch, dtype=torch.float32, device=self.device)
        y = torch.tensor(y_batch, dtype=torch.float32, device=self.device)
        
        # Forward pass
        predictions, activation_masks = self.forward_batch(x)
        
        # Clip predictions
        predictions = torch.clamp(predictions, -100.0, 100.0)
        
        # Compute loss (MSE)
        loss = F.mse_loss(predictions, y, reduction='none')
        batch_errors = loss.mean(dim=-1)  # Per-sample error
        
        # Compute rewards for each sample
        pred_np = predictions.detach().cpu().numpy()
        target_np = y_batch
        batch_rewards = []
        
        for i in range(batch_size):
            reward, level = self.reward_function.compute_reward(
                pred_np[i], target_np[i], return_level=True
            )
            batch_rewards.append(reward)
        
        avg_reward = np.mean(batch_rewards)
        avg_error = batch_errors.mean().item()
        self.last_reward = avg_reward
        
        # Backward pass with gradient accumulation
        grad_output = 2 * (predictions - y)  # Don't divide by batch_size here - do it in layer
        grad_output = torch.clamp(grad_output, -10.0, 10.0)
        
        # Backpropagate through layers
        current_grad = grad_output
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            mask = activation_masks[i]
            current_grad = layer.backward_batch(current_grad, mask, self.learning_rate)
        
        # Update counters
        self.training_steps += batch_size
        self.batch_count += 1
        self.steps_since_modification += 1
        
        # Periodic meta-learner training
        if self.batch_count % 10 == 0:
            self._train_meta_learner()
        
        # Check structure modification
        if self.batch_count % self.check_structure_every == 0:
            self._modify_structure_intelligent()
        
        return avg_reward, avg_error
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Make predictions (can be single sample or batch).
        
        Args:
            x: Input array of shape (input_dim,) or (batch_size, input_dim)
            
        Returns:
            Predictions clipped to reasonable range
        """
        single_sample = x.ndim == 1
        if single_sample:
            x = x.reshape(1, -1)
        
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            output, _ = self.forward_batch(x_tensor)
            output = torch.clamp(output, -100.0, 100.0)
        
        result = output.cpu().numpy()
        if single_sample:
            result = result.squeeze(0)
        
        return result
    
    def _get_network_state(self) -> Dict:
        """Get current network state for meta-learner."""
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
            'neuron_count': sum(layer.neuron_count for layer in self.layers),
            'layer_count': len(self.layers),
            'steps_since_last_mod': self.steps_since_modification,
            'network_age': self.training_steps,
            'plateau_detected': float(self.plateau_detected)
        }
    
    def _train_meta_learner(self):
        """Train meta-learner on modification history."""
        X, y = self.modification_tracker.get_training_data()
        if X.shape[0] >= 20:
            stats = self.meta_learner.train_model(X, y, epochs=5, batch_size=16)
            self.last_meta_stats = stats
        else:
            self.last_meta_stats = None
    
    def _modify_structure_intelligent(self):
        """Intelligently modify network structure using meta-learner."""
        network_state = self._get_network_state()
        strategy_scores = self.meta_learner.get_strategy_scores(network_state)
        
        # Epsilon-greedy exploration
        if np.random.random() < 0.1:
            chosen_strategy = np.random.choice(list(ModificationType))
            if chosen_strategy == ModificationType.NONE:
                return
        else:
            valid_strategies = {}
            for mod_type, score in strategy_scores.items():
                if self._is_strategy_valid(mod_type):
                    valid_strategies[mod_type] = score
            
            if len(valid_strategies) == 0:
                return
            
            chosen_strategy = max(valid_strategies, key=valid_strategies.get)
        
        # Save snapshot and execute
        self.pre_modification_reward = network_state['avg_reward']
        self.steps_since_modification = 0
        
        record = self.modification_tracker.start_modification(
            modification_type=chosen_strategy,
            timestamp=self.training_steps,
            network_state=network_state,
            context={'scores': strategy_scores}
        )
        
        success = self._execute_strategy(chosen_strategy)
        
        if success:
            self.modification_tracker.complete_modification(self._get_network_state())
        else:
            self.modification_tracker.record_rollback()
    
    def _is_strategy_valid(self, strategy: ModificationType) -> bool:
        """Check if a strategy can be applied."""
        if strategy == ModificationType.ADD_NEURON:
            return any(layer.neuron_count < self.max_neurons_per_layer 
                      for layer in self.layers[:-1])
        elif strategy == ModificationType.REMOVE_NEURON:
            return any(layer.neuron_count > self.min_neurons_per_layer 
                      for layer in self.layers[:-1])
        elif strategy == ModificationType.ADD_LAYER:
            return len(self.layers) < self.max_layers
        elif strategy == ModificationType.REMOVE_LAYER:
            return len(self.layers) > self.min_layers + 1
        elif strategy == ModificationType.REWIRE_CONNECTIONS:
            return len(self.layers) >= 2
        elif strategy == ModificationType.ADJUST_THRESHOLDS:
            return True
        return False
    
    def _execute_strategy(self, strategy: ModificationType) -> bool:
        """Execute a modification strategy."""
        try:
            if strategy == ModificationType.ADD_NEURON:
                return self._strategy_add_neuron()
            elif strategy == ModificationType.REMOVE_NEURON:
                return self._strategy_remove_neuron()
            elif strategy == ModificationType.ADD_LAYER:
                return self._strategy_add_layer()
            elif strategy == ModificationType.REMOVE_LAYER:
                return self._strategy_remove_layer()
            elif strategy == ModificationType.ADJUST_THRESHOLDS:
                return self._strategy_adjust_thresholds()
        except Exception as e:
            print(f"[Error] Failed to execute {strategy.value}: {e}")
            return False
        return False
    
    def _strategy_add_neuron(self) -> bool:
        """Add neuron to a layer."""
        # Find layer with most room to grow (excluding output layer)
        candidates = [(i, layer) for i, layer in enumerate(self.layers[:-1]) 
                     if layer.neuron_count < self.max_neurons_per_layer]
        
        if not candidates:
            return False
        
        # Pick randomly for now (could be smarter based on performance)
        layer_idx, layer = candidates[np.random.randint(len(candidates))]
        layer.add_neuron()
        print(f"[Batch {self.batch_count}] Added neuron to layer {layer_idx}")
        return True
    
    def _strategy_remove_neuron(self) -> bool:
        """Remove underperforming neuron."""
        candidates = [(i, layer) for i, layer in enumerate(self.layers[:-1])
                     if layer.neuron_count > self.min_neurons_per_layer]
        
        if not candidates:
            return False
        
        layer_idx, layer = candidates[np.random.randint(len(candidates))]
        layer.remove_neuron()
        print(f"[Batch {self.batch_count}] Removed neuron from layer {layer_idx}")
        return True
    
    def _strategy_add_layer(self) -> bool:
        """Add a new layer."""
        if len(self.layers) >= self.max_layers:
            return False
        
        insert_pos = max(1, len(self.layers) // 2)
        input_dim = self.layers[insert_pos - 1].output_dim
        
        new_layer = BatchedLayer(
            neuron_count=self.min_neurons_per_layer,
            input_dim=input_dim,
            output_dim=input_dim,
            tag_dim=self.tag_dim,
            device=self.device,
            is_output_layer=False
        )
        
        self.layers.insert(insert_pos, new_layer)
        print(f"[Batch {self.batch_count}] Added layer at position {insert_pos}")
        return True
    
    def _strategy_remove_layer(self) -> bool:
        """Remove a layer."""
        if len(self.layers) <= 2:
            return False
        
        # Remove a middle layer
        remove_idx = len(self.layers) // 2
        self.layers.pop(remove_idx)
        print(f"[Batch {self.batch_count}] Removed layer at position {remove_idx}")
        return True
    
    def _strategy_adjust_thresholds(self) -> bool:
        """Adjust activation thresholds."""
        adjustment = np.random.choice([-0.05, 0.05])
        
        for layer in self.layers:
            layer.activation_threshold = np.clip(
                layer.activation_threshold + adjustment,
                0.1, 0.7
            )
        
        direction = "increased" if adjustment > 0 else "decreased"
        print(f"[Batch {self.batch_count}] {direction.capitalize()} activation thresholds")
        return True
    
    def get_network_stats(self) -> Dict:
        """Get comprehensive network statistics."""
        layer_stats = []
        total_neurons = 0
        
        for i, layer in enumerate(self.layers):
            stats = {
                'total_neurons': layer.neuron_count,
                'avg_performance': 0.0,  # Could track if needed
                'layer_activations': 0
            }
            layer_stats.append(stats)
            total_neurons += layer.neuron_count
        
        return {
            'total_layers': len(self.layers),
            'total_neurons': total_neurons,
            'total_broadcast_neurons': 0,
            'training_steps': self.training_steps,
            'batch_count': self.batch_count,
            'last_reward': self.last_reward,
            'avg_reward_50': self.reward_function.get_average_reward(50),
            'avg_error_50': self.reward_function.get_average_error(50),
            'layer_stats': layer_stats,
            'reward_stats': self.reward_function.get_statistics()
        }
    
    def print_network_summary(self):
        """Print network summary."""
        stats = self.get_network_stats()
        
        print("\n" + "="*60)
        print("BATCHED SELF-MODIFYING NETWORK SUMMARY")
        print("="*60)
        print(f"Training Steps: {stats['training_steps']}")
        print(f"Batch Count: {stats['batch_count']}")
        print(f"Total Layers: {stats['total_layers']}")
        print(f"Total Neurons: {stats['total_neurons']}")
        print(f"Last Reward: {stats['last_reward']:.3f}")
        print(f"Avg Reward (50): {stats['avg_reward_50']:.3f}")
        print(f"Avg Error (50): {stats['avg_error_50']:.4f}")
        print(f"Device: {self.device}")
        print("="*60 + "\n")


class BatchedLayer:
    """
    A layer optimized for batched operations.
    
    Instead of storing individual Neuron objects, this stores weight matrices
    and computes everything in parallel.
    """
    
    def __init__(self,
                 neuron_count: int,
                 input_dim: int,
                 output_dim: int,
                 tag_dim: int = 4,
                 device: torch.device = None,
                 is_output_layer: bool = False):
        """
        Initialize batched layer.
        
        Args:
            neuron_count: Number of neurons
            input_dim: Input dimension
            output_dim: Output dimension per neuron
            tag_dim: Tag dimension for routing
            device: PyTorch device
            is_output_layer: Whether this is the output layer
        """
        self.neuron_count = neuron_count
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.tag_dim = tag_dim
        self.device = device if device is not None else DEVICE
        self.is_output_layer = is_output_layer
        
        # Initialize weights for all neurons: (neuron_count, input_dim, output_dim)
        # But for efficiency, we reshape to process all neurons at once
        scale = np.sqrt(2.0 / (input_dim + output_dim))
        
        # Combined weight matrix: (input_dim, neuron_count * output_dim)
        # This allows efficient batch matrix multiplication
        self.weights = torch.nn.Parameter(
            torch.randn(input_dim, neuron_count * output_dim, device=self.device) * scale
        )
        self.bias = torch.nn.Parameter(
            torch.zeros(neuron_count * output_dim, device=self.device)
        )
        
        # If output layer, initialize bias to middle of expected range
        # For addition (0+0 to 3+3) that's ~3, for division (1/3 to 3/1) that's ~1.5
        # Use 2.0 as a compromise
        if is_output_layer:
            self.bias.data.fill_(2.0)
        
        # Functional tags for each neuron: (neuron_count, tag_dim)
        self.functional_tags = torch.randn(
            neuron_count, tag_dim, device=self.device
        )
        self.functional_tags = F.normalize(self.functional_tags, p=2, dim=-1)
        
        # Activation threshold - start low to allow more neurons to activate
        self.activation_threshold = 0.3
        
        # Cache for backward pass
        self.last_input = None
        self.last_pre_activation = None
        self.last_output = None
    
    def forward_batch(self, 
                      x: torch.Tensor, 
                      tags: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batched forward pass.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            tags: Tag tensor (batch_size, tag_dim)
            
        Returns:
            Tuple of (output tensor, activation mask)
        """
        batch_size = x.shape[0]
        self.last_input = x
        
        # Compute tag similarities: (batch_size, neuron_count)
        # Each sample's tag compared to each neuron's functional tag
        similarities = torch.mm(tags, self.functional_tags.T)
        similarities = (similarities + 1.0) / 2.0  # Scale to [0, 1]
        
        # Create soft activation mask (smooth rather than hard cutoff)
        # This allows gradients to flow better
        activation_mask = torch.sigmoid((similarities - self.activation_threshold) * 10.0)
        
        # Forward through all neurons at once
        # x @ weights -> (batch_size, neuron_count * output_dim)
        linear_output = x @ self.weights + self.bias
        self.last_pre_activation = linear_output
        
        # Reshape to (batch_size, neuron_count, output_dim)
        linear_output = linear_output.view(batch_size, self.neuron_count, self.output_dim)
        
        # Apply activation function
        if self.is_output_layer:
            activated = linear_output  # Linear for output
        else:
            activated = torch.tanh(linear_output)  # Tanh for hidden
        
        # Apply mask: scale outputs by activation strength
        # mask shape: (batch_size, neuron_count, 1) for broadcasting
        mask_expanded = activation_mask.unsqueeze(-1)
        masked_output = activated * mask_expanded
        
        # Combine outputs: weighted average by activation strength
        # This gives smoother gradients than hard masking
        active_weight = activation_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        combined = masked_output.sum(dim=1) / active_weight
        
        self.last_output = combined
        
        return combined, activation_mask
    
    def backward_batch(self,
                       grad_output: torch.Tensor,
                       activation_mask: torch.Tensor,
                       learning_rate: float) -> torch.Tensor:
        """
        Batched backward pass.
        
        Args:
            grad_output: Gradient tensor (batch_size, output_dim)
            activation_mask: Which neurons were active (batch_size, neuron_count)
            learning_rate: Learning rate
            
        Returns:
            Gradient w.r.t. input (batch_size, input_dim)
        """
        batch_size = grad_output.shape[0]
        
        if self.last_input is None:
            return torch.zeros(batch_size, self.input_dim, device=self.device)
        
        with torch.no_grad():
            # Expand grad_output for each neuron
            # grad_output: (batch_size, output_dim) -> (batch_size, neuron_count, output_dim)
            active_count = activation_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            grad_per_neuron = grad_output.unsqueeze(1) / active_count.unsqueeze(-1)
            grad_per_neuron = grad_per_neuron.expand(-1, self.neuron_count, -1)
            
            # Apply mask
            mask_expanded = activation_mask.unsqueeze(-1)
            grad_per_neuron = grad_per_neuron * mask_expanded
            
            # Apply activation gradient (tanh for hidden, linear for output)
            if not self.is_output_layer:
                # Need to reconstruct pre-activation output for tanh gradient
                pre_act = self.last_pre_activation.view(batch_size, self.neuron_count, self.output_dim)
                tanh_grad = 1 - torch.tanh(pre_act) ** 2
                grad_per_neuron = grad_per_neuron * tanh_grad
            
            # Clip gradients
            grad_per_neuron = torch.clamp(grad_per_neuron, -5.0, 5.0)
            
            # Reshape for weight update: (batch_size, neuron_count * output_dim)
            grad_flat = grad_per_neuron.view(batch_size, -1)
            
            # Weight gradient: input.T @ grad
            grad_weights = self.last_input.T @ grad_flat / batch_size
            grad_bias = grad_flat.mean(dim=0)
            
            # Clip and update
            grad_weights = torch.clamp(grad_weights, -5.0, 5.0)
            grad_bias = torch.clamp(grad_bias, -5.0, 5.0)
            
            self.weights.data -= learning_rate * grad_weights
            self.bias.data -= learning_rate * grad_bias
            
            # Clamp weights
            self.weights.data.clamp_(-20.0, 20.0)
            self.bias.data.clamp_(-20.0, 20.0)
            
            # Gradient w.r.t. input (for previous layer)
            grad_input = grad_flat @ self.weights.T
        
        return grad_input
    
    def add_neuron(self):
        """Add a neuron to the layer (grows the weight matrices)."""
        with torch.no_grad():
            # Add columns to weights
            scale = np.sqrt(2.0 / (self.input_dim + self.output_dim))
            new_weights = torch.randn(
                self.input_dim, self.output_dim, device=self.device
            ) * scale
            
            self.weights = torch.nn.Parameter(
                torch.cat([self.weights.data, new_weights], dim=1)
            )
            
            # Add to bias
            new_bias = torch.zeros(self.output_dim, device=self.device)
            if self.is_output_layer:
                new_bias.fill_(9.0)
            self.bias = torch.nn.Parameter(
                torch.cat([self.bias.data, new_bias])
            )
            
            # Add functional tag
            new_tag = F.normalize(
                torch.randn(1, self.tag_dim, device=self.device),
                p=2, dim=-1
            )
            self.functional_tags = torch.cat([self.functional_tags, new_tag], dim=0)
            
            self.neuron_count += 1
    
    def remove_neuron(self, index: int = -1):
        """Remove a neuron from the layer."""
        if self.neuron_count <= 1:
            return
        
        with torch.no_grad():
            if index < 0:
                index = self.neuron_count + index
            
            # Calculate slice ranges (each neuron has output_dim columns)
            start = index * self.output_dim
            end = (index + 1) * self.output_dim
            
            # Remove from weights
            self.weights = torch.nn.Parameter(
                torch.cat([
                    self.weights.data[:, :start],
                    self.weights.data[:, end:]
                ], dim=1)
            )
            
            # Remove from bias
            self.bias = torch.nn.Parameter(
                torch.cat([
                    self.bias.data[:start],
                    self.bias.data[end:]
                ])
            )
            
            # Remove functional tag
            self.functional_tags = torch.cat([
                self.functional_tags[:index],
                self.functional_tags[index+1:]
            ], dim=0)
            
            self.neuron_count -= 1
    
    def __repr__(self):
        return f"BatchedLayer(neurons={self.neuron_count}, in={self.input_dim}, out={self.output_dim})"
