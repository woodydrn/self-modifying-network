import numpy as np
from typing import List, Dict, Optional, Tuple
from .neuron import Neuron


class AdaptiveLayer:
    """
    A layer of neurons that can dynamically grow/shrink and route connections selectively.
    
    Attributes:
        neurons: List of neurons in this layer
        layer_tag: Functional tag describing the layer's purpose
        input_dim: Expected input dimension
        output_dim: Expected output dimension  
        routing_map: Maps input tags to preferred neuron indices
        performance_history: Layer-level performance tracking
    """
    
    def __init__(self,
                 initial_neuron_count: int,
                 input_dim: int,
                 output_dim: int,
                 layer_tag: Optional[np.ndarray] = None,
                 broadcast_ratio: float = 0.2,
                 tag_dim: int = 4):
        """
        Initialize an adaptive layer.
        
        Args:
            initial_neuron_count: Number of neurons to start with
            input_dim: Input dimension
            output_dim: Output dimension per neuron
            layer_tag: Functional tag for the layer
            broadcast_ratio: Fraction of neurons that broadcast (connect to all)
            tag_dim: Dimension of tag vectors
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.tag_dim = tag_dim
        self.broadcast_ratio = broadcast_ratio
        
        # Layer functional tag
        if layer_tag is None:
            self.layer_tag = self._random_unit_vector(tag_dim)
        else:
            self.layer_tag = layer_tag
        
        # Initialize neurons
        self.neurons: List[Neuron] = []
        for i in range(initial_neuron_count):
            is_broadcast = (i < int(initial_neuron_count * broadcast_ratio))
            neuron = Neuron(input_dim, output_dim, is_broadcast=is_broadcast, tag_dim=tag_dim)
            neuron.is_output_neuron = False  # Will be set by network if this is output layer
            self.neurons.append(neuron)
        
        # Routing and performance
        self.routing_map: Dict[str, List[int]] = {}
        self.performance_history = []
        self.layer_activations = 0
        
    def _random_unit_vector(self, dim: int) -> np.ndarray:
        """Generate a random unit vector."""
        vec = np.random.randn(dim)
        return vec / (np.linalg.norm(vec) + 1e-8)
    
    def forward(self, x: np.ndarray, input_tag: np.ndarray, 
                next_layer: Optional['AdaptiveLayer'] = None) -> Tuple[np.ndarray, List[int]]:
        """
        Forward pass through the layer with tag-based selective activation.
        
        Args:
            x: Input vector
            input_tag: Tag vector from input
            next_layer: Next layer for selective routing (optional)
            
        Returns:
            Tuple of (combined output, list of active neuron indices)
        """
        outputs = []
        active_indices = []
        neuron_outputs = {}  # Store individual neuron outputs
        
        # Each neuron decides whether to activate
        for i, neuron in enumerate(self.neurons):
            output = neuron.forward(x, input_tag)
            if output is not None:
                neuron_outputs[i] = output
                active_indices.append(i)
        
        # If no next layer, use simple averaging
        if next_layer is None or len(active_indices) == 0:
            if len(neuron_outputs) == 0:
                combined_output = np.zeros(self.output_dim)
            else:
                combined_output = np.mean(list(neuron_outputs.values()), axis=0)
        else:
            # Tag-based selective routing: each neuron sends to specific target neurons
            target_neuron_inputs = {j: [] for j in range(len(next_layer.neurons))}
            
            for i in active_indices:
                neuron = self.neurons[i]
                output = neuron_outputs[i]
                
                # Update connections if needed
                if neuron.target_layer_id != id(next_layer):
                    neuron.update_target_connections(next_layer.neurons, id(next_layer))
                
                # Send output to target neurons
                for target_idx in neuron.target_neuron_indices:
                    if target_idx < len(next_layer.neurons):
                        target_neuron_inputs[target_idx].append(output)
            
            # Combine inputs for each target neuron (use shape of first output)
            combined_output = None
            active_targets = 0
            for target_idx, inputs in target_neuron_inputs.items():
                if len(inputs) > 0:
                    mean_input = np.mean(inputs, axis=0)
                    if combined_output is None:
                        combined_output = mean_input
                    else:
                        combined_output += mean_input
                    active_targets += 1
            
            if combined_output is None:
                combined_output = np.zeros(self.output_dim)
            elif active_targets > 0:
                combined_output /= active_targets
        
        self.layer_activations += 1
        return combined_output, active_indices
    
    def backward(self, grad_output: np.ndarray, active_indices: List[int], learning_rate: float = 0.01):
        """
        Backward pass - only updates active neurons.
        
        Args:
            grad_output: Gradient from next layer
            active_indices: Indices of neurons that were active
            learning_rate: Learning rate
            
        Returns:
            Average gradient w.r.t. input from active neurons
        """
        if len(active_indices) == 0:
            return np.zeros(self.input_dim)
        
        # Distribute gradient to active neurons
        grad_inputs = []
        for idx in active_indices:
            grad_input = self.neurons[idx].backward(grad_output, learning_rate)
            grad_inputs.append(grad_input)
        
        # Average gradients
        avg_grad_input = np.mean(grad_inputs, axis=0)
        return avg_grad_input
    
    def add_neuron(self, near_tag: Optional[np.ndarray] = None, is_broadcast: bool = False, parent_neuron: Optional['Neuron'] = None):
        """
        Add a new neuron to the layer.
        
        Args:
            near_tag: Place new neuron's tag near this tag (random if None)
            is_broadcast: Whether the new neuron should broadcast
            parent_neuron: Parent neuron to inherit weights from (None for random initialization)
        """
        if near_tag is not None:
            # Create tag near the specified tag with small random offset
            new_tag = near_tag + np.random.randn(self.tag_dim) * 0.1
            new_tag /= (np.linalg.norm(new_tag) + 1e-8)
        else:
            new_tag = None
        
        new_neuron = Neuron(
            self.input_dim,
            self.output_dim,
            functional_tag=new_tag,
            is_broadcast=is_broadcast,
            tag_dim=self.tag_dim
        )
        
        # If parent neuron provided, copy its weights with small perturbation
        if parent_neuron is not None:
            # Copy weights and add small random noise for variation
            new_neuron.weights = parent_neuron.weights.copy() + np.random.randn(*parent_neuron.weights.shape) * 0.01
            new_neuron.bias = parent_neuron.bias.copy() + np.random.randn(*parent_neuron.bias.shape) * 0.01
            # Copy tags with small variation
            new_neuron.functional_tag = parent_neuron.functional_tag.copy() + np.random.randn(self.tag_dim) * 0.05
            new_neuron.functional_tag /= (np.linalg.norm(new_neuron.functional_tag) + 1e-8)
            # Copy output neuron flag
            new_neuron.is_output_neuron = parent_neuron.is_output_neuron
        
        self.neurons.append(new_neuron)
    
    def remove_neuron(self, index: int):
        """
        Remove a neuron from the layer.
        
        Args:
            index: Index of neuron to remove
        """
        if 0 <= index < len(self.neurons):
            self.neurons.pop(index)
    
    def remove_underperforming_neurons(self, performance_threshold: float = -1.0) -> int:
        """
        Remove neurons that consistently underperform.
        
        Args:
            performance_threshold: Minimum acceptable average performance
            
        Returns:
            Number of neurons removed
        """
        indices_to_remove = []
        for i, neuron in enumerate(self.neurons):
            if neuron.is_underperforming(performance_threshold):
                indices_to_remove.append(i)
        
        # Remove in reverse order to maintain indices
        for i in sorted(indices_to_remove, reverse=True):
            self.remove_neuron(i)
        
        return len(indices_to_remove)
    
    def remove_rarely_active_neurons(self, min_activations: int = 5) -> int:
        """
        Remove neurons that rarely activate.
        
        Args:
            min_activations: Minimum number of activations required
            
        Returns:
            Number of neurons removed
        """
        indices_to_remove = []
        for i, neuron in enumerate(self.neurons):
            if neuron.is_rarely_active(min_activations):
                indices_to_remove.append(i)
        
        # Remove in reverse order
        for i in sorted(indices_to_remove, reverse=True):
            self.remove_neuron(i)
        
        return len(indices_to_remove)
    
    def get_struggling_neurons(self) -> List[Tuple[int, Neuron]]:
        """
        Get neurons that are struggling (active but poor performance).
        
        Returns:
            List of (index, neuron) tuples for struggling neurons
        """
        struggling = []
        for i, neuron in enumerate(self.neurons):
            if neuron.activation_count > 5 and neuron.get_average_performance() < 0:
                struggling.append((i, neuron))
        return struggling
    
    def reset_activation_counts(self):
        """Reset activation counters for all neurons."""
        for neuron in self.neurons:
            neuron.reset_activation_count()
    
    def record_layer_performance(self, reward: float):
        """
        Record layer-level performance.
        
        Args:
            reward: Performance reward
        """
        self.performance_history.append(reward)
        # Also propagate to active neurons
        for neuron in self.neurons:
            if neuron.was_active:
                neuron.record_performance(reward)
    
    def check_and_avoid_bad_configs(self) -> int:
        """
        Check if any active neurons are in bad configurations and perturb them.
        
        Returns:
            Number of neurons perturbed
        """
        perturbed = 0
        for neuron in self.neurons:
            if neuron.was_active and neuron.is_similar_to_bad_setting():
                # Perturb weights to escape bad configuration
                neuron.weights += np.random.randn(*neuron.weights.shape) * 0.05
                neuron.functional_tag += np.random.randn(*neuron.functional_tag.shape) * 0.05
                neuron.functional_tag /= (np.linalg.norm(neuron.functional_tag) + 1e-8)
                perturbed += 1
        return perturbed
    
    def rewire_neuron(self, 
                     neuron_idx: int, 
                     target_layer_id: int,
                     target_indices: List[int]):
        """
        Rewire a neuron to connect to different targets in another layer.
        
        This is a key modification strategy: instead of adding neurons,
        we can reconnect existing neurons to different parts of the network.
        
        Args:
            neuron_idx: Index of neuron in this layer to rewire
            target_layer_id: ID of target layer to connect to
            target_indices: List of target neuron indices in that layer
        """
        if 0 <= neuron_idx < len(self.neurons):
            neuron = self.neurons[neuron_idx]
            neuron.target_layer_id = target_layer_id
            neuron.target_neuron_indices = target_indices.copy()
    
    def get_layer_proximity_threshold(self, 
                                     current_layer_idx: int,
                                     target_layer_idx: int) -> float:
        """
        Get the tag similarity threshold based on layer distance.
        
        Treats the network as a physical system where closer layers
        connect more easily (lower threshold) and distant layers
        require stronger tag matching (higher threshold).
        
        Args:
            current_layer_idx: Index of current layer
            target_layer_idx: Index of target layer
            
        Returns:
            Threshold for tag similarity (higher = more selective)
        """
        distance = abs(target_layer_idx - current_layer_idx)
        
        if distance == 1:  # Adjacent layers
            return 0.3
        elif distance == 2:  # Skip one layer
            return 0.5
        elif distance == 3:  # Skip two layers
            return 0.7
        else:  # Too far - max threshold
            return 0.9
    
    def get_layer_stats(self) -> Dict:
        """Get statistics about the layer."""
        total_neurons = len(self.neurons)
        broadcast_neurons = sum(1 for n in self.neurons if n.is_broadcast)
        avg_activations = np.mean([n.activation_count for n in self.neurons]) if total_neurons > 0 else 0
        avg_performance = np.mean([n.get_average_performance() for n in self.neurons]) if total_neurons > 0 else 0
        
        return {
            'total_neurons': total_neurons,
            'broadcast_neurons': broadcast_neurons,
            'selective_neurons': total_neurons - broadcast_neurons,
            'avg_activations': avg_activations,
            'avg_performance': avg_performance,
            'layer_activations': self.layer_activations
        }
    
    def get_neuron_count(self) -> int:
        """Get the number of neurons in this layer."""
        return len(self.neurons)
    
    def update_layer_tag(self, tag_gradient: np.ndarray, learning_rate: float = 0.001):
        """
        Update the layer's functional tag.
        
        Args:
            tag_gradient: Gradient for layer tag
            learning_rate: Learning rate
        """
        self.layer_tag += learning_rate * tag_gradient
        self.layer_tag /= (np.linalg.norm(self.layer_tag) + 1e-8)
    
    def adjust_size(self, target_size: int):
        """
        Adjust layer size to target (add or remove neurons as needed).
        
        Args:
            target_size: Desired number of neurons
        """
        current_size = len(self.neurons)
        
        if target_size > current_size:
            # Add neurons
            for _ in range(target_size - current_size):
                is_broadcast = (np.random.rand() < self.broadcast_ratio)
                self.add_neuron(is_broadcast=is_broadcast)
        elif target_size < current_size:
            # Remove least active neurons
            activations = [(i, n.activation_count) for i, n in enumerate(self.neurons)]
            activations.sort(key=lambda x: x[1])  # Sort by activation count
            
            num_to_remove = current_size - target_size
            for i in range(num_to_remove):
                if i < len(activations):
                    self.remove_neuron(activations[i][0])
    
    def __repr__(self):
        stats = self.get_layer_stats()
        return f"AdaptiveLayer(neurons={stats['total_neurons']}, broadcast={stats['broadcast_neurons']}, avg_perf={stats['avg_performance']:.3f})"
