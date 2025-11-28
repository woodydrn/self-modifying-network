import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from .layer import AdaptiveLayer


class IntelligentBackward:
    """
    Intelligent backward pass that assigns functional tags to neurons/layers
    based on their discovered roles and remembers bad update patterns.
    
    Attributes:
        role_library: Library of discovered functional roles
        gradient_patterns: Patterns of gradient flow for different roles
        bad_update_memory: Memory of updates that led to poor performance
    """
    
    # Predefined role tags for common neuron functions
    ROLE_TAGS = {
        'input_processor': np.array([1.0, 0.0, 0.0, 0.0]),
        'feature_detector': np.array([0.0, 1.0, 0.0, 0.0]),
        'pattern_combiner': np.array([0.0, 0.0, 1.0, 0.0]),
        'output_integrator': np.array([0.0, 0.0, 0.0, 1.0]),
        'suppressor': np.array([0.7, 0.3, 0.0, 0.0]),
        'amplifier': np.array([0.3, 0.7, 0.0, 0.0]),
        'nonlinear_transformer': np.array([0.0, 0.5, 0.5, 0.0]),
        'gate': np.array([0.5, 0.0, 0.5, 0.0])
    }
    
    def __init__(self, tag_dim: int = 4):
        """
        Initialize intelligent backward pass.
        
        Args:
            tag_dim: Dimension of tag vectors
        """
        self.tag_dim = tag_dim
        
        # Role discovery
        self.role_library: Dict[str, np.ndarray] = self.ROLE_TAGS.copy()
        self.neuron_roles: Dict[int, str] = {}  # Maps neuron ID to role name
        self.layer_roles: Dict[int, str] = {}  # Maps layer ID to role name
        
        # Gradient pattern analysis
        self.gradient_patterns: Dict[str, List[float]] = defaultdict(list)
        
        # Bad update memory
        self.bad_updates = []  # List of (layer_id, neuron_id, update_vector, reward)
        self.max_bad_memory = 200
        
    def analyze_gradient_flow(self,
                               layer: AdaptiveLayer,
                               layer_id: int,
                               grad_output: np.ndarray,
                               active_indices: List[int]) -> Dict[int, str]:
        """
        Analyze gradient flow patterns to identify neuron roles.
        
        Args:
            layer: The layer being analyzed
            layer_id: ID of the layer
            grad_output: Gradient flowing to this layer
            active_indices: Indices of active neurons
            
        Returns:
            Dictionary mapping neuron indices to role names
        """
        roles = {}
        
        for idx in active_indices:
            neuron = layer.neurons[idx]
            
            if neuron.grad_weights is None or neuron.last_input is None:
                continue
            
            # Analyze gradient characteristics
            grad_magnitude = np.linalg.norm(neuron.grad_weights)
            grad_sparsity = np.sum(np.abs(neuron.grad_weights) < 1e-3) / neuron.grad_weights.size
            
            # Analyze input/output relationship
            input_magnitude = np.linalg.norm(neuron.last_input)
            output_magnitude = np.linalg.norm(neuron.last_output) if neuron.last_output is not None else 0
            
            # Classify role based on patterns
            role = self._classify_neuron_role(
                grad_magnitude=grad_magnitude,
                grad_sparsity=grad_sparsity,
                input_magnitude=input_magnitude,
                output_magnitude=output_magnitude,
                activation_count=neuron.activation_count
            )
            
            roles[idx] = role
            
            # Update neuron's functional tag toward the role tag
            if role in self.role_library:
                role_tag = self.role_library[role]
                # Move neuron's tag slightly toward the role tag
                tag_update = (role_tag - neuron.functional_tag) * 0.1
                neuron.update_tag(tag_update, learning_rate=0.05)
        
        return roles
    
    def _classify_neuron_role(self,
                              grad_magnitude: float,
                              grad_sparsity: float,
                              input_magnitude: float,
                              output_magnitude: float,
                              activation_count: int) -> str:
        """
        Classify neuron's functional role based on characteristics.
        
        Args:
            grad_magnitude: Magnitude of gradients
            grad_sparsity: Sparsity of gradient (fraction of near-zero values)
            input_magnitude: Magnitude of input
            output_magnitude: Magnitude of output
            activation_count: Number of times activated
            
        Returns:
            Role name
        """
        # Input processors: high activation, steady gradients
        if activation_count > 100 and grad_sparsity < 0.3:
            return 'input_processor'
        
        # Feature detectors: selective activation, sparse gradients
        if grad_sparsity > 0.6 and activation_count < 50:
            return 'feature_detector'
        
        # Pattern combiners: moderate everything
        if 0.3 <= grad_sparsity <= 0.6 and 50 <= activation_count <= 100:
            return 'pattern_combiner'
        
        # Output integrators: always active, dense gradients
        if activation_count > 150 and grad_sparsity < 0.2:
            return 'output_integrator'
        
        # Suppressors: small output relative to input
        if output_magnitude < input_magnitude * 0.3:
            return 'suppressor'
        
        # Amplifiers: large output relative to input
        if output_magnitude > input_magnitude * 1.5:
            return 'amplifier'
        
        # Nonlinear transformers: high gradient magnitude
        if grad_magnitude > 1.0:
            return 'nonlinear_transformer'
        
        # Gates: very selective activation
        if grad_sparsity > 0.8:
            return 'gate'
        
        # Default
        return 'feature_detector'
    
    def backward_with_role_assignment(self,
                                       layers: List[AdaptiveLayer],
                                       grad_output: np.ndarray,
                                       active_indices_per_layer: List[List[int]],
                                       learning_rate: float = 0.01) -> List[np.ndarray]:
        """
        Perform backward pass with intelligent role assignment.
        
        Args:
            layers: List of layers (in forward order)
            grad_output: Initial gradient from loss
            active_indices_per_layer: Active neuron indices for each layer
            learning_rate: Learning rate
            
        Returns:
            List of gradients for each layer
        """
        gradients = []
        current_grad = grad_output
        
        # Backward through layers (reverse order)
        for layer_id in range(len(layers) - 1, -1, -1):
            layer = layers[layer_id]
            active_indices = active_indices_per_layer[layer_id]
            
            # Analyze gradient flow and assign roles
            roles = self.analyze_gradient_flow(layer, layer_id, current_grad, active_indices)
            
            # Perform backward pass
            grad_input = layer.backward(current_grad, active_indices, learning_rate)
            gradients.insert(0, grad_input)
            
            # Update current gradient for next layer
            current_grad = grad_input
            
            # Store role information
            for idx, role in roles.items():
                neuron_global_id = f"{layer_id}_{idx}"
                self.neuron_roles[neuron_global_id] = role
        
        return gradients
    
    def remember_bad_update(self,
                            layer_id: int,
                            neuron_id: int,
                            weights_before: np.ndarray,
                            weights_after: np.ndarray,
                            reward: float):
        """
        Remember an update that led to poor performance.
        
        Args:
            layer_id: Layer index
            neuron_id: Neuron index within layer
            weights_before: Weights before update
            weights_after: Weights after update
            reward: Resulting reward
        """
        if reward < -2.0:  # Only remember significantly bad updates
            update_vector = weights_after - weights_before
            
            bad_update = {
                'layer_id': layer_id,
                'neuron_id': neuron_id,
                'update_vector': update_vector.copy(),
                'reward': reward
            }
            
            self.bad_updates.append(bad_update)
            
            # Limit memory size
            if len(self.bad_updates) > self.max_bad_memory:
                self.bad_updates.pop(0)
    
    def is_similar_to_bad_update(self,
                                  layer_id: int,
                                  neuron_id: int,
                                  proposed_update: np.ndarray,
                                  similarity_threshold: float = 0.8) -> bool:
        """
        Check if proposed update is similar to a known bad update.
        
        Args:
            layer_id: Layer index
            neuron_id: Neuron index
            proposed_update: Proposed update vector
            similarity_threshold: Cosine similarity threshold
            
        Returns:
            True if similar to a bad update
        """
        for bad in self.bad_updates:
            if bad['layer_id'] == layer_id and bad['neuron_id'] == neuron_id:
                # Compute cosine similarity
                norm_proposed = proposed_update / (np.linalg.norm(proposed_update) + 1e-8)
                norm_bad = bad['update_vector'] / (np.linalg.norm(bad['update_vector']) + 1e-8)
                
                similarity = np.dot(norm_proposed.flatten(), norm_bad.flatten())
                
                if similarity > similarity_threshold:
                    return True
        
        return False
    
    def suggest_role_based_learning_rate(self, role: str, base_lr: float = 0.01) -> float:
        """
        Suggest learning rate based on neuron's role.
        
        Args:
            role: Neuron's functional role
            base_lr: Base learning rate
            
        Returns:
            Adjusted learning rate
        """
        # Different roles learn at different rates
        role_multipliers = {
            'input_processor': 0.5,      # Slower - foundational
            'feature_detector': 1.0,     # Normal
            'pattern_combiner': 1.2,     # Slightly faster
            'output_integrator': 0.8,    # Slower - affects final output directly
            'suppressor': 0.7,           # Slower - can destabilize
            'amplifier': 0.7,            # Slower - can destabilize
            'nonlinear_transformer': 1.5,  # Faster - needs to adapt quickly
            'gate': 1.0                  # Normal
        }
        
        multiplier = role_multipliers.get(role, 1.0)
        return base_lr * multiplier
    
    def get_layer_role_distribution(self, layer_id: int) -> Dict[str, int]:
        """
        Get distribution of roles within a layer.
        
        Args:
            layer_id: Layer index
            
        Returns:
            Dictionary of role counts
        """
        distribution = defaultdict(int)
        
        for neuron_id, role in self.neuron_roles.items():
            if neuron_id.startswith(f"{layer_id}_"):
                distribution[role] += 1
        
        return dict(distribution)
    
    def assign_layer_role(self, layer_id: int) -> str:
        """
        Assign an overall role to a layer based on its neurons.
        
        Args:
            layer_id: Layer index
            
        Returns:
            Layer role name
        """
        role_dist = self.get_layer_role_distribution(layer_id)
        
        if not role_dist:
            return 'unknown'
        
        # Layer takes the role of its most common neuron type
        dominant_role = max(role_dist.items(), key=lambda x: x[1])[0]
        self.layer_roles[layer_id] = dominant_role
        
        return dominant_role
    
    def get_statistics(self) -> Dict:
        """Get statistics about role assignments."""
        return {
            'total_neurons_classified': len(self.neuron_roles),
            'total_layers_classified': len(self.layer_roles),
            'bad_updates_remembered': len(self.bad_updates),
            'role_distribution': self._get_overall_role_distribution()
        }
    
    def _get_overall_role_distribution(self) -> Dict[str, int]:
        """Get overall distribution of roles across all neurons."""
        distribution = defaultdict(int)
        for role in self.neuron_roles.values():
            distribution[role] += 1
        return dict(distribution)
    
    def __repr__(self):
        stats = self.get_statistics()
        return f"IntelligentBackward(neurons={stats['total_neurons_classified']}, bad_updates={stats['bad_updates_remembered']})"
