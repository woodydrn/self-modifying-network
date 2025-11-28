import numpy as np
from collections import deque
from typing import List, Dict, Optional, Tuple


class Neuron:
    """
    A neuron with functional tags, selective connectivity, and performance memory.
    
    Attributes:
        functional_tag: Vector describing the neuron's role/identity (4D)
        input_tags: List of vectors describing what inputs this neuron responds to (4D each)
        output_tags: List of vectors describing where this neuron sends outputs (4D each)
        weights: Connection weights to target neurons
        bias: Bias term
        is_broadcast: Whether this neuron connects to all neurons or selectively
        activation_count: Number of times this neuron has activated
        performance_history: Recent performance scores
        bad_settings: Memory of parameter configurations that performed poorly
        activation_threshold: Learned threshold for tag matching
    """
    
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 functional_tag: Optional[np.ndarray] = None,
                 is_broadcast: bool = False,
                 tag_dim: int = 4):
        """
        Initialize a neuron.
        
        Args:
            input_dim: Dimension of input
            output_dim: Dimension of output
            functional_tag: Initial functional tag (auto-generated if None)
            is_broadcast: Whether neuron broadcasts to all or selectively connects
            tag_dim: Dimension of tag vectors
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.tag_dim = tag_dim
        self.is_broadcast = is_broadcast
        
        # Initialize tags
        if functional_tag is None:
            self.functional_tag = np.random.randn(tag_dim)
            self.functional_tag /= np.linalg.norm(self.functional_tag) + 1e-8
        else:
            self.functional_tag = functional_tag
            
        # Input/output routing tags
        self.input_tags = [self._random_unit_vector(tag_dim) for _ in range(3)]
        self.output_tags = [self._random_unit_vector(tag_dim) for _ in range(3)]
        
        # Neural parameters - simple initialization
        scale = np.sqrt(2.0 / (input_dim + output_dim))
        self.weights = np.random.randn(input_dim, output_dim) * scale
        self.bias = np.zeros(output_dim)
        
        # Learning parameters - much more aggressive
        self.activation_threshold = 0.5  # Tag similarity threshold for selective activation
        self.learning_rate = 0.1  # Much higher learning rate for actual learning
        
        # Performance tracking
        self.activation_count = 0
        self.performance_history = deque(maxlen=100)
        self.bad_settings = []  # List of (weights, bias, performance) tuples
        
        # Selective connectivity - which target neuron indices this neuron connects to
        self.target_neuron_indices: List[int] = []  # Empty means broadcast to all
        self.target_layer_id: Optional[int] = None  # Which layer to send to
        
        # Gradients (for backprop)
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)
        self.grad_functional_tag = np.zeros_like(self.functional_tag)
        
        # Cache for forward pass
        self.last_input = None
        self.last_output = None
        self.was_active = False
        
        # Output neuron flag (set by network during initialization)
        self.is_output_neuron = False
        
    def _random_unit_vector(self, dim: int) -> np.ndarray:
        """Generate a random unit vector."""
        vec = np.random.randn(dim)
        return vec / (np.linalg.norm(vec) + 1e-8)
    
    def compute_tag_similarity(self, input_tag: np.ndarray) -> float:
        """
        Compute similarity between input tag and neuron's functional tag.
        
        Args:
            input_tag: Tag vector from input
            
        Returns:
            Cosine similarity score [0, 1]
        """
        # Normalize both vectors
        norm_input = input_tag / (np.linalg.norm(input_tag) + 1e-8)
        norm_functional = self.functional_tag / (np.linalg.norm(self.functional_tag) + 1e-8)
        
        # Cosine similarity (convert to [0, 1] range)
        similarity = np.dot(norm_input, norm_functional)
        return (similarity + 1.0) / 2.0
    
    def should_activate(self, input_tag: np.ndarray) -> bool:
        """
        Determine if this neuron should activate based on tag matching.
        
        Args:
            input_tag: Tag vector from input
            
        Returns:
            True if neuron should activate, False otherwise
        """
        if self.is_broadcast:
            return True  # Broadcast neurons always activate
        
        similarity = self.compute_tag_similarity(input_tag)
        return similarity >= self.activation_threshold
    
    def forward(self, x: np.ndarray, input_tag: np.ndarray) -> Optional[np.ndarray]:
        """
        Forward pass through the neuron.
        
        Args:
            x: Input vector
            input_tag: Tag vector from input
            
        Returns:
            Output vector if activated, None if silent
        """
        # Tag-based selective activation - only activate if tag similarity is high enough
        if not self.should_activate(input_tag):
            self.was_active = False
            return None
        
        # Neuron activates
        self.was_active = True
        self.activation_count += 1
        self.last_input = x.copy()
        
        # Compute output
        linear_output = np.dot(x, self.weights) + self.bias
        
        # Use linear activation for output neurons
        # Use tanh for hidden neurons (bounded but smooth gradients)
        if hasattr(self, 'is_output_neuron') and self.is_output_neuron:
            self.last_output = linear_output  # Linear for output
        else:
            self.last_output = np.tanh(linear_output)  # Tanh for hidden - bounded and smooth
        
        return self.last_output
    
    def backward(self, grad_output: np.ndarray, learning_rate: Optional[float] = None) -> np.ndarray:
        """
        Backward pass - only updates if neuron was active.
        
        Args:
            grad_output: Gradient from next layer
            learning_rate: Override learning rate (uses self.learning_rate if None)
            
        Returns:
            Gradient w.r.t. input
        """
        if not self.was_active or self.last_input is None:
            return np.zeros(self.input_dim)
        
        lr = learning_rate if learning_rate is not None else self.learning_rate
        
        # Compute gradient through activation function
        if self.is_output_neuron:
            # Linear activation: gradient passes through unchanged
            grad_activated = grad_output
        else:
            # Tanh gradient: (1 - tanh^2)
            grad_activated = grad_output * (1 - self.last_output ** 2)
        
        # Clip gradients moderately
        grad_activated = np.clip(grad_activated, -5.0, 5.0)
        
        # Compute gradients
        self.grad_weights = np.outer(self.last_input, grad_activated)
        self.grad_bias = grad_activated
        
        # Clip gradients moderately
        self.grad_weights = np.clip(self.grad_weights, -5.0, 5.0)
        self.grad_bias = np.clip(self.grad_bias, -5.0, 5.0)
        
        # Update parameters (GRADIENT DESCENT: subtract gradient!)
        self.weights -= lr * self.grad_weights
        self.bias -= lr * self.grad_bias
        
        # Moderate weight clipping
        self.weights = np.clip(self.weights, -20.0, 20.0)
        self.bias = np.clip(self.bias, -20.0, 20.0)
        
        # Gradient w.r.t. input
        grad_input = np.dot(self.weights, grad_activated)
        
        return grad_input
    
    def update_tag(self, tag_gradient: np.ndarray, learning_rate: float = 0.001):
        """
        Update functional tag based on gradient.
        
        Args:
            tag_gradient: Gradient for the functional tag
            learning_rate: Learning rate for tag update
        """
        if not self.was_active:
            return
        
        self.functional_tag += learning_rate * tag_gradient
        # Re-normalize to unit vector
        self.functional_tag /= (np.linalg.norm(self.functional_tag) + 1e-8)
    
    def record_performance(self, reward: float):
        """
        Record performance score for this neuron.
        
        Args:
            reward: Performance reward value
        """
        self.performance_history.append(reward)
        
        # If performance is very bad, save current settings as "bad"
        if reward < -5.0 and len(self.bad_settings) < 50:
            bad_config = {
                'weights': self.weights.copy(),
                'bias': self.bias.copy(),
                'functional_tag': self.functional_tag.copy(),
                'reward': reward
            }
            self.bad_settings.append(bad_config)
    
    def get_average_performance(self) -> float:
        """Get average performance over recent history."""
        if len(self.performance_history) == 0:
            return 0.0
        return np.mean(self.performance_history)
    
    def is_underperforming(self, threshold: float = -1.0) -> bool:
        """Check if neuron is consistently underperforming."""
        if len(self.performance_history) < 10:
            return False
        return self.get_average_performance() < threshold
    
    def is_rarely_active(self, min_activations: int = 5) -> bool:
        """Check if neuron rarely activates."""
        return self.activation_count < min_activations
    
    def reset_activation_count(self):
        """Reset activation counter (useful for periodic checks)."""
        self.activation_count = 0
    
    def is_similar_to_bad_setting(self, tolerance: float = 0.1) -> bool:
        """
        Check if current parameters are similar to known bad settings.
        
        Args:
            tolerance: Similarity tolerance
            
        Returns:
            True if similar to a bad setting
        """
        if len(self.bad_settings) == 0:
            return False
            
        for bad_config in self.bad_settings:
            # Normalized difference for weights
            weight_norm = np.linalg.norm(self.weights)
            bad_weight_norm = np.linalg.norm(bad_config['weights'])
            if weight_norm > 0 and bad_weight_norm > 0:
                weight_similarity = np.dot(self.weights.flatten(), bad_config['weights'].flatten()) / (weight_norm * bad_weight_norm)
            else:
                weight_similarity = 0
            
            # Tag similarity
            tag_similarity = np.dot(self.functional_tag, bad_config['functional_tag'])
            
            # If both are highly similar (>0.9), this is similar to bad setting
            if weight_similarity > 0.9 and tag_similarity > 0.9:
                return True
        return False
    
    def update_target_connections(self, target_layer_neurons: List['Neuron'], layer_id: int):
        """
        Update which neurons in the target layer this neuron connects to based on tag matching.
        
        Args:
            target_layer_neurons: List of neurons in the next layer
            layer_id: ID of the target layer
        """
        if self.is_broadcast:
            # Broadcast neurons connect to all
            self.target_neuron_indices = list(range(len(target_layer_neurons)))
        else:
            # Selective neurons only connect to matching tags
            self.target_neuron_indices = []
            for idx, target_neuron in enumerate(target_layer_neurons):
                # Check if this neuron's output tags match target's input tags
                max_similarity = 0
                for out_tag in self.output_tags:
                    for in_tag in target_neuron.input_tags:
                        similarity = np.dot(out_tag, in_tag) / (np.linalg.norm(out_tag) * np.linalg.norm(in_tag) + 1e-8)
                        max_similarity = max(max_similarity, similarity)
                
                # Connect if similarity is high enough (threshold 0.3)
                if max_similarity > 0.3:
                    self.target_neuron_indices.append(idx)
            
            # If no connections found, connect to at least one random neuron
            if len(self.target_neuron_indices) == 0:
                self.target_neuron_indices = [np.random.randint(0, len(target_layer_neurons))]
        
        self.target_layer_id = layer_id
    
    def get_state(self) -> Dict:
        """Get neuron state for serialization."""
        return {
            'functional_tag': self.functional_tag,
            'input_tags': self.input_tags,
            'output_tags': self.output_tags,
            'weights': self.weights,
            'bias': self.bias,
            'is_broadcast': self.is_broadcast,
            'activation_threshold': self.activation_threshold,
            'activation_count': self.activation_count,
            'performance_history': list(self.performance_history)
        }
    
    def __repr__(self):
        status = "broadcast" if self.is_broadcast else "selective"
        avg_perf = self.get_average_performance()
        return f"Neuron({status}, activations={self.activation_count}, avg_perf={avg_perf:.3f})"
