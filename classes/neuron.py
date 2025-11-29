import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
from typing import List, Dict, Optional, Tuple

# Check for GPU availability
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LinkOnlyLinear(torch.autograd.Function):
    """
    Custom autograd function that only backpropagates through weights/bias,
    blocking gradient flow to inputs (preserving link-only learning).
    """
    @staticmethod
    def forward(ctx, x, weight, bias):
        ctx.save_for_backward(x, weight)
        return x @ weight + bias
    
    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        # Gradient for weight: x.T @ grad_output
        if x.dim() == 2:
            grad_weight = x.T @ grad_output
        else:
            grad_weight = torch.outer(x, grad_output.squeeze())
        # Gradient for bias
        grad_bias = grad_output.sum(0) if grad_output.dim() == 2 else grad_output
        # Return None for input gradient - blocks backprop through activations
        return None, grad_weight, grad_bias


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
                 tag_dim: int = 4,
                 device: torch.device = None):
        """
        Initialize a neuron.
        
        Args:
            input_dim: Dimension of input
            output_dim: Dimension of output
            functional_tag: Initial functional tag (auto-generated if None)
            is_broadcast: Whether neuron broadcasts to all or selectively connects
            tag_dim: Dimension of tag vectors
            device: PyTorch device (GPU/CPU)
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.tag_dim = tag_dim
        self.is_broadcast = is_broadcast
        self.device = device if device is not None else DEVICE
        
        # Initialize tags (keep as numpy - not learned via backprop)
        if functional_tag is None:
            self.functional_tag = np.random.randn(tag_dim)
            self.functional_tag /= np.linalg.norm(self.functional_tag) + 1e-8
        else:
            self.functional_tag = functional_tag
            
        # Input/output routing tags
        self.input_tags = [self._random_unit_vector(tag_dim) for _ in range(3)]
        self.output_tags = [self._random_unit_vector(tag_dim) for _ in range(3)]
        
        # Neural parameters as PyTorch tensors for GPU acceleration
        scale = np.sqrt(2.0 / (input_dim + output_dim))
        self.weights = torch.nn.Parameter(
            torch.randn(input_dim, output_dim, device=self.device) * scale
        )
        self.bias = torch.nn.Parameter(
            torch.zeros(output_dim, device=self.device)
        )
        
        # Learning parameters - much more aggressive
        self.activation_threshold = 0.5  # Tag similarity threshold for selective activation
        self.learning_rate = 0.1  # Much higher learning rate for actual learning
        
        # Performance tracking
        self.activation_count = 0
        self.performance_history = deque(maxlen=100)
        self.bad_settings = []  # List of compressed bad configs
        
        # Selective connectivity - which target neuron indices this neuron connects to
        self.target_neuron_indices: List[int] = []  # Empty means broadcast to all
        self.target_layer_id: Optional[int] = None  # Which layer to send to
        
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
    
    def forward(self, x: np.ndarray, input_tag: np.ndarray) -> Optional[torch.Tensor]:
        """
        Forward pass through the neuron using PyTorch with link-only gradients.
        
        Args:
            x: Input vector (numpy array)
            input_tag: Tag vector from input
            
        Returns:
            Output tensor if activated, None if silent
        """
        # Tag-based selective activation - only activate if tag similarity is high enough
        if not self.should_activate(input_tag):
            self.was_active = False
            return None
        
        # Neuron activates
        self.was_active = True
        self.activation_count += 1
        
        # Convert to tensor for GPU computation
        x_tensor = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        self.last_input = x_tensor
        
        # Use custom autograd function for link-only backprop
        linear_output = LinkOnlyLinear.apply(x_tensor, self.weights, self.bias)
        
        # Use linear activation for output neurons
        # Use tanh for hidden neurons (bounded but smooth gradients)
        if hasattr(self, 'is_output_neuron') and self.is_output_neuron:
            self.last_output = linear_output  # Linear for output
        else:
            self.last_output = torch.tanh(linear_output)  # Tanh for hidden - bounded and smooth
        
        return self.last_output
    
    def backward(self, grad_output, learning_rate: Optional[float] = None):
        """
        Backward pass - only updates if neuron was active.
        Uses PyTorch tensors for GPU acceleration.
        
        Args:
            grad_output: Gradient from next layer (numpy array or torch tensor)
            learning_rate: Override learning rate (uses self.learning_rate if None)
            
        Returns:
            Gradient w.r.t. input (numpy array for compatibility)
        """
        if not self.was_active or self.last_input is None:
            return np.zeros(self.input_dim)
        
        lr = learning_rate if learning_rate is not None else self.learning_rate
        
        # Convert grad_output to tensor if needed
        if isinstance(grad_output, np.ndarray):
            grad_output = torch.as_tensor(grad_output, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            # Compute gradient through activation function
            if self.is_output_neuron:
                # Linear activation: gradient passes through unchanged
                grad_activated = grad_output
            else:
                # Tanh gradient: (1 - tanh^2)
                grad_activated = grad_output * (1 - self.last_output ** 2)
            
            # Clip gradients moderately
            grad_activated = torch.clamp(grad_activated, -5.0, 5.0)
            
            # Compute gradients
            if self.last_input.dim() == 1:
                grad_weights = torch.outer(self.last_input, grad_activated)
            else:
                grad_weights = self.last_input.T @ grad_activated
            grad_bias = grad_activated
            
            # Clip gradients moderately
            grad_weights = torch.clamp(grad_weights, -5.0, 5.0)
            grad_bias = torch.clamp(grad_bias, -5.0, 5.0)
            
            # Update parameters (GRADIENT DESCENT: subtract gradient!)
            self.weights.data -= lr * grad_weights
            self.bias.data -= lr * grad_bias.squeeze() if grad_bias.dim() > 0 else lr * grad_bias
            
            # Moderate weight clipping
            self.weights.data.clamp_(-20.0, 20.0)
            self.bias.data.clamp_(-20.0, 20.0)
            
            # Gradient w.r.t. input
            grad_input = self.weights @ grad_activated
        
        return grad_input.cpu().numpy() if isinstance(grad_input, torch.Tensor) else grad_input
    
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
        
        # If performance is very bad, save compressed config (not full weights)
        if reward < -5.0 and len(self.bad_settings) < 50:
            bad_config = {
                'weight_norm': self.weights.data.norm().item(),
                'bias_norm': self.bias.data.norm().item(),
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
        Uses compressed comparison for efficiency.
        
        Args:
            tolerance: Similarity tolerance
            
        Returns:
            True if similar to a bad setting
        """
        if len(self.bad_settings) == 0:
            return False
        
        current_weight_norm = self.weights.data.norm().item()
        current_bias_norm = self.bias.data.norm().item()
        
        for bad_config in self.bad_settings:
            # Compare norms (fast approximate comparison)
            weight_norm_ratio = current_weight_norm / (bad_config['weight_norm'] + 1e-8)
            bias_norm_ratio = current_bias_norm / (bad_config['bias_norm'] + 1e-8)
            
            # Tag similarity
            tag_similarity = np.dot(self.functional_tag, bad_config['functional_tag'])
            
            # If norms are similar and tag is similar, likely bad setting
            if 0.9 < weight_norm_ratio < 1.1 and 0.9 < bias_norm_ratio < 1.1 and tag_similarity > 0.9:
                return True
        return False
    
    def update_target_connections(self, target_layer_neurons: List['Neuron'], layer_id: int):
        """
        Update which neurons in the target layer this neuron connects to based on tag matching.
        Vectorized implementation for better performance.
        
        Args:
            target_layer_neurons: List of neurons in the next layer
            layer_id: ID of the target layer
        """
        if self.is_broadcast:
            # Broadcast neurons connect to all
            self.target_neuron_indices = list(range(len(target_layer_neurons)))
        else:
            # Vectorized tag similarity computation
            out_tags = np.array(self.output_tags)  # (3, tag_dim)
            out_norms = np.linalg.norm(out_tags, axis=1, keepdims=True) + 1e-8
            out_tags_norm = out_tags / out_norms
            
            self.target_neuron_indices = []
            for idx, target_neuron in enumerate(target_layer_neurons):
                in_tags = np.array(target_neuron.input_tags)  # (3, tag_dim)
                in_norms = np.linalg.norm(in_tags, axis=1, keepdims=True) + 1e-8
                in_tags_norm = in_tags / in_norms
                
                # Batch similarity: (3, tag_dim) @ (tag_dim, 3) = (3, 3)
                similarities = out_tags_norm @ in_tags_norm.T
                max_sim = np.max(similarities)
                
                # Connect if similarity is high enough (threshold 0.3)
                if max_sim > 0.3:
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
            'weights': self.weights.data.cpu().numpy(),
            'bias': self.bias.data.cpu().numpy(),
            'is_broadcast': self.is_broadcast,
            'activation_threshold': self.activation_threshold,
            'activation_count': self.activation_count,
            'performance_history': list(self.performance_history)
        }
    
    def __repr__(self):
        status = "broadcast" if self.is_broadcast else "selective"
        avg_perf = self.get_average_performance()
        return f"Neuron({status}, activations={self.activation_count}, avg_perf={avg_perf:.3f})"
