import numpy as np
from typing import Dict, List, Tuple, Optional
from .modification_tracker import ModificationType


class MetaLearner:
    """
    A small neural network that learns to predict modification success.
    
    Uses modification history to learn which structural changes work
    in which network states. This enables intelligent strategy selection.
    
    Architecture: 2-layer MLP with 32 hidden units
    Input: Network state + proposed modification type (14 features)
    Output: Success probability (0-1)
    """
    
    def __init__(self, 
                 input_dim: int = 14,
                 hidden_dim: int = 32,
                 learning_rate: float = 0.01):
        """
        Initialize the meta-learner.
        
        Args:
            input_dim: Number of input features (default 14)
            hidden_dim: Number of hidden units (default 32)
            learning_rate: Learning rate for training
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        
        # Initialize weights with Xavier initialization
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        
        self.W2 = np.random.randn(hidden_dim, 1) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(1)
        
        # Training statistics
        self.training_losses: List[float] = []
        self.training_accuracies: List[float] = []
        self.total_updates = 0
        
    def predict(self, features: np.ndarray) -> float:
        """
        Predict success probability for a modification.
        
        Args:
            features: Feature vector (14 dimensions)
            
        Returns:
            Success probability (0-1)
        """
        # Ensure features is 1D
        if features.ndim == 2:
            features = features.flatten()
            
        # Forward pass
        h = self._relu(features @ self.W1 + self.b1)
        logit = h @ self.W2 + self.b2
        prob = self._sigmoid(logit[0])
        
        return prob
        
    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Predict success probabilities for multiple modifications.
        
        Args:
            X: Feature matrix (n_samples, 14)
            
        Returns:
            Success probabilities (n_samples,)
        """
        if X.shape[0] == 0:
            return np.array([])
            
        # Forward pass
        h = self._relu(X @ self.W1 + self.b1)  # (n, hidden_dim)
        logits = h @ self.W2 + self.b2          # (n, 1)
        probs = self._sigmoid(logits.flatten())  # (n,)
        
        return probs
        
    def train_step(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        Perform one training step.
        
        Args:
            X: Feature matrix (n_samples, 14)
            y: Target labels (n_samples,) - 1.0 for success, 0.0 for failure
            
        Returns:
            (loss, accuracy) tuple
        """
        if X.shape[0] == 0:
            return 0.0, 0.5
            
        n = X.shape[0]
        
        # Forward pass
        h = self._relu(X @ self.W1 + self.b1)  # (n, hidden_dim)
        logits = h @ self.W2 + self.b2          # (n, 1)
        probs = self._sigmoid(logits.flatten())  # (n,)
        
        # Compute loss (binary cross-entropy)
        eps = 1e-8  # Numerical stability
        loss = -np.mean(y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps))
        
        # Compute accuracy
        predictions = (probs >= 0.5).astype(float)
        accuracy = np.mean(predictions == y)
        
        # Backward pass
        # d_loss/d_logit = prob - y
        d_logit = (probs - y).reshape(-1, 1)  # (n, 1)
        
        # Gradients for W2 and b2
        d_W2 = (h.T @ d_logit) / n  # (hidden_dim, 1)
        d_b2 = np.mean(d_logit, axis=0)  # (1,)
        
        # Gradient for hidden layer
        d_h = d_logit @ self.W2.T  # (n, hidden_dim)
        d_h_relu = d_h * (h > 0)  # ReLU gradient
        
        # Gradients for W1 and b1
        d_W1 = (X.T @ d_h_relu) / n  # (input_dim, hidden_dim)
        d_b1 = np.mean(d_h_relu, axis=0)  # (hidden_dim,)
        
        # Update weights
        self.W1 -= self.learning_rate * d_W1
        self.b1 -= self.learning_rate * d_b1
        self.W2 -= self.learning_rate * d_W2
        self.b2 -= self.learning_rate * d_b2
        
        # Track statistics
        self.training_losses.append(loss)
        self.training_accuracies.append(accuracy)
        self.total_updates += 1
        
        return loss, accuracy
        
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 10, batch_size: int = 32):
        """
        Train the meta-learner on modification history.
        
        Args:
            X: Feature matrix (n_samples, 14)
            y: Target labels (n_samples,)
            epochs: Number of training epochs
            batch_size: Batch size for mini-batch training
        """
        if X.shape[0] == 0:
            return
            
        n = X.shape[0]
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Mini-batch training
            epoch_losses = []
            epoch_accs = []
            
            for i in range(0, n, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                loss, acc = self.train_step(X_batch, y_batch)
                epoch_losses.append(loss)
                epoch_accs.append(acc)
                
    def get_strategy_scores(self, network_state: Dict) -> Dict[ModificationType, float]:
        """
        Get success probability scores for all modification strategies.
        
        Args:
            network_state: Current network state
            
        Returns:
            Dictionary mapping modification types to success probabilities
        """
        scores = {}
        
        for mod_type in ModificationType:
            if mod_type == ModificationType.NONE:
                continue
                
            # Create feature vector for this modification type
            features = self._create_features(network_state, mod_type)
            
            # Predict success probability
            prob = self.predict(features)
            scores[mod_type] = prob
            
        return scores
        
    def _create_features(self, 
                        network_state: Dict, 
                        modification_type: ModificationType) -> np.ndarray:
        """
        Create feature vector for a proposed modification.
        
        Features (14 total):
        - avg_reward (1)
        - avg_error (1)
        - reward_trend (1)
        - neuron_count (1)
        - layer_count (1)
        - steps_since_last_mod (1)
        - modification_type one-hot (7)
        - network_age (1)
        
        Args:
            network_state: Current network state
            modification_type: Type of modification being proposed
            
        Returns:
            Feature vector (14 dimensions)
        """
        features = [
            network_state.get('avg_reward', 0.0),
            network_state.get('avg_error', 1.0),
            network_state.get('reward_trend', 0.0),
            network_state.get('neuron_count', 10),
            network_state.get('layer_count', 2),
            network_state.get('steps_since_last_mod', 100),
        ]
        
        # One-hot encode modification type
        mod_type_onehot = [0.0] * 7
        mod_type_index = list(ModificationType).index(modification_type)
        mod_type_onehot[mod_type_index] = 1.0
        features.extend(mod_type_onehot)
        
        # Network age
        features.append(network_state.get('network_age', 0))
        
        return np.array(features)
        
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function."""
        return np.maximum(0, x)
        
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        
    def get_statistics(self) -> Dict:
        """
        Get training statistics.
        
        Returns:
            Dictionary with training statistics
        """
        if len(self.training_losses) == 0:
            return {
                'total_updates': 0,
                'avg_loss': 0.0,
                'avg_accuracy': 0.5,
                'recent_loss': 0.0,
                'recent_accuracy': 0.5
            }
            
        recent_window = min(100, len(self.training_losses))
        
        return {
            'total_updates': self.total_updates,
            'avg_loss': np.mean(self.training_losses),
            'avg_accuracy': np.mean(self.training_accuracies),
            'recent_loss': np.mean(self.training_losses[-recent_window:]),
            'recent_accuracy': np.mean(self.training_accuracies[-recent_window:])
        }
        
    def save_weights(self, filepath: str):
        """
        Save model weights to file.
        
        Args:
            filepath: Path to save weights
        """
        np.savez(filepath,
                W1=self.W1, b1=self.b1,
                W2=self.W2, b2=self.b2,
                total_updates=self.total_updates)
                
    def load_weights(self, filepath: str):
        """
        Load model weights from file.
        
        Args:
            filepath: Path to load weights from
        """
        data = np.load(filepath)
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']
        self.total_updates = int(data['total_updates'])
