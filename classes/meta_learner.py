import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from .modification_tracker import ModificationType

# Check for GPU availability
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MetaLearner(nn.Module):
    """
    A PyTorch-based neural network that learns to predict modification success.
    GPU-accelerated with autograd for efficient training.
    
    Uses modification history to learn which structural changes work
    in which network states. This enables intelligent strategy selection.
    
    Architecture: 2-layer MLP with 32 hidden units
    Input: Network state + proposed modification type (15 features)
    Output: Success probability (0-1)
    """
    
    def __init__(self, 
                 input_dim: int = 15,
                 hidden_dim: int = 32,
                 learning_rate: float = 0.01,
                 device: torch.device = None):
        """
        Initialize the meta-learner.
        
        Args:
            input_dim: Number of input features (default 15)
            hidden_dim: Number of hidden units (default 32)
            learning_rate: Learning rate for training
            device: PyTorch device (GPU/CPU)
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.device = device if device is not None else DEVICE
        
        # PyTorch layers with proper initialization
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
        # Xavier initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        
        # Move to device
        self.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        
        # Training statistics
        self.training_losses: List[float] = []
        self.training_accuracies: List[float] = []
        self.total_updates = 0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        h = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(h))
        
    def predict(self, features: np.ndarray) -> float:
        """
        Predict success probability for a modification.
        
        Args:
            features: Feature vector (15 dimensions)
            
        Returns:
            Success probability (0-1)
        """
        self.eval()
        with torch.no_grad():
            x = torch.as_tensor(features, dtype=torch.float32, device=self.device)
            if x.dim() == 2:
                x = x.flatten()
            prob = self.forward(x.unsqueeze(0))
            return prob.item()
        
    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Predict success probabilities for multiple modifications.
        
        Args:
            X: Feature matrix (n_samples, 15)
            
        Returns:
            Success probabilities (n_samples,)
        """
        if X.shape[0] == 0:
            return np.array([])
        
        self.eval()
        with torch.no_grad():
            x = torch.as_tensor(X, dtype=torch.float32, device=self.device)
            probs = self.forward(x)
            return probs.squeeze(-1).cpu().numpy()
        
    def train_step(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        Perform one training step using autograd.
        
        Args:
            X: Feature matrix (n_samples, 15)
            y: Target labels (n_samples,) - 1.0 for success, 0.0 for failure
            
        Returns:
            (loss, accuracy) tuple
        """
        if X.shape[0] == 0:
            return 0.0, 0.5
        
        self.train()
        x = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        targets = torch.as_tensor(y, dtype=torch.float32, device=self.device).unsqueeze(-1)
        
        # Forward pass with autograd
        self.optimizer.zero_grad()
        probs = self.forward(x)
        loss = F.binary_cross_entropy(probs, targets)
        
        # Backward pass (autograd handles gradients)
        loss.backward()
        self.optimizer.step()
        
        # Compute accuracy
        with torch.no_grad():
            predictions = (probs >= 0.5).float()
            accuracy = (predictions == targets).float().mean().item()
        
        loss_val = loss.item()
        
        # Track statistics
        self.training_losses.append(loss_val)
        self.training_accuracies.append(accuracy)
        self.total_updates += 1
        
        return loss_val, accuracy
        
    def train_model(self, X: np.ndarray, y: np.ndarray, epochs: int = 10, batch_size: int = 32) -> Dict:
        """
        Train the meta-learner on modification history.
        
        Args:
            X: Feature matrix (n_samples, 15)
            y: Target labels (n_samples,)
            epochs: Number of training epochs
            batch_size: Batch size for mini-batch training
            
        Returns:
            Dictionary with training statistics
        """
        if X.shape[0] == 0:
            return {'samples': 0, 'loss': 0.0, 'accuracy': 0.0}
            
        n = X.shape[0]
        all_losses = []
        all_accs = []
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Mini-batch training
            for i in range(0, n, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                loss, acc = self.train_step(X_batch, y_batch)
                all_losses.append(loss)
                all_accs.append(acc)
        
        return {
            'samples': n,
            'loss': np.mean(all_losses),
            'accuracy': np.mean(all_accs),
            'epochs': epochs
        }
                
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
        
        Features (15 total):
        - avg_reward (1)
        - avg_error (1)
        - reward_trend (1)
        - neuron_count (1)
        - layer_count (1)
        - steps_since_last_mod (1)
        - modification_type one-hot (7)
        - network_age (1)
        - plateau_detected (1) - indicates if stuck in local minimum
        
        Args:
            network_state: Current network state
            modification_type: Type of modification being proposed
            
        Returns:
            Feature vector (15 dimensions)
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
        
        # Plateau detection - crucial signal for meta-learner
        # When stuck, meta-learner should prefer more aggressive strategies
        features.append(network_state.get('plateau_detected', 0.0))
        
        return np.array(features, dtype=np.float32)
        
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
        torch.save({
            'state_dict': self.state_dict(),
            'total_updates': self.total_updates,
            'training_losses': self.training_losses,
            'training_accuracies': self.training_accuracies
        }, filepath)
                
    def load_weights(self, filepath: str):
        """
        Load model weights from file.
        
        Args:
            filepath: Path to load weights from
        """
        data = torch.load(filepath, map_location=self.device)
        self.load_state_dict(data['state_dict'])
        self.total_updates = data.get('total_updates', 0)
        self.training_losses = data.get('training_losses', [])
        self.training_accuracies = data.get('training_accuracies', [])
