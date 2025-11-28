import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple
from enum import Enum


class ModificationType(Enum):
    """Types of structural modifications."""
    ADD_NEURON = "add_neuron"
    REMOVE_NEURON = "remove_neuron"
    ADD_LAYER = "add_layer"
    REMOVE_LAYER = "remove_layer"
    REWIRE_CONNECTIONS = "rewire_connections"
    ADJUST_THRESHOLDS = "adjust_thresholds"
    NONE = "none"  # No modification


class ModificationRecord:
    """
    Record of a single structural modification attempt.
    
    Tracks the context, action, and outcome of a modification to enable
    learning which modifications work in which situations.
    """
    
    def __init__(self,
                 modification_type: ModificationType,
                 timestamp: int,
                 network_state_before: Dict,
                 context: Optional[Dict] = None):
        """
        Initialize a modification record.
        
        Args:
            modification_type: Type of modification attempted
            timestamp: Training step when modification occurred
            network_state_before: Network state snapshot before modification
            context: Additional context about why this modification was chosen
        """
        self.modification_type = modification_type
        self.timestamp = timestamp
        self.network_state_before = network_state_before
        self.network_state_after: Optional[Dict] = None
        self.context = context or {}
        
        # Outcome tracking
        self.reward_trajectory: List[float] = []  # Rewards for next 100 steps
        self.error_trajectory: List[float] = []   # Errors for next 100 steps
        self.success: Optional[bool] = None       # Was this modification beneficial?
        self.reward_improvement: float = 0.0      # Change in average reward
        self.completed: bool = False              # Has trajectory collection finished?
        self.rolled_back: bool = False            # Was this modification rolled back?
        
    def set_network_state_after(self, state: Dict):
        """Set the network state after modification."""
        self.network_state_after = state
        
    def add_trajectory_point(self, reward: float, error: float):
        """Add a reward/error observation to the trajectory."""
        if len(self.reward_trajectory) < 100:
            self.reward_trajectory.append(reward)
            self.error_trajectory.append(error)
            
    def finalize(self):
        """
        Finalize the record and determine success.
        
        Success criteria:
        - Average reward improved by at least 1.0
        - OR error decreased by at least 10%
        - AND was not rolled back
        """
        if self.rolled_back:
            self.success = False
            self.completed = True
            return
            
        if len(self.reward_trajectory) == 0:
            self.success = False
            self.completed = True
            return
            
        # Calculate improvement
        baseline_reward = self.network_state_before.get('avg_reward', 0.0)
        after_reward = np.mean(self.reward_trajectory)
        self.reward_improvement = after_reward - baseline_reward
        
        baseline_error = self.network_state_before.get('avg_error', 1.0)
        after_error = np.mean(self.error_trajectory)
        error_improvement = (baseline_error - after_error) / max(baseline_error, 0.01)
        
        # Success if reward improved significantly OR error decreased significantly
        self.success = (self.reward_improvement > 1.0) or (error_improvement > 0.1)
        self.completed = True
        
    def mark_rolled_back(self):
        """Mark this modification as rolled back (failed)."""
        self.rolled_back = True
        self.success = False
        self.completed = True
        
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'type': self.modification_type.value,
            'timestamp': self.timestamp,
            'state_before': self.network_state_before,
            'state_after': self.network_state_after,
            'context': self.context,
            'reward_trajectory': self.reward_trajectory,
            'error_trajectory': self.error_trajectory,
            'success': self.success,
            'reward_improvement': self.reward_improvement,
            'rolled_back': self.rolled_back
        }


class ModificationTracker:
    """
    Tracks all structural modifications and their outcomes.
    
    Maintains a history of modifications to enable meta-learning:
    which modifications work in which situations.
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize the modification tracker.
        
        Args:
            max_history: Maximum number of modification records to keep
        """
        self.max_history = max_history
        self.history: deque = deque(maxlen=max_history)
        self.active_record: Optional[ModificationRecord] = None
        
        # Statistics
        self.total_modifications = 0
        self.total_rollbacks = 0
        self.success_count_by_type: Dict[ModificationType, int] = {
            mod_type: 0 for mod_type in ModificationType
        }
        self.attempt_count_by_type: Dict[ModificationType, int] = {
            mod_type: 0 for mod_type in ModificationType
        }
        
    def start_modification(self,
                          modification_type: ModificationType,
                          timestamp: int,
                          network_state: Dict,
                          context: Optional[Dict] = None) -> ModificationRecord:
        """
        Start tracking a new modification.
        
        Args:
            modification_type: Type of modification being attempted
            timestamp: Current training step
            network_state: Current network state snapshot
            context: Additional context about this modification
            
        Returns:
            The new modification record
        """
        # Finalize any active record
        if self.active_record is not None and not self.active_record.completed:
            self.active_record.finalize()
            self.history.append(self.active_record)
            
        # Create new record
        record = ModificationRecord(
            modification_type=modification_type,
            timestamp=timestamp,
            network_state_before=network_state,
            context=context
        )
        
        self.active_record = record
        self.total_modifications += 1
        self.attempt_count_by_type[modification_type] += 1
        
        return record
        
    def complete_modification(self, network_state_after: Dict):
        """
        Mark that the modification has been applied.
        
        Args:
            network_state_after: Network state after modification
        """
        if self.active_record is not None:
            self.active_record.set_network_state_after(network_state_after)
            
    def update_trajectory(self, reward: float, error: float):
        """
        Add a trajectory point to the active modification record.
        
        Args:
            reward: Current reward value
            error: Current error value
        """
        if self.active_record is not None and not self.active_record.completed:
            self.active_record.add_trajectory_point(reward, error)
            
            # Auto-finalize if trajectory is complete
            if len(self.active_record.reward_trajectory) >= 100:
                self.finalize_active_record()
                
    def finalize_active_record(self):
        """Finalize the active modification record and add to history."""
        if self.active_record is not None:
            self.active_record.finalize()
            
            # Update statistics
            if self.active_record.success:
                self.success_count_by_type[self.active_record.modification_type] += 1
                
            self.history.append(self.active_record)
            self.active_record = None
            
    def record_rollback(self):
        """Record that the active modification was rolled back."""
        if self.active_record is not None:
            self.active_record.mark_rolled_back()
            self.total_rollbacks += 1
            self.finalize_active_record()
            
    def get_success_rate(self, modification_type: ModificationType) -> float:
        """
        Get success rate for a specific modification type.
        
        Args:
            modification_type: Type of modification to query
            
        Returns:
            Success rate (0.0 to 1.0), or 0.5 if no attempts yet
        """
        attempts = self.attempt_count_by_type.get(modification_type, 0)
        if attempts == 0:
            return 0.5  # Neutral prior
            
        successes = self.success_count_by_type.get(modification_type, 0)
        return successes / attempts
        
    def get_recent_modifications(self, n: int = 10) -> List[ModificationRecord]:
        """
        Get the n most recent modification records.
        
        Args:
            n: Number of records to retrieve
            
        Returns:
            List of recent modification records
        """
        return list(self.history)[-n:]
        
    def get_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get training data for meta-learner.
        
        Returns:
            X: Feature array (n_samples, n_features)
            y: Target array (n_samples,) - 1.0 for success, 0.0 for failure
        """
        completed_records = [r for r in self.history if r.completed]
        
        if len(completed_records) == 0:
            return np.array([]), np.array([])
            
        X = []
        y = []
        
        for record in completed_records:
            features = self._extract_features(record)
            X.append(features)
            y.append(1.0 if record.success else 0.0)
            
        return np.array(X), np.array(y)
        
    def _extract_features(self, record: ModificationRecord) -> np.ndarray:
        """
        Extract feature vector from a modification record.
        
        Features (15 total):
        - avg_reward (1)
        - avg_error (1)
        - reward_trend (1)
        - neuron_count (1)
        - layer_count (1)
        - steps_since_last_mod (1)
        - modification_type one-hot (7)
        - network_age (1)
        - plateau_detected (1)
        
        Args:
            record: Modification record
            
        Returns:
            Feature vector (15 dimensions)
        """
        state = record.network_state_before
        
        features = [
            state.get('avg_reward', 0.0),
            state.get('avg_error', 1.0),
            state.get('reward_trend', 0.0),
            state.get('neuron_count', 10),
            state.get('layer_count', 2),
            state.get('steps_since_last_mod', 100),
        ]
        
        # One-hot encode modification type
        mod_type_onehot = [0.0] * 7
        mod_type_index = list(ModificationType).index(record.modification_type)
        mod_type_onehot[mod_type_index] = 1.0
        features.extend(mod_type_onehot)
        
        # Network age
        features.append(state.get('network_age', 0))
        
        # Plateau detection signal
        features.append(state.get('plateau_detected', 0.0))
        
        return np.array(features)
        
    def get_statistics(self) -> Dict:
        """
        Get comprehensive statistics about modifications.
        
        Returns:
            Dictionary with statistics
        """
        completed = [r for r in self.history if r.completed]
        
        stats = {
            'total_modifications': self.total_modifications,
            'total_rollbacks': self.total_rollbacks,
            'completed_records': len(completed),
            'success_rate_overall': sum(1 for r in completed if r.success) / max(len(completed), 1),
            'success_rate_by_type': {},
            'avg_reward_improvement': {},
        }
        
        # Per-type statistics
        for mod_type in ModificationType:
            type_records = [r for r in completed if r.modification_type == mod_type]
            if len(type_records) > 0:
                success_rate = sum(1 for r in type_records if r.success) / len(type_records)
                avg_improvement = np.mean([r.reward_improvement for r in type_records])
                stats['success_rate_by_type'][mod_type.value] = success_rate
                stats['avg_reward_improvement'][mod_type.value] = avg_improvement
                
        return stats
        
    def print_statistics(self):
        """Print formatted statistics."""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("MODIFICATION TRACKER STATISTICS")
        print("="*60)
        print(f"Total Modifications: {stats['total_modifications']}")
        print(f"Total Rollbacks: {stats['total_rollbacks']}")
        print(f"Completed Records: {stats['completed_records']}")
        print(f"Overall Success Rate: {stats['success_rate_overall']*100:.1f}%")
        print()
        
        print("Success Rate by Type:")
        for mod_type, rate in stats['success_rate_by_type'].items():
            improvement = stats['avg_reward_improvement'].get(mod_type, 0.0)
            print(f"  {mod_type:20s}: {rate*100:5.1f}% (avg improvement: {improvement:+.2f})")
        print("="*60)
