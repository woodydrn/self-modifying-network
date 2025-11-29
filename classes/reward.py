import numpy as np
from typing import List
from typing import Dict, List, Tuple
from collections import deque


class GradedRewardFunction:
    """
    A reward function that provides graded feedback based on prediction quality.
    Closer predictions get higher rewards, far predictions get negative rewards.
    
    Attributes:
        thresholds: Dict of performance thresholds for each reward level
        rewards: Dict of reward values for each level
        history: Recent reward history
    """
    
    def __init__(self,
                 excellent_threshold: float = 0.05,
                 good_threshold: float = 0.15,
                 poor_threshold: float = 0.30,
                 bad_threshold: float = 0.50):
        """
        Initialize graded reward function.
        
        Args:
            excellent_threshold: Error threshold for excellent performance (default 5%)
            good_threshold: Error threshold for good performance (default 15%)
            poor_threshold: Error threshold for poor performance (default 30%)
            bad_threshold: Error threshold for bad performance (default 50%)
        """
        # Thresholds (as fraction of target value or absolute for small values)
        self.thresholds = {
            'excellent': excellent_threshold,
            'good': good_threshold,
            'poor': poor_threshold,
            'bad': bad_threshold
        }
        
        # Reward values - simplified to +1, 0, -1
        self.rewards = {
            'excellent': 1.0,
            'good': 1.0,
            'okay': 0.0,
            'poor': 0.0,
            'bad': -1.0,
            'terrible': -1.0
        }
        
        # History
        self.reward_history = deque(maxlen=1000)
        self.error_history = deque(maxlen=1000)
        self.level_counts = {level: 0 for level in self.rewards.keys()}
        
    def compute_error(self, prediction: np.ndarray, target: np.ndarray) -> float:
        """
        Compute normalized error between prediction and target.
        
        Args:
            prediction: Predicted value(s)
            target: Target value(s)
            
        Returns:
            Normalized error value
        """
        # Compute absolute error
        abs_error = np.abs(prediction - target)
        
        # Normalize by target magnitude (use 1.0 floor to handle zero targets stably)
        target_magnitude = np.maximum(np.abs(target), 1.0)
        relative_error = abs_error / target_magnitude
        
        # Return mean relative error for vector outputs
        return np.mean(relative_error)
    
    def compute_reward(self, 
                       prediction: np.ndarray, 
                       target: np.ndarray,
                       return_level: bool = False) -> float:
        """
        Compute graded reward based on prediction quality.
        
        Args:
            prediction: Predicted value(s)
            target: Target value(s)
            return_level: If True, also return the performance level name
            
        Returns:
            Reward value (and optionally performance level)
        """
        error = self.compute_error(prediction, target)
        
        # Determine performance level
        if error <= self.thresholds['excellent']:
            level = 'excellent'
        elif error <= self.thresholds['good']:
            level = 'good'
        elif error <= self.thresholds['poor']:
            level = 'okay'
        elif error <= self.thresholds['bad']:
            level = 'poor'
        elif error <= 1.0:
            level = 'bad'
        else:
            level = 'terrible'
        
        reward = self.rewards[level]
        
        # Record history
        self.reward_history.append(reward)
        self.error_history.append(error)
        self.level_counts[level] += 1
        
        if return_level:
            return reward, level
        return reward
    
    def compute_multi_target_reward(self,
                                     predictions: List[np.ndarray],
                                     targets: List[np.ndarray]) -> List[float]:
        """
        Compute rewards for multiple prediction-target pairs.
        
        Args:
            predictions: List of predictions
            targets: List of targets
            
        Returns:
            List of reward values
        """
        rewards = []
        for pred, target in zip(predictions, targets):
            reward = self.compute_reward(pred, target)
            rewards.append(reward)
        return rewards
    
    def get_average_reward(self, last_n: int = 100) -> float:
        """
        Get average reward over recent history.
        
        Args:
            last_n: Number of recent rewards to average
            
        Returns:
            Average reward
        """
        if len(self.reward_history) == 0:
            return 0.0
        
        recent = list(self.reward_history)[-last_n:]
        return np.mean(recent)
    
    def get_recent_rewards(self, last_n: int = 100) -> List[float]:
        """
        Get recent reward values.
        
        Args:
            last_n: Number of recent rewards to return
            
        Returns:
            List of recent rewards
        """
        if len(self.reward_history) == 0:
            return []
        
        return list(self.reward_history)[-last_n:]
    
    def get_average_error(self, last_n: int = 100) -> float:
        """
        Get average error over recent history.
        
        Args:
            last_n: Number of recent errors to average
            
        Returns:
            Average error
        """
        if len(self.error_history) == 0:
            return 0.0
        
        recent = list(self.error_history)[-last_n:]
        return np.mean(recent)
    
    def is_improving(self, window: int = 50) -> bool:
        """
        Check if performance is improving over time.
        
        Args:
            window: Window size for comparison
            
        Returns:
            True if recent performance is better than earlier performance
        """
        if len(self.reward_history) < 2 * window:
            return False
        
        recent = list(self.reward_history)[-window:]
        earlier = list(self.reward_history)[-2*window:-window]
        
        return np.mean(recent) > np.mean(earlier)
    
    def is_struggling(self, threshold: float = -2.0, window: int = 50) -> bool:
        """
        Check if network is struggling (poor recent performance).
        
        Args:
            threshold: Reward threshold for struggling
            window: Window size for evaluation
            
        Returns:
            True if average recent reward is below threshold
        """
        if len(self.reward_history) < window:
            return False
        
        recent_avg = self.get_average_reward(window)
        return recent_avg < threshold
    
    def is_performing_well(self, threshold: float = 5.0, window: int = 50) -> bool:
        """
        Check if network is performing well.
        
        Args:
            threshold: Reward threshold for good performance
            window: Window size for evaluation
            
        Returns:
            True if average recent reward is above threshold
        """
        if len(self.reward_history) < window:
            return False
        
        recent_avg = self.get_average_reward(window)
        return recent_avg > threshold
    
    def get_statistics(self) -> Dict:
        """
        Get comprehensive reward statistics.
        
        Returns:
            Dictionary of statistics
        """
        total_samples = len(self.reward_history)
        
        stats = {
            'total_samples': total_samples,
            'average_reward': self.get_average_reward(total_samples) if total_samples > 0 else 0.0,
            'average_error': self.get_average_error(total_samples) if total_samples > 0 else 0.0,
            'recent_reward': self.get_average_reward(50),
            'recent_error': self.get_average_error(50),
            'level_distribution': {}
        }
        
        # Add level distribution percentages
        if total_samples > 0:
            for level, count in self.level_counts.items():
                stats['level_distribution'][level] = (count / total_samples) * 100
        
        return stats
    
    def adapt_thresholds(self, difficulty_factor: float = 1.0):
        """
        Adapt reward thresholds based on difficulty.
        Higher difficulty = more lenient thresholds.
        
        Args:
            difficulty_factor: Multiplier for thresholds (>1 = easier, <1 = harder)
        """
        for key in self.thresholds:
            self.thresholds[key] *= difficulty_factor
    
    def reset_statistics(self):
        """Reset all statistics and history."""
        self.reward_history.clear()
        self.error_history.clear()
        self.level_counts = {level: 0 for level in self.rewards.keys()}
    
    def __repr__(self):
        stats = self.get_statistics()
        return f"GradedReward(avg={stats['average_reward']:.2f}, recent={stats['recent_reward']:.2f}, samples={stats['total_samples']})"


class AccumulatedReward:
    """
    Accumulates rewards over multiple training steps for more stable learning.
    """
    
    def __init__(self, 
                 graded_reward: GradedRewardFunction,
                 accumulation_window: int = 10):
        """
        Initialize accumulated reward.
        
        Args:
            graded_reward: The base graded reward function
            accumulation_window: Number of steps to accumulate over
        """
        self.graded_reward = graded_reward
        self.accumulation_window = accumulation_window
        self.accumulated_rewards = deque(maxlen=accumulation_window)
        
    def add_reward(self, prediction: np.ndarray, target: np.ndarray):
        """
        Add a reward to the accumulation buffer.
        
        Args:
            prediction: Predicted value
            target: Target value
        """
        reward = self.graded_reward.compute_reward(prediction, target)
        self.accumulated_rewards.append(reward)
    
    def get_accumulated_reward(self) -> float:
        """
        Get the accumulated (averaged) reward.
        
        Returns:
            Average accumulated reward
        """
        if len(self.accumulated_rewards) == 0:
            return 0.0
        return np.mean(self.accumulated_rewards)
    
    def should_update(self) -> bool:
        """
        Check if enough rewards have accumulated for an update.
        
        Returns:
            True if buffer is full
        """
        return len(self.accumulated_rewards) >= self.accumulation_window
    
    def reset(self):
        """Reset accumulation buffer."""
        self.accumulated_rewards.clear()
