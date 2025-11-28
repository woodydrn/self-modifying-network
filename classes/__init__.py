"""
Self-Modifying Neural Network Package

Core components for a neural network that can modify its own structure
during training based on performance feedback.
"""

from .network import SelfModifyingNetwork
from .layer import AdaptiveLayer
from .neuron import Neuron
from .reward import GradedRewardFunction
from .backward import IntelligentBackward
from .modification_tracker import ModificationTracker
from .meta_learner import MetaLearner

__all__ = [
    'SelfModifyingNetwork',
    'AdaptiveLayer',
    'Neuron',
    'GradedRewardFunction',
    'IntelligentBackward',
    'ModificationTracker',
    'MetaLearner',
]
