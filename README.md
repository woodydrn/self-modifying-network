# Self-Modifying Neural Network with Meta-Learning

A dynamic neural network that reshapes its own structure through intelligent meta-learning, autonomous growth/pruning mechanisms, and continuous learning like a human brain.

## 🚀 GPU Acceleration

**NEW**: This project now supports NVIDIA GPU acceleration! With your RTX 5080, you can train significantly faster.

**Quick GPU Setup:**
```powershell
# 1. Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 2. Install other dependencies
pip install -r requirements.txt

# 3. Verify GPU is working
python check_gpu.py
```

📖 **See [GPU_SETUP.md](GPU_SETUP.md) for detailed GPU installation and troubleshooting**

## Overview

This is a unique neural network architecture featuring:

- **GPU Acceleration** - Leverage NVIDIA GPUs for 10-50x faster training
- **Continuous Learning** - Trains forever like a human brain, automatically saves/loads checkpoints
- **Meta-Learning System** - Learns which structural modifications work best and applies them intelligently
- **Intelligent Self-Modification** - 6 different strategies (add/remove neurons/layers, rewire, adjust thresholds)
- **Rollback Mechanism** - Monitors modifications and reverts changes that hurt performance
- **Multi-Task Learning** - Learns both addition and division simultaneously with tag-based routing
- **Modification Tracking** - Records success/failure of every structural change for meta-learning

## Quick Start

### Continuous Training Mode

Train the network continuously (runs forever until stopped):

```bash
# Continue from saved checkpoint
python continuous_train.py

# Start with fresh brain
python continuous_train.py --new

# Test only (no training)
python continuous_train.py --test
```

**Features:**
- Auto-saves every 100 batches
- Prints progress every 10 batches
- Press Ctrl+C to stop safely (auto-saves on exit)
- Network learns both addition (0-18 range) and division (0.1-9 range)

### Configuration

Edit `continuous_train.py` to adjust:
- `SAMPLES_PER_BATCH = 50` - Samples per training batch
- `SAVE_INTERVAL_BATCHES = 100` - How often to auto-save
- `MAX_DIGIT = 9` - Maximum digit for math problems
- `PRINT_INTERVAL = 10` - How often to print stats

## Key Features

### 1. Meta-Learning System
- **Modification Tracker** - Records all structural changes with 100-step reward trajectories
- **Meta-Learner Network** - 2-layer MLP predicts success probability of each strategy
- **Intelligent Selection** - ε-greedy (90% exploit best, 10% explore random)
- **Success Classification** - Modifications marked successful if reward improves after 50 steps
- **Training Data Extraction** - Converts modification history to features for meta-learner

**Modification Strategies:**
1. `ADD_NEURON` - Adds neuron to struggling layer (inherits weights from best parent)
2. `REMOVE_NEURON` - Removes underperforming or inactive neurons
3. `ADD_LAYER` - Inserts new layer when network struggling
4. `REMOVE_LAYER` - Removes poorest performing layer
5. `REWIRE_CONNECTIONS` - Changes cross-layer routing based on proximity
6. `ADJUST_THRESHOLDS` - Increases/decreases activation thresholds

### 2. Rollback Mechanism
- **Snapshot System** - Saves network state before each modification
- **50-Step Monitoring** - Tracks reward trajectory after change
- **Automatic Revert** - Rolls back if reward drops significantly (>0.5 on -1/0/+1 scale)
- **Completion Logging** - Records whether modification was kept or rolled back

### 3. Continuous Learning Architecture
- **No Epochs** - Trains on infinite stream of random problems
- **Batch-Based** - Processes 50 samples per batch
- **Mixed Task Sampling** - 50/50 split between addition and division
- **Checkpoint System** - Saves full network state (weights, structure, meta-learner, tracker)
- **Resume Training** - Seamlessly continues from last checkpoint

### 4. Reward System
- **Simplified Scoring** - Returns +1 (good), 0 (neutral), or -1 (bad)
- **Error-Based Classification** - Compares prediction error to target magnitude
- **Threshold Levels:**
  - Excellent: <5% error → +1
  - Good: <15% error → +1
  - Okay: <30% error → 0
  - Poor: <50% error → 0
  - Bad: >50% error → -1
  - Terrible: >100% error → -1

### 5. Network Architecture Details
- **Initial Structure** - 2 layers, 8 neurons per layer
- **Activation Functions:**
  - Hidden layers: `tanh(x)` - bounded, smooth gradients
  - Output layer: linear - full range for regression
- **Weight Initialization** - Xavier/Glorot with scale √(2/(input_dim + output_dim))
- **Output Bias Init** - 9.0 (middle of 0-18 target range)
- **Learning Rate** - 0.1 (aggressive for fast learning)
- **Gradient Clipping** - ±5.0 to prevent explosion
- **Weight Clipping** - ±20.0 to allow sufficient range
- **Activation Threshold** - 0.01 (nearly all neurons fire - tag routing disabled for better learning)

### 6. Modification Timing
- **Check Interval** - Every 5000 steps (allows ~100 batches of learning before structure changes)
- **Rollback Disabled** - Currently disabled to allow exploration during early training
- **Max Layers** - 5 layers maximum
- **Max Neurons Per Layer** - 15 neurons maximum
- **Growth Threshold** - Adds structure if reward < -0.5
- **Pruning Threshold** - Removes structure if reward > 0.5

## Architecture Diagram

```
Input [num1, num2, op1, op2] (4D)
    ↓
[Tag Conversion] (4D tag vector)
    ↓
Layer 1: 8 neurons (tanh activation)
    ↓
Layer 2: 8 neurons (tanh activation)  ← Can grow to 256 neurons
    ↓                                    Can add up to 15 layers total
Output: 1 neuron (linear activation)
    ↓
Prediction (0-18 for addition, 0.1-9 for division)
    ↓
[Reward Computation] → +1, 0, or -1
    ↓
[Gradient Computation] (MSE loss)
    ↓
[Backward Pass] (all neurons participate)
    ↓
[Weight Updates] (clipped to ±20)
    ↓
[Modification Check] (every 5000 steps)
    ↓
[Meta-Learner Prediction] (which strategy to apply)
    ↓
[Execute Strategy] (add/remove/rewire)
    ↓
[Save Checkpoint] (every 100 batches)
```

## Components

### Core Files

**`neuron.py`** - Individual Neuron
- Functional tag (4D identity vector)
- Weight matrix and bias vector
- Forward pass with activation (tanh/linear)
- Backward pass with gradient computation
- Performance history tracking
- Parent weight inheritance for new neurons

**`layer.py`** - Adaptive Layer
- Collection of neurons
- Forward pass with neuron aggregation
- Backward pass distributing gradients
- Dynamic neuron addition/removal
- Rewiring connections between layers
- Layer-level performance stats

**`network.py`** - Self-Modifying Network
- Multi-layer architecture management
- Input-to-tag conversion
- Intelligent modification system
- Snapshot/restore for rollback
- Training step with forward/backward
- Meta-learner training every 500 steps

**`reward.py`** - Graded Reward Function
- Error computation (relative to target)
- Reward classification (+1/0/-1)
- Recent reward/error tracking
- Statistics collection

**`modification_tracker.py`** - Modification Tracking
- Records all structural changes
- Tracks 100-step reward trajectories
- Success/failure classification
- Training data extraction for meta-learner
- Statistics by modification type

**`meta_learner.py`** - Meta-Learning Network
- 2-layer MLP (14 features → 32 hidden → 1 output)
- Predicts modification success probability
- Trains on historical modification data
- Guides strategy selection

**`continuous_train.py`** - Continuous Training Script
- Infinite training loop
- Random problem generation
- Auto-save/load checkpoints
- Progress monitoring
- Test evaluation

## Usage Examples

### Train Continuously

```python
from continuous_train import ContinuousTrainer
from network import SelfModifyingNetwork

# Create or load network
network = SelfModifyingNetwork(
    input_dim=4,
    output_dim=1,
    initial_layers=2,
    initial_neurons_per_layer=8,
    learning_rate=0.1
)

# Start training
trainer = ContinuousTrainer(network)
trainer.run_continuous()  # Runs until Ctrl+C
```

### Test Network

```python
# Load saved network
network = load_network()

# Test on specific problems
x_addition = np.array([5, 3, 0, 0])  # 5 + 3
pred = network.predict(x_addition)
print(f"5 + 3 = {pred[0]:.2f}")  # Should be ~8

x_division = np.array([0, 0, 9, 3])  # 9 / 3
pred = network.predict(x_division)
print(f"9 / 3 = {pred[0]:.2f}")  # Should be ~3
```

### Manual Training Step

```python
import numpy as np

# Generate problem
num1, num2 = 7, 4
x = np.array([num1, num2, 0, 0])  # Addition
target = np.array([num1 + num2])  # = 11

# Train
reward = network.train_step(x, target)
print(f"Reward: {reward}")  # +1, 0, or -1
```


## Command Line Usage

### Start Fresh Training
```bash
python continuous_train.py --new
```
Creates new network with default configuration and begins training from scratch.

### Resume Training
```bash
python continuous_train.py
```
Loads existing checkpoint and continues training.

### Test Network
```bash
python continuous_train.py --test
```
Loads network and runs test evaluation on 100 random problems.

## Network Statistics


View comprehensive statistics:

```python
network.print_network_summary()
```

Output includes:
- Training steps completed
- Layer and neuron counts
- Recent reward performance
- Recent error metrics
- Modification statistics

## Implementation Notes

### Tag System
The network uses 4D tag vectors for neuron identity and routing. Tags are primarily used for neuron initialization and identification. **Note:** Tag-based selective activation is currently disabled for better gradient flow - all neurons participate in forward/backward passes.

### Activation Flow
1. Input `x` converted to tag via random projection (for neuron initialization)
2. All neurons in each layer activate and compute output
3. Layer aggregates outputs from all neurons
4. Output passed to next layer or final prediction

### Learning Process
1. **Forward Pass**: All neurons activate with tanh (hidden) or linear (output)
2. **Loss Computation**: MSE between prediction and target
3. **Reward Calculation**: +1 (good), 0 (neutral), or -1 (bad) based on error
4. **Backward Pass**: Gradients computed and clipped (±5.0)
5. **Weight Update**: All active neurons update with learning_rate=0.1
6. **Weight Clipping**: Weights clipped to ±20.0 range
7. **Structure Check**: Every 5000 steps, meta-learner suggests modifications

### Meta-Learning System
- 2-layer MLP predicts modification success
- Trained on historical modification outcomes
- Features include: reward trends, network size, modification type
- Selects from 6 strategies: add neuron, remove neuron, add layer, remove layer, rewire, adjust learning
- Trains every 500 steps on accumulated modification history

### Modification Strategies
1. **Add Neuron** - Adds neuron to layer with lowest average reward
2. **Remove Neuron** - Removes neuron with lowest performance from largest layer
3. **Add Layer** - Inserts new layer between existing layers
4. **Remove Layer** - Removes layer with lowest average performance
5. **Rewire** - Reconnects neurons between adjacent layers
6. **Adjust Learning** - Modifies per-neuron learning rates

### Checkpoint System
- Auto-saves every 100 batches to `network_checkpoint.pkl`
- Includes full network state (weights, structure, history)
- Can resume training from any checkpoint
- Test mode loads checkpoint for evaluation

## Example Training Output

```
Batch 10, Step 500: Reward=-1.00, Error=56.23, pred=9.87, target=12.45, input=[7, 5, 0, 0]
Batch 20, Step 1000: Reward=-1.00, Error=45.12, pred=11.23, target=15.67, input=[8, 7, 0, 0]
Batch 30, Step 1500: Reward=0.00, Error=23.45, pred=8.76, target=11.23, input=[5, 6, 0, 0]
...
Saved network to network_checkpoint.pkl

Meta-learner training...
Strategy Selected: add_neuron
Added neuron to layer 0. New size: 9 neurons
...
Batch 100, Step 5000: Reward=1.00, Error=3.21, pred=8.12, target=8.00, input=[5, 3, 0, 0]
Saved network to network_checkpoint.pkl
```

## Troubleshooting

### Network predictions stuck at constant value
- Check learning rate (should be ~0.1 for this task)
- Verify gradient clipping not too tight (±5.0 recommended)
- Ensure weight clipping allows sufficient range (±20.0)
- Confirm activation functions: tanh for hidden, linear for output
- Check that tag-based selective activation is disabled (all neurons should fire)

### Network exploding to large values
- Reduce learning rate (try 0.01 or 0.001)
- Tighten gradient clipping (try ±2.0 or ±1.0)
- Add output clipping in predict() method
- Use bounded activation (tanh instead of ReLU)

### Structure not evolving
- Reduce check_structure_every (try 1000-3000 steps)
- Verify rollback is not too aggressive
- Check reward thresholds for growth/pruning
- Ensure meta-learner has training data

### Memory/performance issues
- Reduce max_layers or max_neurons_per_layer
- Increase check_structure_every for less frequent modifications
- Clear modification history periodically

## Theory and Inspiration

This architecture combines:
- **Self-Modifying Code**: Network rewrites its own structure
- **Meta-Learning**: Network learns which modifications work
- **Continual Learning**: Trains indefinitely without catastrophic forgetting
- **Neural Architecture Search**: Automated architecture optimization

Key innovations:
- Meta-learner guides structural changes based on past outcomes
- Modification tracking provides supervised signal for meta-learning
- Rollback mechanism prevents destructive changes (currently disabled)
- Continuous training mode enables long-term evolution

## Current Status

**Working Configuration (as of latest update):**
- 2 layers, 8 neurons per layer initially
- Learning rate: 0.1
- Activation: tanh (hidden), linear (output)
- Gradient clip: ±5.0
- Weight clip: ±20.0
- Tag-based selective activation: **DISABLED** (all neurons always fire)
- Rollback: **DISABLED** (allows exploration)
- Modification interval: 5000 steps
- Tasks: Addition (0-18) and Division (0.1-9), 50/50 split

**Recent Fixes:**
- Disabled tag-based sparse activation to enable gradient flow
- Adjusted learning rate from 0.5 → 0.1
- Loosened weight clipping from ±10 → ±20
- Increased modification interval from 300 → 5000 steps
- Disabled aggressive rollback (was 82% rollback rate)
- Fixed output bias initialization (0 → 9.0)

## Future Directions

- Re-enable selective tag-based activation once basic learning works
- Implement adaptive learning rate schedules
- Add more sophisticated modification strategies
- Experiment with different activation functions
- Implement curriculum learning (start easy, increase difficulty)
- Add visualization of network structure evolution
- Support multi-task learning beyond addition/division

## License

MIT License - See LICENSE file for details

## Future Enhancements

- Attention-based tag matching
- Hierarchical tag structures
- Transfer learning between tasks via tag alignment
- Multi-objective reward functions
- Genetic algorithm for tag evolution
- Synaptic plasticity in connection weights

## License

MIT License

## Contributing

Contributions welcome! Areas of interest:
- Alternative tag matching mechanisms
- More sophisticated role taxonomies
- Visualization tools for tag space evolution
- Benchmark comparisons with traditional architectures
- Applications to reinforcement learning

## Citation

If you use this code in your research, please cite:

```
@software{self_modifying_network,
  title = {Self-Modifying Neural Network with Tag-Based Activation},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/self-modifying-network}
}
```
