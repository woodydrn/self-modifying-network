# TensorBoard Integration

## What Was Added

TensorBoard logging has been integrated into the continuous training system to provide real-time visualization of network training and evolution.

## Logged Metrics

### Training Metrics (`Training/`)
- **Reward** - Average reward over last 50 samples
- **Error** - Average prediction error
- **Plateau_Detected** - Binary flag (1 = stuck in local minimum)
- **Reward_Variance** - Variance over 50-batch window (for plateau detection)

### Network Structure (`Network/`)
- **Total_Neurons** - Total neuron count across all layers
- **Total_Layers** - Number of layers in the network
- **Layer_X_Neurons** - Neuron count for each individual layer

### Modifications (`Modifications/`)
- **Event** - Marks when a structural modification occurs
- **Type** - Text description of modification type
- **Success** - Whether the modification was successful

### Test Results (`Test/`)
- **Addition_Accuracy** - Accuracy on addition problems (%)
- **Division_Accuracy** - Accuracy on division problems (%)
- **Overall_Accuracy** - Combined accuracy across both tasks (%)

## Files Modified

1. **requirements.txt** - Added `tensorboard>=2.11.0`
2. **continuous_train.py**:
   - Added TensorBoard imports
   - Created `SummaryWriter` in `ContinuousTrainer.__init__()`
   - Added logging in `train_batch()` method
   - Added `log_modification()` method
   - Enhanced `print_status()` with per-layer logging
   - Added test accuracy logging
   - Proper writer cleanup on exit

3. **.gitignore** - Added `runs/` directory (TensorBoard logs)
4. **README.md** - Added TensorBoard documentation
5. **launch_tensorboard.ps1** - Convenience script for Windows

## Usage

### Start Training with TensorBoard
```powershell
# Terminal 1: Start training
python continuous_train.py

# Terminal 2: Start TensorBoard
.\launch_tensorboard.ps1
# Or: tensorboard --logdir=runs

# Browser: Open http://localhost:6006
```

### What You'll See

- **Real-time reward curves** showing learning progress
- **Plateau detection events** when network gets stuck
- **Network growth** as layers/neurons are added
- **Modification patterns** showing which strategies are attempted
- **Test accuracy** after evaluation runs

### Comparing Runs

TensorBoard automatically creates timestamped directories:
```
runs/
├── run_20250128_143022/
├── run_20250128_150415/
└── run_20250128_162344/
```

You can compare multiple runs to see:
- Which hyperparameters work best
- How different starting conditions affect learning
- Whether the meta-learner improves over time

## Benefits

1. **Real-time Monitoring** - See training progress without stopping
2. **Pattern Recognition** - Identify when network gets stuck
3. **Structure Evolution** - Watch network grow/shrink intelligently
4. **Debugging** - Quickly spot issues (flatlines, explosions, etc.)
5. **Comparison** - Compare different configurations side-by-side

## Example Insights

- **Plateau Detection**: When variance drops below threshold, you'll see the plateau flag activate
- **Meta-Learner Impact**: Watch if modifications cluster around plateau events
- **Layer Growth**: See which layers grow fastest (usually middle layers)
- **Learning Curves**: Typical pattern shows initial rapid improvement, then plateaus requiring structural changes
