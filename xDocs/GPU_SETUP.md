# GPU Setup Guide for Self-Modifying Network

## Requirements
- NVIDIA GPU (RTX 5080 or compatible)
- Windows 10/11
- Python 3.8+

## Installation Steps

### 1. Install NVIDIA Drivers
Download and install the latest drivers from:
https://www.nvidia.com/download/index.aspx

### 2. Install PyTorch with CUDA Support

First, uninstall any existing PyTorch:
```powershell
pip uninstall torch torchvision torchaudio
```

Then install PyTorch with CUDA 12.1:
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Or for CUDA 11.8:
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install Project Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Verify GPU Setup
Run the GPU checker script:
```powershell
python check_gpu.py
```

You should see output like:
```
===========================================================
GPU AVAILABILITY CHECK
===========================================================
✓ PyTorch installed: 2.x.x

CUDA available: True

✓ CUDA is available!

CUDA Version: 12.1
cuDNN Version: xxxxx
Number of GPUs: 1

GPU 0: NVIDIA GeForce RTX 5080
  Compute Capability: 8.9
  Total Memory: 16.00 GB
  Multi-Processors: xx

===========================================================
TESTING GPU COMPUTATION
===========================================================
✓ ALL TESTS PASSED - GPU IS READY!
```

## Using GPU in Training

The network will automatically use GPU if available. You can force CPU mode by modifying the code:

```python
# In train.py, continuous_train.py, or train-2.py
device_config = get_device_config(force_cpu=True)  # Force CPU
# or
device_config = get_device_config(force_cpu=False)  # Use GPU if available
```

## Performance Tips

1. **Larger Batch Sizes**: With GPU, you can increase batch sizes for faster training
2. **Monitor GPU Memory**: Use `nvidia-smi` to monitor GPU usage
3. **Clear Cache**: If you encounter memory issues, the code will automatically clear cache

## Troubleshooting

### Issue: CUDA not available
**Solution**: 
- Ensure NVIDIA drivers are installed
- Reinstall PyTorch with CUDA support
- Check that your GPU is CUDA-capable

### Issue: Out of memory
**Solution**:
- Reduce batch size
- Reduce number of neurons per layer
- Clear GPU cache between runs

### Issue: Slow performance
**Solution**:
- Verify GPU is being used (check with `nvidia-smi`)
- Update NVIDIA drivers
- Close other GPU-intensive applications

## Monitoring GPU Usage

### During Training
The network will print GPU information at startup:
```
===========================================================
DEVICE CONFIGURATION
===========================================================
Using GPU: NVIDIA GeForce RTX 5080
CUDA Version: 12.1
Total GPU Memory: 16.00 GB
...
```

### Real-time Monitoring
In a separate PowerShell window:
```powershell
nvidia-smi -l 1  # Updates every 1 second
```

Or for continuous monitoring:
```powershell
while ($true) { cls; nvidia-smi; Start-Sleep -Seconds 1 }
```

## Expected Performance Boost

With RTX 5080, you should see:
- 10-50x faster matrix operations compared to CPU
- Ability to train larger networks
- Better performance with more neurons/layers

The actual speedup depends on:
- Network size (larger networks benefit more)
- Batch size
- Type of operations being performed
