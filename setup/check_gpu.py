"""
GPU Availability Checker
Run this script to verify your NVIDIA GPU is properly configured.
"""

import sys

def check_gpu():
    print("="*60)
    print("GPU AVAILABILITY CHECK")
    print("="*60)
    
    # Check PyTorch installation
    try:
        import torch
        print(f"✓ PyTorch installed: {torch.__version__}")
    except ImportError:
        print("✗ PyTorch not installed!")
        print("\nTo install PyTorch with CUDA support, run:")
        print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        return False
    
    # Check CUDA availability
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("\n✗ CUDA is not available!")
        print("\nPossible reasons:")
        print("1. PyTorch was installed without CUDA support")
        print("2. NVIDIA GPU drivers are not installed")
        print("3. CUDA toolkit is not installed")
        print("\nTo fix:")
        print("1. Install NVIDIA GPU drivers from: https://www.nvidia.com/download/index.aspx")
        print("2. Install PyTorch with CUDA:")
        print("   pip uninstall torch torchvision torchaudio")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        return False
    
    # Display GPU information
    print(f"\n✓ CUDA is available!")
    print(f"\nCUDA Version: {torch.version.cuda}")
    print(f"cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Total Memory: {props.total_memory / 1e9:.2f} GB")
        print(f"  Multi-Processors: {props.multi_processor_count}")
    
    # Test GPU computation
    print("\n" + "="*60)
    print("TESTING GPU COMPUTATION")
    print("="*60)
    
    try:
        import time
        
        device = torch.device('cuda')
        
        # Small test
        print("\nSmall matrix test (100x100)...")
        x = torch.randn(100, 100, device=device)
        y = torch.randn(100, 100, device=device)
        z = torch.matmul(x, y)
        print("✓ Small test passed")
        
        # Large test
        print("\nLarge matrix test (5000x5000)...")
        x = torch.randn(5000, 5000, device=device)
        y = torch.randn(5000, 5000, device=device)
        
        start = time.time()
        z = torch.matmul(x, y)
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        
        print(f"✓ GPU computation time: {gpu_time*1000:.2f} ms")
        
        # Compare with CPU
        print("\nComparing with CPU...")
        x_cpu = x.cpu()
        y_cpu = y.cpu()
        
        start = time.time()
        z_cpu = torch.matmul(x_cpu, y_cpu)
        cpu_time = time.time() - start
        
        print(f"  CPU computation time: {cpu_time*1000:.2f} ms")
        print(f"  Speedup: {cpu_time/gpu_time:.1f}x faster on GPU")
        
        # Memory info
        print("\n" + "="*60)
        print("GPU MEMORY")
        print("="*60)
        print(f"Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        print(f"Max Allocated: {torch.cuda.max_memory_allocated(0) / 1e9:.2f} GB")
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED - GPU IS READY!")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"\n✗ GPU test failed: {e}")
        return False


if __name__ == "__main__":
    success = check_gpu()
    sys.exit(0 if success else 1)
