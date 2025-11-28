import torch
import numpy as np


class DeviceConfig:
    """Configuration for GPU/CPU device selection."""
    
    def __init__(self, force_cpu: bool = False):
        """
        Initialize device configuration.
        
        Args:
            force_cpu: If True, force CPU usage even if GPU is available
        """
        self.force_cpu = force_cpu
        self.device = self._get_device()
        self._print_device_info()
    
    def _get_device(self) -> torch.device:
        """Get the appropriate device (CUDA GPU or CPU)."""
        if self.force_cpu:
            return torch.device('cpu')
        
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            print("WARNING: CUDA is not available. Using CPU.")
            return torch.device('cpu')
    
    def _print_device_info(self):
        """Print information about the selected device."""
        print("="*60)
        print("DEVICE CONFIGURATION")
        print("="*60)
        
        if self.device.type == 'cuda':
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"PyTorch Version: {torch.__version__}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            
            # Memory info
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            allocated_memory = torch.cuda.memory_allocated(0) / 1e9
            cached_memory = torch.cuda.memory_reserved(0) / 1e9
            
            print(f"Total GPU Memory: {total_memory:.2f} GB")
            print(f"Allocated Memory: {allocated_memory:.2f} GB")
            print(f"Cached Memory: {cached_memory:.2f} GB")
        else:
            print("Using CPU")
            print(f"PyTorch Version: {torch.__version__}")
        
        print("="*60)
    
    def to_device(self, tensor):
        """
        Move a numpy array or torch tensor to the configured device.
        
        Args:
            tensor: Numpy array or torch tensor
            
        Returns:
            Torch tensor on the configured device
        """
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor).float()
        elif not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor).float()
        
        return tensor.to(self.device)
    
    def to_numpy(self, tensor):
        """
        Convert a torch tensor to numpy array.
        
        Args:
            tensor: Torch tensor
            
        Returns:
            Numpy array
        """
        if isinstance(tensor, torch.Tensor):
            return tensor.cpu().detach().numpy()
        return tensor
    
    def clear_cache(self):
        """Clear GPU cache if using CUDA."""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            print("GPU cache cleared")


# Global device configuration
_device_config = None


def get_device_config(force_cpu: bool = False) -> DeviceConfig:
    """
    Get or create the global device configuration.
    
    Args:
        force_cpu: If True, force CPU usage
        
    Returns:
        DeviceConfig instance
    """
    global _device_config
    if _device_config is None:
        _device_config = DeviceConfig(force_cpu=force_cpu)
    return _device_config


def get_device() -> torch.device:
    """Get the current device."""
    return get_device_config().device


if __name__ == "__main__":
    # Test GPU availability
    config = DeviceConfig()
    
    print("\nTesting GPU operations...")
    if config.device.type == 'cuda':
        # Test tensor operations
        x = torch.randn(1000, 1000, device=config.device)
        y = torch.randn(1000, 1000, device=config.device)
        
        import time
        start = time.time()
        z = torch.matmul(x, y)
        torch.cuda.synchronize()  # Wait for GPU to finish
        elapsed = time.time() - start
        
        print(f"Matrix multiplication (1000x1000): {elapsed*1000:.2f} ms")
        print("GPU is working correctly!")
    else:
        print("No GPU available for testing")
