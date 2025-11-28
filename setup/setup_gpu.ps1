# GPU Setup Script for Self-Modifying Network
# Run this script to set up PyTorch with CUDA support

Write-Host "=" * 70
Write-Host "Self-Modifying Network - GPU Setup" -ForegroundColor Cyan
Write-Host "=" * 70
Write-Host ""

# Check Python version
Write-Host "Checking Python version..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
Write-Host "  $pythonVersion"

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python is not installed or not in PATH!" -ForegroundColor Red
    Write-Host "Please install Python 3.8+ from https://www.python.org/downloads/"
    exit 1
}

Write-Host ""

# Ask user for CUDA version
Write-Host "Which CUDA version do you want to use?" -ForegroundColor Yellow
Write-Host "  1. CUDA 12.1 (Recommended for RTX 40/50 series)"
Write-Host "  2. CUDA 11.8 (For older GPUs)"
Write-Host "  3. CPU only (No GPU support)"
Write-Host ""
$choice = Read-Host "Enter choice (1, 2, or 3)"

switch ($choice) {
    "1" {
        $cudaVersion = "cu121"
        $cudaUrl = "https://download.pytorch.org/whl/cu121"
        Write-Host "Installing PyTorch with CUDA 12.1..." -ForegroundColor Green
    }
    "2" {
        $cudaVersion = "cu118"
        $cudaUrl = "https://download.pytorch.org/whl/cu118"
        Write-Host "Installing PyTorch with CUDA 11.8..." -ForegroundColor Green
    }
    "3" {
        $cudaVersion = "cpu"
        $cudaUrl = "https://download.pytorch.org/whl/cpu"
        Write-Host "Installing PyTorch (CPU only)..." -ForegroundColor Green
    }
    default {
        Write-Host "Invalid choice. Defaulting to CUDA 12.1..." -ForegroundColor Yellow
        $cudaVersion = "cu121"
        $cudaUrl = "https://download.pytorch.org/whl/cu121"
    }
}

Write-Host ""

# Uninstall existing PyTorch
Write-Host "Uninstalling any existing PyTorch installations..." -ForegroundColor Yellow
pip uninstall -y torch torchvision torchaudio 2>$null

Write-Host ""

# Install PyTorch
Write-Host "Installing PyTorch with $cudaVersion..." -ForegroundColor Yellow
pip install torch torchvision torchaudio --index-url $cudaUrl

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install PyTorch!" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Install other dependencies
Write-Host "Installing other dependencies from requirements.txt..." -ForegroundColor Yellow
pip install -r requirements.txt

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install dependencies!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "=" * 70
Write-Host "Installation Complete!" -ForegroundColor Green
Write-Host "=" * 70
Write-Host ""

# Run GPU check
if ($cudaVersion -ne "cpu") {
    Write-Host "Running GPU verification..." -ForegroundColor Yellow
    Write-Host ""
    python check_gpu.py
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "=" * 70
        Write-Host "SUCCESS! Your GPU is ready to use!" -ForegroundColor Green
        Write-Host "=" * 70
        Write-Host ""
        Write-Host "You can now run:" -ForegroundColor Cyan
        Write-Host "  python train.py           - Train regression model"
        Write-Host "  python continuous_train.py - Continuous learning mode"
        Write-Host "  python train-2.py          - Math operations training"
    } else {
        Write-Host ""
        Write-Host "WARNING: GPU verification failed!" -ForegroundColor Yellow
        Write-Host "The network will fall back to CPU." -ForegroundColor Yellow
        Write-Host "See GPU_SETUP.md for troubleshooting steps."
    }
} else {
    Write-Host "CPU-only installation complete." -ForegroundColor Green
    Write-Host ""
    Write-Host "You can now run:" -ForegroundColor Cyan
    Write-Host "  python train.py           - Train regression model"
    Write-Host "  python continuous_train.py - Continuous learning mode"
    Write-Host "  python train-2.py          - Math operations training"
}

Write-Host ""
