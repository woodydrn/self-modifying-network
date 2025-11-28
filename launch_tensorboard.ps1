# Launch TensorBoard to view training metrics
# Usage: .\launch_tensorboard.ps1

Write-Host "Launching TensorBoard..." -ForegroundColor Green
Write-Host ""
Write-Host "Open your browser to: http://localhost:6006" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop TensorBoard" -ForegroundColor Yellow
Write-Host ""

& .\.venv\Scripts\python.exe -m tensorboard.main --logdir=runs --host=localhost --port=6006
