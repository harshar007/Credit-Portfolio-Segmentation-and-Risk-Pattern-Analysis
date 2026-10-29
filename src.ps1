# PowerShell runner for Credit Portfolio Segmentation and Risk Pattern Analysis
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "Executing Credit Portfolio Segmentation Pipeline (src)" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan

python -m src $args
