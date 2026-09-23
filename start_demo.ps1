# start_demo.ps1 - helper to start Flask API and static server and open browser
# Run this from the repository root in PowerShell: .\start_demo.ps1

$repo = "C:\Users\user\OneDrive\Desktop\Axlero Internship\FedMed\fedmed-federated-learning.worktrees\fedmed-federated-learning-privacy-ml"
$venvActivate = Join-Path $repo ".venv\Scripts\Activate.ps1"

# Start Flask API in a new PowerShell window
Start-Process powershell -ArgumentList '-NoExit', "-Command cd `"$repo`"; if (Test-Path `"$venvActivate`") { . `"$venvActivate`" }; python -m fedmed.api.run_demo_api"

Start-Sleep -Milliseconds 800

# Start static server (serve fedmed/docs) in another new PowerShell window
Start-Process powershell -ArgumentList '-NoExit', "-Command cd `"$repo`"; python -m http.server 8000 --directory `"$repo\fedmed\docs`""

Start-Sleep -Seconds 1

# Open the one-pager in the default browser
Start-Process "http://localhost:8000/one_pager.html"

Write-Host "Started demo: Flask API on port 5000 and static server on port 8000. Browser opened." -ForegroundColor Green
