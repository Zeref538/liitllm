# Wait for the running prep session to release the runner lock, confirm it
# actually succeeded, then start the seed-2 training chain (baseline2, ablation2).
#
# Deliberately does NOT start training if prep failed: the training kernels mount
# the prep output, so launching them without a corpus wastes a GPU session and
# leaves a confusing half-run to clean up.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONUTF8 = 1

$lock = Join-Path $PSScriptRoot ".run_pipeline.lock"
while (Test-Path $lock) { Start-Sleep -Seconds 60 }

$tail = Get-Content (Join-Path $PSScriptRoot "run_pipeline.log") -Tail 40
if (-not ($tail -match "=== prep finished ===")) {
    "[chain] prep did NOT finish cleanly - not starting training. Last lines:"
    $tail | Select-Object -Last 10
    exit 1
}

"[chain] prep finished; starting seed-2 chain"
python run_pipeline.py --arm all
