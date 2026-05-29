param(
    [switch]$ContinueOnError
)

$ErrorActionPreference = "Stop"

$datasets = @("law", "compas", "dutch", "bank")
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$repoRoot = Split-Path -Parent $PSScriptRoot
$logRoot = Join-Path $repoRoot "results\experiments\batch_logs"
$batchDir = Join-Path $logRoot "all_datasets_$timestamp"

New-Item -ItemType Directory -Path $batchDir -Force | Out-Null

Write-Host "Starting full real-dataset experiment batch..."
Write-Host "Batch log directory: $batchDir"

$failed = @()

foreach ($dataset in $datasets) {
    $logPath = Join-Path $batchDir "$dataset.log"
    $startStamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

    "[$startStamp] START dataset=$dataset" | Out-File -FilePath $logPath -Encoding utf8
    Write-Host "Running dataset=$dataset ..."

    # Run from repo root so relative paths used by runner stay stable.
    Push-Location $repoRoot
    try {
        poetry run python src/experiments/main_runner.py --dataset $dataset --experiment all 2>&1 |
            Tee-Object -FilePath $logPath -Append

        if ($LASTEXITCODE -ne 0) {
            throw "Command failed with exit code $LASTEXITCODE"
        }

        $endStamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        "[$endStamp] SUCCESS dataset=$dataset" | Out-File -FilePath $logPath -Append -Encoding utf8
        Write-Host "Completed dataset=$dataset"
    }
    catch {
        $failed += $dataset
        $endStamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        "[$endStamp] FAILED dataset=$dataset :: $($_.Exception.Message)" | Out-File -FilePath $logPath -Append -Encoding utf8
        Write-Host "Failed dataset=$dataset" -ForegroundColor Red

        if (-not $ContinueOnError) {
            Write-Host "Stopping batch after first failure. Use -ContinueOnError to run all datasets." -ForegroundColor Yellow
            exit 1
        }
    }
    finally {
        Pop-Location
    }
}

Write-Host ""
Write-Host "Batch run finished."
Write-Host "Logs: $batchDir"

if ($failed.Count -gt 0) {
    Write-Host "Failed datasets: $($failed -join ', ')" -ForegroundColor Red
    exit 1
}

Write-Host "All datasets completed successfully." -ForegroundColor Green
exit 0
