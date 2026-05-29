param(
    [switch]$ContinueOnError,
    [switch]$DryRun,
    [string]$Dataset = "compas"
)

$ErrorActionPreference = "Stop"

# In PowerShell 7+, avoid converting native stderr lines into ErrorActionPreference-driven failures.
if ($PSVersionTable.PSVersion.Major -ge 7) {
    $PSNativeCommandUseErrorActionPreference = $false
}

function Get-LambdaSuffix {
    param([double]$Lambda)

    # Match Python formatting: f"{float(lambda):g}" then map decimal point to 'p'.
    $text = [string]::Format("{0:G}", $Lambda)
    return $text.Replace('.', 'p').Replace('-', 'm')
}

function Test-RunExists {
    param(
        [string]$Root,
        [string]$Experiment,
        [double]$Lambda
    )

    if (-not (Test-Path $Root)) {
        return $false
    }

    $experimentDirs = Get-ChildItem -Path $Root -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like ("{0}_*" -f $Experiment) }

    foreach ($dir in $experimentDirs) {
        $cfgPath = Join-Path $dir.FullName ("results\{0}_config.json" -f $Experiment)
        if (-not (Test-Path $cfgPath)) {
            continue
        }

        try {
            $cfg = Get-Content $cfgPath -Raw | ConvertFrom-Json
            $runLambda = [double]$cfg.dataset_config.soft_fairness_settings.lambda
            if ([math]::Abs($runLambda - $Lambda) -lt 1e-9) {
                return $true
            }
        }
        catch {
            continue
        }
    }

    return $false
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$resultsRoot = Join-Path $repoRoot ("results\experiments\{0}" -f $Dataset)
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logRoot = Join-Path $repoRoot "results\experiments\batch_logs"
$batchDir = Join-Path $logRoot ("pareto_grid_{0}_{1}" -f $Dataset, $timestamp)

New-Item -ItemType Directory -Path $batchDir -Force | Out-Null

# Comprehensive lambda grid for a dense Pareto front.
$lambdaGrid = @(0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0)

# Existing runs observed in this repository.
$existingSoftFairness = @(0.5, 0.7, 0.8, 1.0, 1.5, 1.8, 1.9, 2.0)
$existingSoftOnly = @(1.9, 2.0)

$todoSoftFairness = $lambdaGrid | Where-Object { $_ -notin $existingSoftFairness }
$todoSoftOnly = $lambdaGrid | Where-Object { $_ -notin $existingSoftOnly }

$plan = @()
foreach ($lambda in $todoSoftOnly) {
    $plan += [PSCustomObject]@{ Experiment = "soft_fairness_only"; Lambda = $lambda }
}
foreach ($lambda in $todoSoftFairness) {
    $plan += [PSCustomObject]@{ Experiment = "soft_fairness"; Lambda = $lambda }
}

$summaryPath = Join-Path $batchDir "plan_summary.txt"
"Dataset: $Dataset" | Out-File -FilePath $summaryPath -Encoding utf8
"DryRun: $DryRun" | Add-Content -Path $summaryPath
"Total planned tasks: $($plan.Count)" | Add-Content -Path $summaryPath
"---" | Add-Content -Path $summaryPath
$plan | ForEach-Object { "{0} lambda={1}" -f $_.Experiment, $_.Lambda } | Add-Content -Path $summaryPath

Write-Host "Pareto grid batch directory: $batchDir"
Write-Host "Total planned tasks: $($plan.Count)"

$failed = @()
$ran = 0
$skipped = 0

foreach ($task in $plan) {
    $exp = $task.Experiment
    $lambda = [double]$task.Lambda

    if (Test-RunExists -Root $resultsRoot -Experiment $exp -Lambda $lambda) {
        Write-Host "SKIP existing run: $exp lambda=$lambda"
        $skipped += 1
        continue
    }

    $cmd = "poetry run python src/experiments/main_runner.py --dataset $Dataset --experiment $exp --lambda-fairness $lambda"
    $cmdArgs = @(
        "run",
        "python",
        "src/experiments/main_runner.py",
        "--dataset", $Dataset,
        "--experiment", $exp,
        "--lambda-fairness", "$lambda"
    )

    $runLabel = "{0}_lambda_{1}" -f $exp, (Get-LambdaSuffix -Lambda $lambda)
    $logPath = Join-Path $batchDir ("{0}.log" -f $runLabel)

    $startStamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$startStamp] START $cmd" | Out-File -FilePath $logPath -Encoding utf8
    Write-Host "RUN $cmd"

    if ($DryRun) {
        "DRY RUN - command not executed" | Add-Content -Path $logPath
        $skipped += 1
        continue
    }

    Push-Location $repoRoot
    try {
        # In Windows PowerShell, native stderr can surface as NativeCommandError records.
        # Keep warnings non-terminating here and rely on process exit code for failure.
        $prevErrorActionPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        & poetry @cmdArgs 2>&1 | Tee-Object -FilePath $logPath -Append
        $cmdExitCode = $LASTEXITCODE
        $ErrorActionPreference = $prevErrorActionPreference

        if ($cmdExitCode -ne 0) {
            throw "Command failed with exit code $cmdExitCode"
        }
        $ran += 1
    }
    catch {
        $failed += $runLabel
        "ERROR: $_" | Add-Content -Path $logPath
        if (-not $ContinueOnError) {
            Pop-Location
            throw
        }
    }
    finally {
        Pop-Location
    }
}

Write-Host "Completed. Ran=$ran Skipped=$skipped Failed=$($failed.Count)"
if ($failed.Count -gt 0) {
    Write-Host "Failed tasks:"
    $failed | ForEach-Object { Write-Host "  $_" }
    if (-not $ContinueOnError) {
        exit 1
    }
}
