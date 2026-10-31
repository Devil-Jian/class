param(
    [Parameter(Mandatory=$false)]
    [string]$WorkDir = "R:\downstream_classification",

    [Parameter(Mandatory=$false)]
    [string]$Device = "cuda:0",

    [Parameter(Mandatory=$false)]
    [int]$BatchSize = 4,

    [Parameter(Mandatory=$false)]
    [int]$Workers = 2,

    [Parameter(Mandatory=$false)]
    [switch]$Force
)

$ErrorActionPreference = "Stop"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ">> Paper Results Generator: Evaluating All best.pt on Test Set" -ForegroundColor Cyan
Write-Host ">> WorkDir: $WorkDir | Device: $Device" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

$scriptPath = Join-Path $PSScriptRoot "evaluate_all_checkpoints.py"

$cmdArgs = @(
    $scriptPath,
    "--work-dir", $WorkDir,
    "--device", $Device,
    "--batch-size", $BatchSize,
    "--workers", $Workers
)

if ($Force) {
    $cmdArgs += "--force"
}

python @cmdArgs
