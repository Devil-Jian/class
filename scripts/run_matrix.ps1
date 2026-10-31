param(
    [Parameter(Mandatory=$false)]
    [string]$Model = "all",

    [Parameter(Mandatory=$false)]
    [string]$Group = "all",

    [Parameter(Mandatory=$false)]
    [int]$Epochs = 30,

    [Parameter(Mandatory=$false)]
    [int]$BatchSize = 4,

    [Parameter(Mandatory=$false)]
    [int]$Workers = 2,

    [Parameter(Mandatory=$false)]
    [string]$Device = "cuda:0",

    [Parameter(Mandatory=$false)]
    [string]$WorkDir = "R:\downstream_classification"
)

$ErrorActionPreference = "Stop"

$srcDir = (Resolve-Path "$PSScriptRoot\..\src").Path
$repoRoot = (Resolve-Path "$PSScriptRoot\..").Path
$env:PYTHONPATH = "$srcDir;$env:PYTHONPATH"

if (-not $env:HF_ENDPOINT) {
    $env:HF_ENDPOINT = "https://hf-mirror.com"
}

$manifestPath = "$repoRoot\manifests_mini\cases_balanced_mini.jsonl"
if (-not (Test-Path $manifestPath)) {
    $manifestPath = "$WorkDir\manifests\cases_balanced_mini.jsonl"
}

$modelsToRun = if ($Model -eq "all") {
    @("dinov2s_mil", "resnet50_mil", "deit_mil")
} else {
    @($Model)
}

$groupsToRun = if ($Group -eq "all") {
    @("cect_real", "cect_syn", "cect_real_syn", "cect_real_aug")
} elseif ($Group -eq "cect_aug") {
    @("cect_real_aug")
} else {
    @($Group)
}

Write-Host "==========================================================================" -ForegroundColor Cyan
Write-Host ">> Windows Matrix Training Runner (Auto Test Evaluation)" -ForegroundColor Cyan
Write-Host ">> Models: $($modelsToRun -join ', ')" -ForegroundColor Cyan
Write-Host ">> Groups: $($groupsToRun -join ', ')" -ForegroundColor Cyan
Write-Host ">> Device: $Device | Epochs: $Epochs | Batch Size: $BatchSize" -ForegroundColor Cyan
Write-Host ">> Manifest: $manifestPath" -ForegroundColor Cyan
Write-Host "==========================================================================" -ForegroundColor Cyan

foreach ($m in $modelsToRun) {
    foreach ($g in $groupsToRun) {
        $outDir = "$WorkDir\experiments_mini\$m\$g"
        
        Write-Host "`n>> [Starting Experiment] Model: $m | Group: $g on $Device ..." -ForegroundColor Green
        
        $cmdArgs = @(
            "train",
            "--manifest", $manifestPath,
            "--labels", "$WorkDir\manifests\labels.json",
            "--selection", "$WorkDir\manifests\matched_bags.json",
            "--png-index", "$WorkDir\png_dataset\png_index.json",
            "--stage", "selected",
            "--group", $g,
            "--model", $m,
            "--pretrained",
            "--freeze-encoder",
            "--batch-size", $BatchSize,
            "--epochs", $Epochs,
            "--encode-chunk", "8",
            "--workers", $Workers,
            "--device", $Device,
            "--class-weight", "none",
            "--mix-mode", "full",
            "--out", $outDir
        )

        python -m cectcls.cli @cmdArgs
        Write-Host ">> [Completed] Experiment ($m / $g) successfully!" -ForegroundColor Green
    }
}
