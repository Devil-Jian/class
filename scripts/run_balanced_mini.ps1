param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("ncct", "cect_real", "cect_syn", "cect_real_syn", "cect_real_aug", "all")]
    [string]$Group = "cect_real",

    [Parameter(Mandatory=$false)]
    [string]$Model = "resnet18_mil",

    [Parameter(Mandatory=$false)]
    [int]$Epochs = 30,

    [Parameter(Mandatory=$false)]
    [int]$BatchSize = 4,

    [Parameter(Mandatory=$false)]
    [int]$Workers = 2,

    [Parameter(Mandatory=$false)]
    [string]$Device = "cuda:0",

    [Parameter(Mandatory=$false)]
    [switch]$FreezeEncoder,

    [Parameter(Mandatory=$false)]
    [string]$WorkDir = "R:\downstream_classification",

    [Parameter(Mandatory=$false)]
    [ValidateSet("full", "balanced")]
    [string]$MixMode = "full"
)

$ErrorActionPreference = "Stop"

# Ensure local src directory is prioritized over site-packages
$srcDir = (Resolve-Path "$PSScriptRoot\..\src").Path
$repoRoot = (Resolve-Path "$PSScriptRoot\..").Path
$env:PYTHONPATH = "$srcDir;$env:PYTHONPATH"

# Use domestic HuggingFace mirror to prevent SSL/EOF download failure in China
if (-not $env:HF_ENDPOINT) {
    $env:HF_ENDPOINT = "https://hf-mirror.com"
}

# Normalize model name so resnet50 and resnet50_mil both point to resnet50_mil
$Model = switch ($Model) {
    "resnet50" { "resnet50_mil" }
    "resnet18" { "resnet18_mil" }
    "deit"     { "deit_mil" }
    "dinov2"   { "dinov2s_mil" }
    default    { $Model }
}

# Find manifest file (check repo manifests_mini first, then WorkDir)
$manifestPath = "$repoRoot\manifests_mini\cases_balanced_mini.jsonl"
if (-not (Test-Path $manifestPath)) {
    $manifestPath = "$WorkDir\manifests\cases_balanced_mini.jsonl"
}

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ">> Windows Balanced-Mini (245 Cases, 5 Classes 1:1:1:1:1) Runner" -ForegroundColor Cyan
Write-Host ">> Target Group: $Group | Model: $Model | Epochs: $Epochs" -ForegroundColor Cyan
Write-Host ">> Device: $Device | Batch Size: $BatchSize | Workers: $Workers" -ForegroundColor Cyan
Write-Host ">> Freeze Encoder: $FreezeEncoder | Mix Mode: $MixMode" -ForegroundColor Cyan
Write-Host ">> Manifest: $manifestPath" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

$groupsToRun = if ($Group -eq "all") {
    @("ncct", "cect_real", "cect_syn", "cect_real_syn", "cect_real_aug")
} else {
    @($Group)
}

foreach ($g in $groupsToRun) {
    Write-Host "`n>> [Starting Mini Experiment] Group: $g on $Device ..." -ForegroundColor Green
    $outDir = "$WorkDir\experiments_mini\$Model\$g"
    
    $cmdArgs = @(
        "train",
        "--manifest", $manifestPath,
        "--labels", "$WorkDir\manifests\labels.json",
        "--selection", "$WorkDir\manifests\matched_bags.json",
        "--png-index", "$WorkDir\png_dataset\png_index.json",
        "--stage", "selected",
        "--group", $g,
        "--model", $Model,
        "--pretrained",
        "--batch-size", $BatchSize,
        "--epochs", $Epochs,
        "--encode-chunk", "8",
        "--workers", $Workers,
        "--device", $Device,
        "--class-weight", "none",
        "--mix-mode", $MixMode,
        "--out", $outDir
    )

    if ($FreezeEncoder) {
        $cmdArgs += "--freeze-encoder"
    }

    python -m cectcls.cli @cmdArgs
    Write-Host ">> [Finished Mini Experiment] Group: $g successfully! Output: $outDir" -ForegroundColor Green
}
