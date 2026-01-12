param(
  [Parameter(Mandatory = $true)]
  [string]$InputJsonl,

  [Parameter(Mandatory = $true)]
  [string]$OutDir,

  [double]$MinConf = 0.7,

  [int]$Epochs = 5,

  [int]$GridW = 18,

  [int]$GridH = 11,

  [int]$ImgSize = 224,

  [int]$BatchSize = 32,

  [double]$Lr = 1e-3,

  [string]$DebugDir = "",

  [string]$VenvPath = ".\\.venv\\Scripts\\Activate.ps1"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

if (!(Test-Path $VenvPath)) {
  throw "Venv activate script not found: $VenvPath"
}
. $VenvPath

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$manifestPath = Join-Path $OutDir "manifest.jsonl"

if ([string]::IsNullOrWhiteSpace($DebugDir)) {
  $candidateDebugDir = Join-Path (Split-Path $InputJsonl -Parent) "debug_pos"
  if (Test-Path $candidateDebugDir) {
    $DebugDir = $candidateDebugDir
  }
}

Write-Host "=== build position manifest ==="
$buildArgs = @(
  "tools/build_pos_dataset.py",
  "--input", $InputJsonl,
  "--out", $manifestPath,
  "--min-conf", $MinConf,
  "--grid-w", $GridW,
  "--grid-h", $GridH
)
if (-not [string]::IsNullOrWhiteSpace($DebugDir)) {
  $buildArgs += @("--debug-dir", $DebugDir)
}
python @buildArgs

Write-Host ""
Write-Host "=== train position model ==="
python train_pos_model.py `
  --train-manifest $manifestPath `
  --val-manifest $manifestPath `
  --grid-w $GridW `
  --grid-h $GridH `
  --img-size $ImgSize `
  --batch-size $BatchSize `
  --epochs $Epochs `
  --lr $Lr `
  --out $OutDir

Write-Host ""
Write-Host "=== outputs ==="
Write-Host "manifest: $manifestPath"
Write-Host "config:   $OutDir\\config.json"
Write-Host "metrics:  $OutDir\\metrics.jsonl"
Write-Host "model:    $OutDir\\model.pt"
