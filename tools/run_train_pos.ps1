param(
  [Parameter(Mandatory = $true)]
  [string]$InputJsonl,

  [Parameter(Mandatory = $true)]
  [string]$OutDir,

  [double]$MinConf = 0.7,

  [int]$Epochs = 5,

  [double]$ValRatio = 0.1,

  [int]$Seed = 42,

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

function Get-LineCount {
  param([string]$Path)
  if (!(Test-Path $Path)) {
    return 0
  }
  return (Get-Content -Path $Path | Measure-Object -Line).Lines
}

if (!(Test-Path $VenvPath)) {
  throw "Venv activate script not found: $VenvPath"
}
. $VenvPath

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$trainManifestPath = Join-Path $OutDir "train_manifest.jsonl"
$valManifestPath = Join-Path $OutDir "val_manifest.jsonl"

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
  "--out-train", $trainManifestPath,
  "--out-val", $valManifestPath,
  "--val-ratio", $ValRatio,
  "--seed", $Seed,
  "--min-conf", $MinConf,
  "--grid-w", $GridW,
  "--grid-h", $GridH
)
if (-not [string]::IsNullOrWhiteSpace($DebugDir)) {
  $buildArgs += @("--debug-dir", $DebugDir)
}
python @buildArgs
$trainCount = Get-LineCount -Path $trainManifestPath
$valCount = Get-LineCount -Path $valManifestPath
Write-Host ("train_samples={0} val_samples={1}" -f $trainCount, $valCount)
if ($trainCount -eq 0 -or $valCount -eq 0) {
  Write-Error ("Train/val manifest is empty (train={0}, val={1}). Check -ValRatio/-MinConf/-DebugDir." -f $trainCount, $valCount)
  exit 1
}

Write-Host ""
Write-Host "=== train position model ==="
python train_pos_model.py `
  --train-manifest $trainManifestPath `
  --val-manifest $valManifestPath `
  --grid-w $GridW `
  --grid-h $GridH `
  --img-size $ImgSize `
  --batch-size $BatchSize `
  --epochs $Epochs `
  --lr $Lr `
  --out $OutDir

Write-Host ""
Write-Host "=== outputs ==="
Write-Host "train_manifest: $trainManifestPath"
Write-Host "val_manifest:   $valManifestPath"
Write-Host "config:   $OutDir\\config.json"
Write-Host "metrics:  $OutDir\\metrics.jsonl"
Write-Host "model:    $OutDir\\model.pt"
