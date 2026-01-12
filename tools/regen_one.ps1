param(
  [Parameter(Mandatory=$true)]
  [string]$Video,

  [int]$Fps = 12,

  [string]$VenvPath = ".\.venv\Scripts\Activate.ps1",

  [string]$InferModel = "runs\20260111_190610\model.pt",

  [string]$OutDir = "data\batch3_fps12",

  # infer_hand ROI params
  [double]$y_ratio = 0.80,
  [double]$height_ratio = 0.14,
  [double]$x_margin_ratio = 0.13,
  [double]$x_offset_ratio = 0.09,

  [switch]$IncludePrevAction = $true
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Move to repo root (script dir is tools/)
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

# Activate venv
if (!(Test-Path $VenvPath)) {
  throw "Venv activate script not found: $VenvPath"
}
. $VenvPath

# Resolve video path
$videoPath = (Resolve-Path $Video).Path
$stem = [IO.Path]::GetFileNameWithoutExtension($videoPath)

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$hand    = Join-Path $OutDir "$stem.hand.txt"
$actions = Join-Path $OutDir "$stem.actions.jsonl"
$state   = Join-Path $OutDir "$stem.state_role_v2.jsonl"

Write-Host "=== regen_one ==="
Write-Host "repoRoot: $repoRoot"
Write-Host "video:    $videoPath"
Write-Host "fps:      $Fps"
Write-Host "outDir:   $OutDir"
Write-Host ""

# 1) infer_hand (UTF-8 output)
python -m scripts.infer_hand `
  --video $videoPath `
  --video-fps $Fps `
  --model $InferModel `
  --y-ratio $y_ratio --height-ratio $height_ratio --x-margin-ratio $x_margin_ratio --x-offset-ratio $x_offset_ratio |
  Out-File -FilePath $hand -Encoding utf8

# 2) extract_actions
python -m scripts.extract_actions_from_hand --in $hand --out $actions

# 3) build_state_role_dataset (v2)
$buildArgs = @("--hand", $hand, "--actions", $actions, "--out", $state)
if ($IncludePrevAction) { $buildArgs += "--include-prev-action" }
python -m scripts.build_state_role_dataset @buildArgs

# Summary
$handLines = (Get-Content $hand | Where-Object { $_ -and ($_ -notlike "infer_hand start:*") }).Count
$actionsCnt = (Get-Content $actions).Count
$stateCnt = (Get-Content $state).Count

Write-Host ""
Write-Host "=== summary ==="
Write-Host "hand_lines    = $handLines"
Write-Host "actions_count = $actionsCnt"
Write-Host "state_count   = $stateCnt"
Write-Host "hand_out      = $hand"
Write-Host "actions_out   = $actions"
Write-Host "state_out     = $state"
