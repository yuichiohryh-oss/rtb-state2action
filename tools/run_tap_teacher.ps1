param(
  [int]$Seconds = 180,

  [string]$OutDir = "runs",

  [string]$Serial = "",

  [string]$Adb = "",

  [string]$Scrcpy = "",

  [string]$VenvPath = ".\\.venv\\Scripts\\Activate.ps1",

  [string[]]$LabelArgs = @()
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

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runDir = Join-Path $OutDir "run_$timestamp"
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

Write-Host "=== tap teacher ==="
Write-Host "repoRoot: $repoRoot"
Write-Host "runDir:   $runDir"
Write-Host "seconds:  $Seconds"
if ($Serial) { Write-Host "serial:   $Serial" }
Write-Host ""

$captureArgs = @("--out", $runDir, "--record-seconds", $Seconds)
if ($Serial) { $captureArgs += @("--serial", $Serial) }
if ($Adb) { $captureArgs += @("--adb", $Adb) }
if ($Scrcpy) { $captureArgs += @("--scrcpy", $Scrcpy) }
python tools/capture_scrcpy_taps.py @captureArgs

$tapsCsv = Join-Path $runDir "taps.csv"
$metaJson = Join-Path $runDir "meta.json"
$actionsTap = Join-Path $runDir "actions_tap.jsonl"
$actionsTapPos = Join-Path $runDir "actions_tap_pos.jsonl"
$debugDir = Join-Path $runDir "debug_pos"

python tools/taps_to_actions_jsonl.py --taps $tapsCsv --meta $metaJson --out $actionsTap

$labelArgs = @(
  "--video", (Join-Path $runDir "video.mp4"),
  "--actions", $actionsTap,
  "--out", $actionsTapPos,
  "--grid-w", "18",
  "--grid-h", "11",
  "--dt-ms", "300",
  "--self-side-only",
  "--debug-dir", $debugDir
)
if ($LabelArgs.Count -gt 0) { $labelArgs += $LabelArgs }
python tools/label_position_from_diff.py @labelArgs

Write-Host ""
Write-Host "=== outputs ==="
Write-Host "video:         $runDir\\video.mp4"
Write-Host "taps:          $tapsCsv"
Write-Host "meta:          $metaJson"
Write-Host "actions_tap:   $actionsTap"
Write-Host "actions_pos:   $actionsTapPos"
Write-Host "debug_pos:     $debugDir"
