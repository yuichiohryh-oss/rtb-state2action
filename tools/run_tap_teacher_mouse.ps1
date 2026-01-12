param(
  [int]$Seconds = 180,

  [string]$OutDir = "runs",

  [string]$Scrcpy = "",

  [int]$GridW = 18,

  [int]$GridH = 11,

  [int]$DtMs = 300,

  [switch]$Verbose
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Move to repo root (script dir is tools/)
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runDir = Join-Path $OutDir "run_$timestamp"
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

Write-Host "=== tap teacher (mouse) ==="
Write-Host "repoRoot: $repoRoot"
Write-Host "runDir:   $runDir"
Write-Host "seconds:  $Seconds"
if ($Scrcpy) { Write-Host "scrcpy:   $Scrcpy" }
Write-Host ""

$captureArgs = @("--out", $runDir, "--record-seconds", $Seconds)
if ($Scrcpy) { $captureArgs += @("--scrcpy", $Scrcpy) }
$captureArgs += "--playback"
if ($Verbose) { $captureArgs += "--verbose" }
python tools/capture_scrcpy_mouse.py @captureArgs
if ($LASTEXITCODE -ne 0) { throw "capture_scrcpy_mouse.py failed with exit code $LASTEXITCODE" }

$tapsCsv = Join-Path $runDir "taps.csv"
$actionsTap = Join-Path $runDir "actions_tap.jsonl"
$actionsTapPos = Join-Path $runDir "actions_tap_pos.jsonl"
$debugDir = Join-Path $runDir "debug_pos"

python tools/taps_to_actions_jsonl.py --taps $tapsCsv --out $actionsTap
if ($LASTEXITCODE -ne 0) { throw "taps_to_actions_jsonl.py failed with exit code $LASTEXITCODE" }

$labelArgs = @(
  "--video", (Join-Path $runDir "video.mp4"),
  "--actions", $actionsTap,
  "--out", $actionsTapPos,
  "--grid-w", $GridW,
  "--grid-h", $GridH,
  "--dt-ms", $DtMs,
  "--debug-dir", $debugDir
)
python tools/label_position_from_diff.py @labelArgs
if ($LASTEXITCODE -ne 0) { throw "label_position_from_diff.py failed with exit code $LASTEXITCODE" }

Write-Host ""
Write-Host "=== outputs ==="
Write-Host "video:         $runDir\\video.mp4"
Write-Host "taps:          $tapsCsv"
Write-Host "actions_tap:   $actionsTap"
Write-Host "actions_pos:   $actionsTapPos"
Write-Host "debug_pos:     $debugDir"
