param(
  [Parameter(Mandatory = $true)]
  [string]$Model,

  [Parameter(Mandatory = $true)]
  [string]$Manifest,

  [Parameter(Mandatory = $true)]
  [string]$Out,

  [int]$GridW = 18,

  [int]$GridH = 11,

  [int]$ImgSize = 224,

  [int]$TopK = 5,

  [int]$BatchSize = 64,

  [int]$NumWorkers = 0,

  [string]$Device = "cpu",

  [double]$WidthMult = 1.0,

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

python infer_pos_model.py `
  --model $Model `
  --manifest $Manifest `
  --grid-w $GridW `
  --grid-h $GridH `
  --img-size $ImgSize `
  --out $Out `
  --topk $TopK `
  --batch-size $BatchSize `
  --num-workers $NumWorkers `
  --device $Device `
  --width-mult $WidthMult
