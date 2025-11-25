<# run_tbq_solver_h3d.ps1
- hand3d_dyn 파이프라인(DSHOW) + Charuco+QR T_B<-Q 추출
- 고정: left=1, right=2, 1280x720@24
- 실행: .\run_tbq_solver_h3d.ps1
#>

$PythonCmd   = "py -3.11"
$ProjectRoot = (Get-Location).Path

# 경로
$CalibPath = "stereo_pairs\calib_charuco_stereo.npz"
$OutNPZ    = "transforms\T_BQ.npz"
$OutJSON   = "transforms\T_BQ.json"

# 카메라(h3d와 동일 톤) — 고정(요청사항)
$LeftCamId  = 1
$RightCamId = 2
$Width      = 1280
$Height     = 720
$Fps        = 24

# h3d 런타임
$Alpha       = 0.25
$MinConf     = 0.35
$DetectScale = 1.0
$DetectEvery = 1
$MaxPairDtMs = 30

# Charuco/QR 실측
$DictId     = 4          # 5x5_50
$SquaresX   = 5
$SquaresY   = 7
$SquareLen  = 0.018      # 18 mm
$MarkerLen  = 0.013      # 13 mm
$QrSize     = 0.065      # 65 mm

$ArgsList = @(
  "`"$ProjectRoot\tbq_auto_solver_h3d.py`"",
  "--calib", "`"$CalibPath`"",
  "--left-id", $LeftCamId, "--right-id", $RightCamId,
  "--width", $Width, "--height", $Height, "--fps", $Fps,
  "--alpha", $Alpha, "--min-conf", $MinConf,
  "--detect-scale", $DetectScale, "--detect-every", $DetectEvery,
  "--max-pair-dt-ms", $MaxPairDtMs,
  "--dict", $DictId,
  "--squares-x", $SquaresX, "--squares-y", $SquaresY,
  "--square-len", $SquareLen, "--marker-len", $MarkerLen,
  "--qr-size", $QrSize,
  "--out-npz", "`"$OutNPZ`"", "--out-json", "`"$OutJSON`"",
  "--preview"
)

Write-Host "=== tbq_auto_solver_h3d (left=1, right=2, 1280x720@24) ===" -ForegroundColor Cyan
Invoke-Expression "$PythonCmd $($ArgsList -join ' ')"
