# ===============================================
# run_hl_sender.ps1
# - PC -> HoloLens 손 3D 송신 실행 스크립트
# - 위쪽 변수만 바꾸면 끝. 파워쉘 매개변수로도 덮어쓰기 가능
# ===============================================

param(
  [int]$LeftId    = 1,
  [int]$RightId   = 2,
  [int]$Width     = 1280,
  [int]$Height    = 720,
  [int]$Fps       = 24,
  [string]$Hand   = "right",
  [double]$MinConf = 0.35,
  [double]$DetectScale = 1.0,
  [int]$DetectEvery = 1,

  # 경로 & 네트워크
  [string]$BaseDir   = "C:\Users\jykan\Project\MGDIProject\WebcamAccess",
  [string]$CalibRel  = "stereo_pairs\calib_charuco_stereo.npz",
  [string]$TransformRel = "stereo_pairs\T_HC.npz",
  [string]$Udp       = "192.168.50.212:19560",

  # venv를 명시하고 싶으면 여기에 (비어있으면 현재 파워쉘의 python을 사용)
  [string]$VenvActivate = ""
)

$ErrorActionPreference = "Stop"

# python 경로 확인(가상환경 옵션)
if ($VenvActivate -and (Test-Path $VenvActivate)) {
  Write-Host "[INFO] Activate venv: $VenvActivate" -ForegroundColor Cyan
  & $VenvActivate
}

# 경로 구성
$CalibPath = Join-Path $BaseDir $CalibRel
$TransPath = Join-Path $BaseDir $TransformRel
$SenderPy  = Join-Path $BaseDir "hl_sender.py"

# 존재 확인
if (!(Test-Path $SenderPy)) { Write-Host "[ERR] hl_sender.py not found: $SenderPy" -ForegroundColor Red; exit 1 }
if (!(Test-Path $CalibPath)) { Write-Host "[ERR] calib npz not found: $CalibPath" -ForegroundColor Red; exit 1 }
if (!(Test-Path $TransPath)) { Write-Host "[ERR] T_HC npz not found: $TransPath" -ForegroundColor Red; exit 1 }

# 설정 에코
Write-Host "=== HL Sender Config ===" -ForegroundColor Yellow
Write-Host ("Left/Right:      {0} / {1}" -f $LeftId, $RightId)
Write-Host ("Res/FPS:         {0}x{1} @ {2}" -f $Width, $Height, $Fps)
Write-Host ("Hand/MinConf:    {0} / {1}" -f $Hand, $MinConf)
Write-Host ("DetectScale/Ev:  {0} / {1}" -f $DetectScale, $DetectEvery)
Write-Host ("Calib:           {0}" -f $CalibPath)
Write-Host ("Transform:       {0}" -f $TransPath)
Write-Host ("UDP:             {0}" -f $Udp)
Write-Host "========================" -ForegroundColor Yellow

# 인자 배열 구성
$argsList = @(
  $SenderPy,
  "--left-id", $LeftId,
  "--right-id", $RightId,
  "--width", $Width,
  "--height", $Height,
  "--fps", $Fps,
  "--calib", $CalibPath,
  "--transform", $TransPath,
  "--udp", $Udp,
  "--hand", $Hand,
  "--min-conf", $MinConf,
  "--detect-scale", $DetectScale,
  "--detect-every", $DetectEvery
)

# 실행
& python @argsList
if ($LASTEXITCODE -ne 0) {
  Write-Host ("[ERR] hl_sender.py failed (exit={0})" -f $LASTEXITCODE) -ForegroundColor Red
  exit $LASTEXITCODE
}
