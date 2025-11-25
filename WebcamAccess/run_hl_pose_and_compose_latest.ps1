# ============================================
# run_hl_pose_and_compose_latest.ps1  (no-else, safe braces)
# 1) pick latest "cap_YYYYMMDD_HHMMSS" under anchor_caps
# 2) compute T_H_from_B.npz from photo.png + pose.json
#    - hl_board_pose_from_photo.py 내부에서 HL(유니티 y-up) → OpenCV 카메라(y-down)로 자동 변환
# 3) compose T_HC.npz / T_HC.json from T_H_from_B and T_C_from_B
#    - 보정 옵션: --fix-board-z (none|ax|ay), --fix-board-rot (none|rz90|rz-90|rz180)
# ============================================

$ErrorActionPreference = "Stop"

# --- paths (edit if needed) ---
$baseDir   = "C:\Users\jykan\Project\MGDIProject\WebcamAccess"
$anchorDir = Join-Path $baseDir "anchor_caps"
$boardCal  = Join-Path $baseDir "stereo_pairs\calib_charuco_stereo.npz"
$TCBPath   = Join-Path $baseDir "stereo_pairs\T_C_from_B.npz"
$THBOut    = Join-Path $baseDir "stereo_pairs\T_H_from_B.npz"
$THCOut    = Join-Path $baseDir "stereo_pairs\T_HC.npz"
$THCJson   = Join-Path $baseDir "stereo_pairs\T_HC.json"

$pyPoseScript    = Join-Path $baseDir "hl_board_pose_from_photo.py"
$pyComposeScript = Join-Path $baseDir "compose_T_HC.py"

# --- board-frame fix mode for H-side ---
#  └─ Z-플립:   "none" | "ax" | "ay"   (우측곱 180°)
#  └─ Z-회전:   "none" | "rz90" | "rz-90" | "rz180"  (우측곱)
$FixBoardZ   = "none"
$FixBoardRot = "none"

function Fail($msg) { Write-Host "[ERR] $msg" -ForegroundColor Red; exit 1 }
function Ok($msg)   { Write-Host $msg -ForegroundColor Green }

# --- sanity checks ---
if (!(Test-Path $anchorDir)) { Fail "anchor_caps folder not found: $anchorDir" }
if (!(Test-Path $boardCal))  { Fail "calib_charuco_stereo.npz not found: $boardCal" }
if (!(Test-Path $TCBPath))   { Fail "T_C_from_B.npz not found: $TCBPath (run PC board pose step first)" }
if (!(Test-Path $pyPoseScript))    { Fail "hl_board_pose_from_photo.py not found: $pyPoseScript" }
if (!(Test-Path $pyComposeScript)) { Fail "compose_T_HC.py not found: $pyComposeScript" }

# --- pick latest session folder ---
$latest = Get-ChildItem -Directory $anchorDir | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $latest) { Fail "no session folder under anchor_caps" }
$session = $latest.FullName
Write-Host ("Latest session: {0}" -f $session) -ForegroundColor Cyan

# --- input files ---
$photo = Join-Path $session "photo.png"
$pose  = Join-Path $session "pose.json"
if (!(Test-Path $photo)) { Fail ("missing photo.png: {0}" -f $photo) }
if (!(Test-Path $pose))  { Fail ("missing pose.json: {0}" -f $pose) }

# --- STEP 1/2: compute T_H_from_B ---
Write-Host "`n[STEP 1/2] compute T_H_from_B (photo -> board pose in H) ..." -ForegroundColor Yellow
$poseArgs = @(
  $pyPoseScript,
  "--photo", $photo,
  "--pose",  $pose,
  "--board-meta-from", $boardCal,
  "--out", $THBOut
)
& python @poseArgs
if ($LASTEXITCODE -ne 0) { Fail ("hl_board_pose_from_photo.py failed (exit={0})" -f $LASTEXITCODE) }
if (!(Test-Path $THBOut)) { Fail ("output not found: {0}" -f $THBOut) }
Ok ("OK: created {0}" -f $THBOut)

# --- STEP 2/2: compose T_HC (apply optional H-side board-frame fix) ---
Write-Host "`n[STEP 2/2] compose T_HC = T_H_from_B * inv(T_C_from_B) ..." -ForegroundColor Yellow
Write-Host ("   fix-board-z   = {0}" -f $FixBoardZ)
Write-Host ("   fix-board-rot = {0}" -f $FixBoardRot)

$compArgs = @(
  $pyComposeScript,
  "--THB", $THBOut,
  "--TCB", $TCBPath,
  "--fix-board-z",   $FixBoardZ,      # ax|ay|none
  "--fix-board-rot", $FixBoardRot,    # rz90|rz-90|rz180|none
  "--out", $THCOut,
  "--json-out", $THCJson
)
& python @compArgs
if ($LASTEXITCODE -ne 0) { Fail ("compose_T_HC.py failed (exit={0})" -f $LASTEXITCODE) }
if (!(Test-Path $THCOut))  { Fail ("output not found: {0}" -f $THCOut) }
if (!(Test-Path $THCJson)) { Fail ("json output not found: {0}" -f $THCJson) }
Ok ("DONE: created final transform {0}" -f $THCOut)

# --- summary ---
Write-Host "`n[SUMMARY]" -ForegroundColor Cyan
Write-Host ("  session : {0}" -f $session)
Write-Host ("  input   : {0}, {1}" -f $photo, $pose)
Write-Host ("  board   : {0}" -f $boardCal)
Write-Host ("  T_CB    : {0}" -f $TCBPath)
Write-Host ("  T_HB    : {0}" -f $THBOut)
Write-Host ("  T_HC    : {0}" -f $THCOut)
Write-Host ("  T_HC.js : {0}" -f $THCJson)
Write-Host ("  fix-z   : {0}" -f $FixBoardZ)
Write-Host ("  fix-rot : {0}" -f $FixBoardRot)
