0단계
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv311\Scripts\Activate.ps1

1단계. 웹캠(PC) 스테레오 보정
# 보정 이미지 수집 + 자동 보정 (좌=0, 우=2 예시)
.\run_stereo.ps1

2단계. 보정 품질 요약 + 정렬맵 생성
python .\stereo_postcalib_tool.py `
  --npz .\stereo_pairs\calib_charuco_stereo.npz `
  --out-json .\stereo_pairs\calib_summary.json `
  --out-maps .\stereo_pairs\rectify_maps.npz `
  --alpha 0.0


3단계. 웹캠으로 “고정된 보드” 포즈 계산 → T_C←B
python .\pc_board_pose_rectified.py `
  --left-id 1 --right-id 2 `
  --width 1280 --height 720 --fps 24 `
  --calib .\stereo_pairs\calib_charuco_stereo.npz `
  --rectify-maps .\stereo_pairs\rectify_maps.npz `
  --alpha 0.0 `
  --shots 8 `
  --min-charuco 12 `
  --out .\stereo_pairs\T_C_from_B.npz `
  --log-json .\stereo_pairs\T_C_from_B_log.json `
  --backend dshow

--> 간단히 .\run_pc_board_pose_rectified.ps1


4단계. HoloLens (CalibrationScene) — 앵커 인식 + 보드 촬영
	1.	HoloLens에서 CalibrationScene 실행
	2.	앵커(QR) 을 2–3초간 바라봄 → [Anchor locked] 로그 확인
	3.	같은 보드(웹캠이 본 그 보드)를 HoloLens 카메라로 정면 보게 하고
음성 명령: “capture / 캡처 / 촬영”

→ HoloLens가 사진(photo.png) + 카메라→월드 포즈(pose.json)를 PC로 TCP 전송
PC 측 수신 서버 (미리 실행)
python .\hl_anchor_capture_server.py --port 19610 --once

5단계. T_H←C 계산 
.\run_hl_pose_and_compose_latest.ps1

--> 쓰는법:
어떻게 쓰면 되나
	•	처음 시도(베이스라인 진단)
	 1.	$FixBoardZ="none", $FixBoardRot="none", $RecomputeTHB=$true, $AutoSearch=$false
	 2.	스크립트 실행 → 콘솔에 det/||R^TR-I||/Z·+Z/XY_score 진단 확인
	•	최적 조합 찾기 → 재합성 (추천)
	 •	같은 세션에서 보드 사진/포즈가 바뀌지 않았다면 Step1은 다시 할 필요 없음.
	 •	$RecomputeTHB=$false, $AutoSearch=$true로 바꿔 실행 →
fix_h_board_frame.py가 [BEST] rz90+Ax 같은 조합을 찾고, 그 조합으로 compose_T_HC.py를 자동 재호출해 T_HC.npz/JSON 갱신.
	•	수동 재합성만 하고 싶을 때
	 •	$RecomputeTHB=$false로 두고, $FixBoardZ/$FixBoardRot에 원하는 조합을 직접 넣은 뒤 실행.
	 •	이때는 compose_T_HC.py만 다시 돌게 된다.

운용 Q&A

python fix_h_board_frame.py `
   --THB stereo_pairs\T_H_from_B.npz `
   --TCB stereo_pairs\T_C_from_B.npz `
   --out stereo_pairs\T_HC_fixed.npz

Q1. 처음에는 none으로 돌리고, 나중에 fix_h_board_frame.py로 최적 조합을 찾은 다음엔 어떻게?
	•	추천: fix_h_board_frame.py를 돌려 [BEST] rz90+Ax 같은 결과를 얻은 뒤,
이 스크립트의 상단 변수만 바꿔서 재실행해.

$FixBoardRot = "rz90"
$FixBoardZ   = "ax"

→ 두 스텝이 다시 돌지만, HL 사진/포즈가 안 바뀌었으면 Step1을 생략하고 아래 한 줄만으로도 충분:

python compose_T_HC.py --THB stereo_pairs\T_H_from_B.npz --TCB stereo_pairs\T_C_from_B.npz `
  --fix-board-rot none --fix-board-z ax `
  --out stereo_pairs\T_HC.npz --json-out stereo_pairs\T_HC.json



Q2. compose_T_HC.py만 따로 실행해도 되나?
	•	맞아. THB(=T_H_from_B.npz)가 최신이면 compose만 실행해도 된다. 위 한 줄 참고.





---- 이 아래는 예전방식 ----

5단계 후:
python fix_h_board_frame.py `
   --THB stereo_pairs\T_H_from_B.npz `
   --TCB stereo_pairs\T_C_from_B.npz `
   --out stereo_pairs\T_HC_fixed.npz
해서 베스트 찾아서 최종 파일 만들어줌



6단계. 최종 정합 패키지(JSON) 생성 
# Z 정렬 후 X도 XY평면에서 정렬(더 평행)
python .\make_calib_package.py `
  --thc .\stereo_pairs\T_HC.npz `
  --outdir .\calib `
  --align zx `
  --input-conv opencv `
  --anchor-id "Lab_QR1" `
  --note "align=zx; "

혹은 아래처럼 새로 생성
python compose_T_HC.py `
  --THB stereo_pairs\T_H_from_B.npz `
  --TCB stereo_pairs\T_C_from_B.npz `
  --fix-board-z ay --out stereo_pairs\T_HC.npz --json-out stereo_pairs\T_HC.json

  --fix-board-z ay `  --> 이부분 바꾸어야 함. 


7단계. 검증
python .\calib\test_apply_thc.py