#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2 as cv
import numpy as np

IDX = 0           # 너희 환경: left=0
W, H, FPS = 960, 544, 24

cap = cv.VideoCapture(IDX, cv.CAP_DSHOW)
if not cap.isOpened():
    raise SystemExit("Open failed (idx=0, DSHOW). 다른 앱이 카메라 점유 중인지 확인")

# *** 여기 핵심: BGR 변환을 확실히 켠다 ***
cap.set(cv.CAP_PROP_CONVERT_RGB, 1)

# 요청값
cap.set(cv.CAP_PROP_FRAME_WIDTH,  W)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, H)
cap.set(cv.CAP_PROP_FPS,          FPS)

# 워밍업
for _ in range(15):
    ok, frame = cap.read()
    if ok: break
    cv.waitKey(1)

if not ok or frame is None:
    cap.release()
    raise SystemExit("Read failed: 프라이버시 권한/점유 문제 가능")

# 실제 열린 값
aw = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
ah = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
af = cap.get(cv.CAP_PROP_FPS)
print(f"[INFO] opened at {aw}x{ah} @ {af:.0f}")

while True:
    ok, frame = cap.read()
    if not ok or frame is None:
        cv.waitKey(1)
        continue

    # 화면이 까만지 수치로도 확인
    mean_val = frame.mean()
    txt = f"{aw}x{ah}@{af:.0f}  mean={mean_val:.1f}"
    cv.putText(frame, txt, (10,30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv.imshow("raw_preview", frame)
    k = cv.waitKey(1) & 0xFF
    if k == 27: break  # ESC

cap.release()
cv.destroyAllWindows()
