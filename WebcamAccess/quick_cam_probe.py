# quick_cam_probe.py
import cv2, time

def try_open(idx, backend, w, h, fps=15, set_mjpg=True):
    cap = cv2.VideoCapture(idx, backend)
    if not cap.isOpened():
        print(f"[FAIL] open idx={idx} backend={backend}")
        return
    if set_mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)
    time.sleep(0.2)
    ok, frame = cap.read()
    cap.release()
    print(f"[{ 'OK ' if ok and frame is not None else 'BAD'}] idx={idx} backend={backend} {w}x{h}@{fps}")
    return ok

# 후보 인덱스 0..5, 백엔드 대조
cands = [0,1,2,3,4,5]
backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF]
for b in backends:
    for i in cands:
        # 저해상도 → 표준 HD 순서로 테스트
        try_open(i, b, 640, 480, 15)
        try_open(i, b, 1280, 720, 24)
