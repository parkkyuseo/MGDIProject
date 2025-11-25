# probe_formats_win.py
import cv2, time

BACKENDS = [
    ('MSMF', getattr(cv2, 'CAP_MSMF', 1400)),
    ('DSHOW', getattr(cv2, 'CAP_DSHOW', 700)),
]
RES = [(1280,720), (960,540), (640,480)]
FPS = [30, 24, 20]
MJPG = cv2.VideoWriter_fourcc(*'MJPG')

def fcc_str(val):
    return "".join([chr((int(val) >> 8*i) & 0xFF) for i in range(4)]) if val else "????"

for be_name, be in BACKENDS:
    print(f"\n=== BACKEND: {be_name} ===")
    for cam_id in [0,1,2,3]:
        cap = cv2.VideoCapture(cam_id, be)
        if not cap.isOpened():
            cap.release()
            continue
        print(f"\n[cam {cam_id}] opened")
        for (w,h) in RES:
            for fps in FPS:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                cap.set(cv2.CAP_PROP_FOURCC, MJPG)   # MJPG 강제 시도
                cap.set(cv2.CAP_PROP_FPS, fps)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                time.sleep(0.05)
                rw = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                rh = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                rf = cap.get(cv2.CAP_PROP_FPS)
                fcc = fcc_str(cap.get(cv2.CAP_PROP_FOURCC))
                ok, frm = cap.read()
                print(f"  try {w}x{h}@{fps:>2}  -> actual {int(rw)}x{int(rh)}@{rf:.1f}  FOURCC={fcc:4}  read={ok}")
        cap.release()
