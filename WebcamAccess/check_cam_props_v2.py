# check_cam_props_v2.py
import cv2, sys, platform, time

cam_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
W = int(sys.argv[2]) if len(sys.argv) > 2 else 1280
H = int(sys.argv[3]) if len(sys.argv) > 3 else 720
FPS = int(sys.argv[4]) if len(sys.argv) > 4 else 24

sysname = platform.system().lower()
if 'darwin' in sysname or 'mac' in sysname:
    backend = getattr(cv2, 'CAP_AVFOUNDATION', 1200)  # macOS 전용
elif 'windows' in sysname or 'win' in sysname:
    backend = getattr(cv2, 'CAP_DSHOW', 700)          # Windows 전용
else:
    backend = getattr(cv2, 'CAP_V4L2', 200)

cap = cv2.VideoCapture(cam_id, backend)
print("opened initially:", cap.isOpened())

# 원하는 설정 적용: FOURCC → 해상도 → FPS 순서 권장(드라이버마다 다름)
# 1) FOURCC: MJPG 시도
ok_fcc = cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# 2) 해상도
ok_w = cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
ok_h = cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

# 3) FPS
ok_fps = cap.set(cv2.CAP_PROP_FPS, FPS)

# 드라이버가 적용 시간을 필요로 할 수 있으므로 조금 대기 후 읽기
time.sleep(0.1)

# 실제 적용 값 읽기
rw = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
rh = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
rfps = cap.get(cv2.CAP_PROP_FPS)
fcc = int(cap.get(cv2.CAP_PROP_FOURCC))
fcc_str = "".join([chr((fcc >> 8*i) & 0xFF) for i in range(4)]) if fcc else "????"

print(f"set MJPG={ok_fcc}, W={ok_w}, H={ok_h}, FPS={ok_fps}")
print(f"actual W,H,FPS = {rw:.0f} x {rh:.0f} @ {rfps:.1f}")
print(f"actual FOURCC  = {fcc_str}")

# 간단 프레임 읽기 테스트
ok, frame = cap.read()
print("read frame:", ok, "|" , None if not ok else frame.shape)
cap.release()
