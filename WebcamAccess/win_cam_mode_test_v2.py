import cv2, time

# 바꿔볼 후보들
BACKENDS = {
    "dshow": cv2.CAP_DSHOW,
    "msmf":  cv2.CAP_MSMF,
}
FOURCCS = ["MJPG", "YUY2"]   # MJPG 우선, 안 되면 YUY2
WIDTH, HEIGHT, FPS = 1920, 1080, 30

# 실제 쓰려는 두 카메라 인덱스 지정 (win_cam_sanity.py 결과 참고)
IDX_L, IDX_R = 0, 1

def open_cam(idx, api, fourcc):
    cap = cv2.VideoCapture(idx, api)
    if not cap.isOpened():
        return None, f"open fail (idx={idx}, api={api})"

    # 색공간 변환 강제 (중요!)
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)

    # 해상도/코덱/FPS 요청
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    time.sleep(0.2)

    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        return None, "read fail"

    # 실제 적용된 값 읽기
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fcc_str = "".join([chr((fcc >> 8*i) & 0xFF) for i in range(4)])

    return (cap, frame, w, h, fps, fcc_str), None

def test_pair(api_name, fourcc):
    api = BACKENDS[api_name]
    print(f"\n=== backend={api_name} fourcc={fourcc} ===")
    L = open_cam(IDX_L, api, fourcc)
    R = open_cam(IDX_R, api, fourcc)

    if L[1] or R[1]:
        print("  -> FAIL:",
              L[1] if L[1] else "ok",
              "|",
              R[1] if R[1] else "ok")
        # 클린업
        if not L[1]: L[0][0].release()
        if not R[1]: R[0][0].release()
        return False

    capL, frmL, wL, hL, fpsL, fccL = L[0]
    capR, frmR, wR, hR, fpsR, fccR = R[0]
    print(f"  LEFT : {wL}x{hL} @{fpsL:.1f} FOURCC={fccL}")
    print(f"  RIGHT: {wR}x{hR} @{fpsR:.1f} FOURCC={fccR}")

    # 간단히 화면 띄워서 '초록 화면' 여부 확인
    cv2.imshow(f"LEFT {api_name} {fourcc}", frmL)
    cv2.imshow(f"RIGHT {api_name} {fourcc}", frmR)
    cv2.waitKey(500)   # 0.5s 미리보기
    cv2.destroyAllWindows()

    capL.release(); capR.release()
    return True

if __name__ == "__main__":
    any_ok = False
    for api_name in ("msmf", "dshow"):       # MSMF 먼저, 그다음 DSHOW
        for fourcc in FOURCCS:               # MJPG 먼저, 안되면 YUY2
            ok = test_pair(api_name, fourcc)
            any_ok = any_ok or ok
    print("\nSUMMARY:", "OK at least one combo" if any_ok else "No working combo")
