import cv2, time, numpy as np

# ===== 설정 =====
IDX_L, IDX_R = 0, 1             # 찾은 카메라 인덱스로 바꾸세요
BACKENDS = [("msmf", cv2.CAP_MSMF), ("dshow", cv2.CAP_DSHOW)]
FOURCCS  = ["MJPG", "YUY2"]
RESLIST  = [(1280,720), (1920,1080)]  # 필요시 (640,480)도 추가

# ===== 유틸 =====
def fourcc_to_str(fccint):
    return "".join([chr((fccint >> 8*i) & 0xFF) for i in range(4)])

def open_cam(idx, api, fourcc, W, H, FPS=30):
    cap = cv2.VideoCapture(idx, api)
    if not cap.isOpened(): return None, "open fail"

    # Windows에서 녹색 화면 방지: BGR 변환 강제
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)

    # 모드 설정
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    # 버퍼를 얕게
    try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except: pass

    # 워밍업
    t0 = time.time()
    for _ in range(5):
        cap.read()
    # 1 프레임 읽기
    ret, frm = cap.read()
    if not ret or frm is None:
        cap.release(); return None, "read fail"

    # 실제 적용 값
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fcc = fourcc_to_str(int(cap.get(cv2.CAP_PROP_FOURCC)))
    info = dict(w=w,h=h,fps=fps,fcc=fcc)
    return (cap, frm, info), None

def show_pair(frmL, frmR, title, infoL, infoR):
    visL = frmL.copy(); visR = frmR.copy()
    for img, tag, info in [(visL,"LEFT",infoL), (visR,"RIGHT",infoR)]:
        cv2.putText(img, f"{tag} {info['w']}x{info['h']} @{info['fps']:.1f} FOURCC={info['fcc']}",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)
    panel = np.hstack([visL, visR])
    cv2.imshow(title, panel)

# ===== 메인 =====
def main():
    bi, fi, ri = 0, 0, 1  # 초기 선택: msmf + MJPG + 1920x1080
    capL = capR = None

    def try_open():
        nonlocal capL, capR
        if capL: capL.release()
        if capR: capR.release()
        bname, api = BACKENDS[bi]
        fourcc = FOURCCS[fi]
        W,H = RESLIST[ri]

        left = open_cam(IDX_L, api, fourcc, W, H)
        right= open_cam(IDX_R, api, fourcc, W, H)
        if left[1] or right[1]:
            if left[0]: left[0][0].release()
            if right[0]: right[0][0].release()
            return None, None, f"open fail: {bname} + {fourcc} {W}x{H}"
        capL, frmL, infoL = left[0]
        capR, frmR, infoR = right[0]
        return (frmL, infoL), (frmR, infoR), f"{bname} + {fourcc} {W}x{H}"

    (frmL, infoL), (frmR, infoR), title = try_open()
    if frmL is None:
        print("초기 조합 열기 실패. IDX_L/IDX_R나 장치 연결을 확인하세요.")
        return

    print("키 안내:  A/D=코덱  W/S=백엔드  Q/E=해상도  SPACE=리프레시  P=스냅샷 저장  ESC=종료")
    while True:
        # 최신 프레임 읽기
        okL, frmL = capL.read()
        okR, frmR = capR.read()
        if not (okL and okR):
            # 장치 hiccup → 재오픈
            (frmL, infoL), (frmR, infoR), title = try_open()
            if frmL is None: break

        show_pair(frmL, frmR, title, infoL, infoR)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key in (ord('a'), ord('A')):  # 코덱 이전
            fi = (fi - 1) % len(FOURCCS)
            (frmL, infoL), (frmR, infoR), title = try_open()
        elif key in (ord('d'), ord('D')):  # 코덱 다음
            fi = (fi + 1) % len(FOURCCS)
            (frmL, infoL), (frmR, infoR), title = try_open()
        elif key in (ord('w'), ord('W')):  # 백엔드 이전
            bi = (bi - 1) % len(BACKENDS)
            (frmL, infoL), (frmR, infoR), title = try_open()
        elif key in (ord('s'), ord('S')):  # 백엔드 다음
            bi = (bi + 1) % len(BACKENDS)
            (frmL, infoL), (frmR, infoR), title = try_open()
        elif key in (ord('q'), ord('Q')):  # 해상도 이전
            ri = (ri - 1) % len(RESLIST)
            (frmL, infoL), (frmR, infoR), title = try_open()
        elif key in (ord('e'), ord('E')):  # 해상도 다음
            ri = (ri + 1) % len(RESLIST)
            (frmL, infoL), (frmR, infoR), title = try_open()
        elif key in (ord(' '),):          # 리프레시
            (frmL, infoL), (frmR, infoR), title = try_open()
        elif key in (ord('p'), ord('P')): # 스냅샷 저장
            bname, _ = BACKENDS[bi]
            fourcc = FOURCCS[fi]
            W,H = RESLIST[ri]
            cv2.imwrite(f"LEFT_{bname}_{fourcc}_{W}x{H}.png", frmL)
            cv2.imwrite(f"RIGHT_{bname}_{fourcc}_{W}x{H}.png", frmR)
            print("snapshot saved.")

    if capL: capL.release()
    if capR: capR.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
