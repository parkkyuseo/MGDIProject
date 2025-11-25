#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tbq_auto_solver_h3d.py
- hand3d_dyn의 정렬 프리뷰 + P1(좌 rectified intrinsics) 사용
- CharucoDetector → ArucoDetector → detectMarkers+interpolate 3단계 폴백
- R: CLAHE on/off, D: 딕셔너리 순환(5x5_50→4x4_50→6x6_50)
- ESC/q 또는 창 X로 안전 종료
- 카메라 인덱스: left=1, right=2 (고정, 자동 폴백 없음)

의존성: numpy, opencv-contrib-python, hand3d_dyn.py
"""

import argparse, json, os, time
from typing import Tuple, Optional
import numpy as np
import cv2 as cv
import hand3d_dyn as h3d

# ----- ArUco dict helpers -----
DICT_TABLE = {
    0: cv.aruco.DICT_4X4_50,    1: cv.aruco.DICT_4X4_100,  2: cv.aruco.DICT_4X4_250,  3: cv.aruco.DICT_4X4_1000,
    4: cv.aruco.DICT_5X5_50,    5: cv.aruco.DICT_5X5_100,  6: cv.aruco.DICT_5X5_250,  7: cv.aruco.DICT_5X5_1000,
    8: cv.aruco.DICT_6X6_50,    9: cv.aruco.DICT_6X6_100, 10: cv.aruco.DICT_6X6_250, 11: cv.aruco.DICT_6X6_1000,
    12: cv.aruco.DICT_7X7_50,  13: cv.aruco.DICT_7X7_100, 14: cv.aruco.DICT_7X7_250, 15: cv.aruco.DICT_7X7_1000,
}
DICT_CYCLE = [4, 0, 8]  # 5x5_50 → 4x4_50 → 6x6_50

def get_dict(did:int):
    return cv.aruco.getPredefinedDictionary(DICT_TABLE[did]) if hasattr(cv.aruco, "getPredefinedDictionary") else cv.aruco.Dictionary_get(DICT_TABLE[did])

def make_detector_params():
    p = cv.aruco.DetectorParameters_create() if hasattr(cv.aruco,"DetectorParameters_create") else cv.aruco.DetectorParameters()
    # 작은 마커 대응 튜닝
    p.minMarkerPerimeterRate = 0.02
    p.maxMarkerPerimeterRate = 4.0
    p.adaptiveThreshWinSizeMin = 3
    p.adaptiveThreshWinSizeMax = 23
    p.adaptiveThreshWinSizeStep = 4
    p.adaptiveThreshConstant   = 7
    if hasattr(cv.aruco, "CORNER_REFINE_SUBPIX"):
        p.cornerRefinementMethod = cv.aruco.CORNER_REFINE_SUBPIX
    p.minCornerDistanceRate = 0.02
    p.minMarkerDistanceRate = 0.02
    return p

def clahe_gray(bgr, clip=3.0, grid=8):
    g = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=clip, tileGridSize=(grid,grid))
    return clahe.apply(g)

# ----- math utils -----
def rvec_tvec_to_Rt(rvec, tvec):
    R,_ = cv.Rodrigues(rvec); t = tvec.reshape(3,)
    return R.astype(float), t.astype(float)

def Rt_to_T(R,t):
    T = np.eye(4, dtype=float); T[:3,:3]=R; T[:3,3]=t.reshape(3,); return T

def T_to_Rt(T):
    return T[:3,:3].copy(), T[:3,3].copy()

def save_npz(path, R, t):
    os.makedirs(os.path.dirname(path), exist_ok=True); np.savez(path, R=R, t=t)

def save_json(path, R, t):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path,"w",encoding="utf-8") as f:
        json.dump({"R":R.tolist(),"t":t.tolist()}, f, ensure_ascii=False)

# ----- QR pose -----
def solve_T_CQ_from_qr_points(pts_px: np.ndarray, Krect: np.ndarray, qr_size_m: float):
    # TL, TR, BR, BL
    half = qr_size_m/2.0
    obj = np.array([[-half,half,0],[half,half,0],[half,-half,0],[-half,-half,0]], dtype=np.float32)
    img = pts_px.astype(np.float32).reshape(-1,1,2)
    ok, rvec, tvec = cv.solvePnP(obj, img, Krect, None, flags=cv.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        ok2, rvec, tvec = cv.solvePnP(obj, img, Krect, None, flags=cv.SOLVEPNP_ITERATIVE)
        if not ok2: raise RuntimeError("solvePnP QR failed")
    return rvec_tvec_to_Rt(rvec, tvec)

# ----- Charuco pose (CharucoDetector → ArucoDetector → classic) -----
def detect_charuco_pose(img_rect_bgr, board, Krect, use_clahe=True):
    gray = clahe_gray(img_rect_bgr) if use_clahe else cv.cvtColor(img_rect_bgr, cv.COLOR_BGR2GRAY)
    ar = cv.aruco
    params = make_detector_params()
    ar_dict = board.getDictionary() if hasattr(board,"getDictionary") else board.dictionary

    # 1) CharucoDetector
    ChDet = getattr(ar, "CharucoDetector", None)
    if ChDet is not None:
        try:
            chd = ChDet(board, detectorParams=params)
            ret = chd.detectBoard(gray)
            if isinstance(ret, tuple):
                if len(ret)==2: ch_corners, ch_ids = ret
                elif len(ret)>=3: ch_corners, ch_ids = ret[0], ret[1]
                else: ch_corners, ch_ids = None, None
            else:
                ch_corners, ch_ids = ret, None
            if ch_corners is not None and ch_ids is not None and len(ch_ids) >= 6:
                ok_pose, rvec, tvec = ar.estimatePoseCharucoBoard(ch_corners, ch_ids, board, Krect, None)
                if ok_pose:
                    R,t = rvec_tvec_to_Rt(rvec,tvec)
                    return True, R, t, None, None
        except Exception:
            pass

    # 2) ArUcoDetector or detectMarkers
    ArDet = getattr(ar, "ArucoDetector", None)
    corners = ids = None
    try:
        if ArDet is not None:
            corners, ids, _rej = ArDet(ar_dict, params).detectMarkers(gray)
        else:
            corners, ids, _rej = ar.detectMarkers(gray, ar_dict, parameters=params)
    except Exception:
        corners = ids = None

    if ids is None or len(ids)==0:
        return False, None, None, None, None

    ret, ch_corners, ch_ids = ar.interpolateCornersCharuco(corners, ids, gray, board)
    if not ret or ch_ids is None or len(ch_ids) < 6:
        return False, None, None, corners, ids

    ok_pose, rvec, tvec = ar.estimatePoseCharucoBoard(ch_corners, ch_ids, board, Krect, None)
    if not ok_pose:
        return False, None, None, corners, ids

    R,t = rvec_tvec_to_Rt(rvec,tvec)
    return True, R, t, corners, ids

# ----- HUD & quit -----
def draw_status_lamp(img, center, ok, label):
    color = (50,220,50) if ok else (0,0,255)
    cv.circle(img, center, 8, color, -1, cv.LINE_AA)
    cv.putText(img, label, (center[0]+12, center[1]+5), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv.LINE_AA)

def want_quit(key: int) -> bool:
    return key in (27, ord('q'), ord('Q'))

def window_closed(win_name: str) -> bool:
    try:
        return cv.getWindowProperty(win_name, cv.WND_PROP_VISIBLE) < 1
    except Exception:
        return True

# =============================== main ===============================
def main():
    ap = argparse.ArgumentParser()
    # h3d init (left=1, right=2 고정으로 실행할 것. 자동폴백 없음)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--left-id", type=int, required=True)
    ap.add_argument("--right-id", type=int, required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--min-conf", type=float, default=0.35)
    ap.add_argument("--detect-scale", type=float, default=1.0)
    ap.add_argument("--detect-every", type=int, default=1)
    ap.add_argument("--max-pair-dt-ms", type=int, default=30)
    # Charuco/QR
    ap.add_argument("--dict", type=int, default=4)  # 5x5_50
    ap.add_argument("--squares-x", type=int, required=True)
    ap.add_argument("--squares-y", type=int, required=True)
    ap.add_argument("--square-len", type=float, required=True)
    ap.add_argument("--marker-len", type=float, required=True)
    ap.add_argument("--qr-size", type=float, required=True)
    # Output
    ap.add_argument("--out-npz", default="transforms/T_BQ.npz")
    ap.add_argument("--out-json", default="transforms/T_BQ.json")
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()

    # 1) h3d initialize (left=1, right=2 사용)
    inst = h3d.initialize(
        calib_path=args.calib,
        cam_left=args.left_id, cam_right=args.right_id,
        width=args.width, height=args.height, fps=args.fps,
        alpha=args.alpha, min_conf=args.min_conf,
        max_pair_dt_ms=args.max_pair_dt_ms,
        detect_scale=args.detect_scale, detect_every=args.detect_every
    )

    # 2) dict & board
    dict_idx = DICT_CYCLE.index(args.dict) if args.dict in DICT_CYCLE else 0
    ar_dict = get_dict(DICT_CYCLE[dict_idx])
    if hasattr(cv.aruco,"CharucoBoard_create"):
        board = cv.aruco.CharucoBoard_create(args.squares_x, args.squares_y, args.square_len, args.marker_len, ar_dict)
    else:
        board = cv.aruco.CharucoBoard((args.squares_x,args.squares_y), args.square_len, args.marker_len, ar_dict)

    # 3) P1 + preview frames
    print("[INFO] waiting for rectified intrinsics (P1) and preview frames...")
    P1 = None; t0 = time.time()
    while True:
        frames = inst.get_preview_frames()
        P1 = getattr(inst, "P1", None)
        if frames is not None and P1 is not None: break
        if time.time()-t0 > 5.0:
            print("[WARN] still waiting..."); t0=time.time()
        time.sleep(0.01)
    Krect = np.array(P1[:3,:3], dtype=float)

    # 4) 창 준비
    win = "tbq_auto_solver_h3d"
    if args.preview:
        cv.namedWindow(win, cv.WINDOW_NORMAL)
        cv.resizeWindow(win, 960, 540)

    qrdec = cv.QRCodeDetector()
    use_clahe = True
    print("[INFO] Show board. SPACE=capture, ESC/q=exit, D=cycle dict, R=toggle CLAHE")

    R_CB = t_CB = R_CQ = t_CQ = None

    # 5) 루프
    while True:
        frames = inst.get_preview_frames()
        if frames is None:
            cv.waitKey(1); continue
        imgL, _imgR = frames
        img = imgL.copy()

        # Charuco
        okB, R_CB_tmp, t_CB_tmp, corners, ids = detect_charuco_pose(img, board, Krect, use_clahe=use_clahe)
        if corners is not None and ids is not None:
            cv.aruco.drawDetectedMarkers(img, corners, ids)
        if okB:
            cv.drawFrameAxes(img, Krect, None, cv.Rodrigues(R_CB_tmp)[0], t_CB_tmp.reshape(3,1), args.square_len*2)

        # QR
        okQ = False
        try:
            data_str, points, _ = qrdec.detectAndDecode(img)
        except Exception:
            points = None
        if points is not None and points.size == 8:
            pts = points.reshape(-1,2).astype(np.float32)
            for i in range(4):
                p1 = tuple(map(int, pts[i])); p2 = tuple(map(int, pts[(i+1)%4]))
                cv.line(img, p1, p2, (0,255,0), 2)
            try:
                R_CQ_tmp, t_CQ_tmp = solve_T_CQ_from_qr_points(pts, Krect, args.qr_size)
                okQ = True
                cv.drawFrameAxes(img, Krect, None, cv.Rodrigues(R_CQ_tmp)[0], t_CQ_tmp.reshape(3,1), args.qr_size*1.2)
            except Exception:
                okQ = False

        # HUD
        draw_status_lamp(img, (12,20), okB, 'B')
        draw_status_lamp(img, (12,44), okQ, 'Q')

        # 미리보기/종료
        if args.preview:
            cv.imshow(win, img)
            k = cv.waitKey(1) & 0xFF
            if want_quit(k) or window_closed(win):
                try: cv.destroyWindow(win)
                except Exception: pass
                print("[INFO] Exit without saving."); return
        else:
            k = 255; cv.waitKey(1)

        # 키 처리
        if k in (ord('d'), ord('D')):
            dict_idx = (dict_idx + 1) % len(DICT_CYCLE)
            did = DICT_CYCLE[dict_idx]
            ar_dict = get_dict(did)
            if hasattr(cv.aruco,"CharucoBoard_create"):
                board = cv.aruco.CharucoBoard_create(args.squares_x, args.squares_y, args.square_len, args.marker_len, ar_dict)
            else:
                board = cv.aruco.CharucoBoard((args.squares_x,args.squares_y), args.square_len, args.marker_len, ar_dict)
            print(f"[INFO] Switched ArUco dict id={did} (5x5_50=4, 4x4_50=0, 6x6_50=8)")

        if k in (ord('r'), ord('R')):
            use_clahe = not use_clahe
            print(f"[INFO] CLAHE {'ON' if use_clahe else 'OFF'}")

        if k == 32 and okB and okQ:  # SPACE
            R_CB, t_CB = R_CB_tmp, t_CB_tmp
            R_CQ, t_CQ = R_CQ_tmp, t_CQ_tmp
            print("[INFO] Captured both poses.")
            break

    # 6) 저장
    if args.preview:
        try: cv.destroyWindow(win)
        except Exception: pass

    if R_CB is None or R_CQ is None:
        print("[ERR] Missing pose(s)."); return

    T_CB = Rt_to_T(R_CB, t_CB)
    T_CQ = Rt_to_T(R_CQ, t_CQ)
    T_BQ = np.linalg.inv(T_CB) @ T_CQ
    R_BQ, t_BQ = T_to_Rt(T_BQ)

    save_npz(args.out_npz, R_BQ, t_BQ)
    save_json(args.out_json, R_BQ, t_BQ)
    print(f"[OK] Saved T_B<-Q:\n  {args.out_npz}\n  {args.out_json}")

if __name__ == "__main__":
    main()
