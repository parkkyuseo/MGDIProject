#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
import numpy as np
import cv2

aru = cv2.aruco

DICT_NAME_TO_ENUM = {
    "4x4_50": aru.DICT_4X4_50,   "4x4_100": aru.DICT_4X4_100,
    "4x4_250": aru.DICT_4X4_250, "4x4_1000": aru.DICT_4X4_1000,
    "5x5_50": aru.DICT_5X5_50,   "5x5_100": aru.DICT_5X5_100,
    "5x5_250": aru.DICT_5X5_250, "5x5_1000": aru.DICT_5X5_1000,
    "6x6_50": aru.DICT_6X6_50,   "6x6_100": aru.DICT_6X6_100,
    "6x6_250": aru.DICT_6X6_250, "6x6_1000": aru.DICT_6X6_1000,
    "7x7_50": aru.DICT_7X7_50,   "7x7_100": aru.DICT_7X7_100,
    "7x7_250": aru.DICT_7X7_250, "7x7_1000": aru.DICT_7X7_1000,
}

def get_dict(name):
    enum = DICT_NAME_TO_ENUM[name.lower()]
    if hasattr(aru, "getPredefinedDictionary"):
        return aru.getPredefinedDictionary(enum)
    return aru.Dictionary_get(enum)

def make_board(sx, sy, square, marker, dict_name):
    d = get_dict(dict_name)
    board = None
    if hasattr(aru, "CharucoBoard_create"):
        board = aru.CharucoBoard_create(sx, sy, square, marker, d)
    else:
        C = getattr(aru, "CharucoBoard", None)
        if C and hasattr(C, "create"):
            board = C.create(sx, sy, square, marker, d)
        elif C:
            board = C((sx, sy), square, marker, d)
    if board is None:
        raise SystemExit("OpenCV build lacks CharucoBoard API (install opencv-contrib-python).")
    return board, d

def detect_charuco(gray, board, d):
    params = getattr(aru, "DetectorParameters", lambda: None)()
    # 튜닝이 필요하면 여기에서 params.* 값을 조정
    ArDet = getattr(aru, "ArucoDetector", None)
    ChDet = getattr(aru, "CharucoDetector", None)

    if ChDet is not None:
        try:
            ret, cc, ci = ChDet(board).detectBoard(gray)
            if ret and cc is not None and ci is not None and len(cc) >= 6:
                return True, cc, ci
        except Exception:
            pass

    if ArDet is not None:
        corners, ids, rej = ArDet(d, params).detectMarkers(gray)
    else:
        corners, ids, rej = aru.detectMarkers(gray, d, parameters=params)

    if ids is None or len(ids) == 0:
        return False, None, None

    if hasattr(aru, "refineDetectedMarkers"):
        try:
            aru.refineDetectedMarkers(gray, board, corners, ids, rej or [])
        except Exception:
            pass

    ret, cc, ci = aru.interpolateCornersCharuco(corners, ids, gray, board)
    ok = (ret and cc is not None and len(cc) >= 6)
    return ok, cc if ok else None, ci if ok else None

def main():
    ap = argparse.ArgumentParser(description="Offline ChArUco stereo calibration (Windows/macOS)")
    ap.add_argument("--dir", required=True, help="folder that contains left_*.png and right_*.png")
    ap.add_argument("--left-prefix", default="left_", help="left image prefix (default: left_)")
    ap.add_argument("--right-prefix", default="right_", help="right image prefix (default: right_)")
    ap.add_argument("--ext", default=".png", choices=[".png",".jpg",".jpeg"], help="image ext")
    ap.add_argument("--dict", default="4x4_50", choices=[k for k in DICT_NAME_TO_ENUM.keys()],
                    help="charuco dictionary")
    ap.add_argument("--squares-x", type=int, default=7)
    ap.add_argument("--squares-y", type=int, default=5)
    ap.add_argument("--square", type=float, default=0.017, help="square length (meter)")
    ap.add_argument("--marker", type=float, default=0.013, help="marker length (meter)")
    ap.add_argument("--min-common", type=int, default=10, help="min common IDs per pair (>=6)")
    ap.add_argument("--out", default="calib_charuco_stereo_offline.npz")
    ap.add_argument("--use-intrinsic-guess", action="store_true",
                    help="use CALIB_USE_INTRINSIC_GUESS instead of CALIB_FIX_INTRINSIC")
    args = ap.parse_args()

    root = Path(args.dir)
    # 확장자 보정: ".png" 또는 "png" 모두 허용
    ext = args.ext if args.ext.startswith(".") else f".{args.ext}"

    # 파일 목록 (← 하이픈이 아니라 언더스코어 속성 사용!)
    L = sorted(str(p) for p in root.glob(f"{args.left_prefix}*{ext}"))
    R = sorted(str(p) for p in root.glob(f"{args.right_prefix}*{ext}"))

    if len(L) == 0 or len(L) != len(R):
        raise SystemExit(f"no or mismatched pairs in {root} (found L={len(L)}, R={len(R)})")


    board, d = make_board(args.squares_x, args.squares_y, args.square, args.marker, args.dict)

    objpoints, img1, img2 = [], [], []
    imsize = None
    used = 0
    dropped = 0

    for i, (lp, rp) in enumerate(zip(L, R)):
        g1 = cv2.imread(lp, cv2.IMREAD_GRAYSCALE)
        g2 = cv2.imread(rp, cv2.IMREAD_GRAYSCALE)
        if g1 is None or g2 is None:
            print(f"[{i:03d}] read fail: {lp} / {rp}")
            continue
        if imsize is None:
            imsize = (g1.shape[1], g1.shape[0])

        ok1, c1, id1 = detect_charuco(g1, board, d)
        ok2, c2, id2 = detect_charuco(g2, board, d)
        n1 = 0 if not ok1 else len(id1)
        n2 = 0 if not ok2 else len(id2)

        common = 0
        if ok1 and ok2:
            s1 = set(int(x) for x in id1.reshape(-1))
            s2 = set(int(x) for x in id2.reshape(-1))
            keep_ids = sorted(list(s1 & s2))
            common = len(keep_ids)
        else:
            keep_ids = []

        print(f"[{i:03d}] L_ids={n1:2d} R_ids={n2:2d} common={common:2d}  ({Path(lp).name})")
        if common < args.min_common:
            dropped += 1
            continue

        # 공통 ID 순서대로 좌/우 2D 포인트 정렬
        idx1 = [int(np.where(id1.reshape(-1) == k)[0][0]) for k in keep_ids]
        idx2 = [int(np.where(id2.reshape(-1) == k)[0][0]) for k in keep_ids]
        pts1 = c1.reshape(-1, 2)[idx1].astype(np.float32)
        pts2 = c2.reshape(-1, 2)[idx2].astype(np.float32)

        # ChArUco 코너의 3D 좌표 (보드 평면)
        try:
            corners3d = np.array(board.chessboardCorners, np.float32)
        except AttributeError:
            corners3d = np.array(board.getChessboardCorners(), np.float32)
        obj = corners3d[np.array(keep_ids, np.int32)]

        objpoints.append(obj)
        img1.append(pts1)
        img2.append(pts2)
        used += 1

    if used < 3:
        print(f"\nNot enough valid stereo pairs after matching: used={used}, dropped={dropped}")
        print("→ Increase coverage(≥15%), reduce reflections, and keep pairs with common IDs ≥ 8–12.")
        sys.exit(0)

    # 단안 보정
    flags = 0
    err1, K1, D1, *_ = cv2.calibrateCamera(objpoints, img1, imsize, None, None, flags=flags)
    err2, K2, D2, *_ = cv2.calibrateCamera(objpoints, img2, imsize, None, None, flags=flags)

    # 스테레오 보정
    stereo_flags = cv2.CALIB_FIX_INTRINSIC
    if args.use_intrinsic_guess:
        stereo_flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_ZERO_TANGENT_DIST)

    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-9)
    rms, K1o, D1o, K2o, D2o, R, T, E, F = cv2.stereoCalibrate(
        objpoints, img1, img2, K1, D1, K2, D2, imsize,
        flags=stereo_flags, criteria=crit
    )

    # R 직교화
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R[:, -1] *= -1

    out = root / args.out
    meta = dict(
        imsize=imsize,
        mono_reproj_err_L=float(err1),
        mono_reproj_err_R=float(err2),
        stereo_reproj_err=float(rms),
        used_pairs=int(used),
        dropped_pairs=int(dropped),
        board=dict(squaresX=args.squares_x, squaresY=args.squares_y,
                   squareLength=args.square, markerLength=args.marker,
                   dict=args.dict)
    )
    np.savez_compressed(
        out,
        K1=K1o, D1=D1o, K2=K2o, D2=D2o, R=R, t=T, E=E, F=F,
        meta=json.dumps(meta)
    )
    print("\nSaved:", out)
    print("Stereo RMS:", rms, "  baseline |t| (m):", float(np.linalg.norm(T)))

if __name__ == "__main__":
    main()
