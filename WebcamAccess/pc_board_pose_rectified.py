# pc_board_pose_rectified.py
# Python 3.11 / OpenCV 4.x
# - 정렬된 좌영상(C_rect)에서 Charuco 보드 포즈(T_C←B) 추정
# - Space -> 3초 카운트다운 -> 자동 캡처 시작 -> --shots 회 성공 시 자동 종료
# - 결과: npz {"R": (3,3), "t": (3,)}  단위: 미터

import argparse
import json
import time
import math
from pathlib import Path
from typing import Tuple, List

import numpy as np
import cv2

# ---------------------------
# Utilities
# ---------------------------

def load_calib_with_meta(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    K1 = data["K1"]; D1 = data["D1"]
    K2 = data["K2"]; D2 = data["D2"]
    R  = data["R"];  t  = data["t"].reshape(3)
    meta_raw = data.get("meta", None)
    if meta_raw is None:
        raise RuntimeError("calib npz has no 'meta'")
    if isinstance(meta_raw, np.ndarray):
        meta_raw = meta_raw.item()
    meta = json.loads(meta_raw) if isinstance(meta_raw, str) else meta_raw
    imsize = meta.get("imsize", None)
    if imsize is None:
        raise RuntimeError("meta['imsize'] is missing (expected [width, height])")
    board_meta = meta.get("board", None)
    if board_meta is None:
        raise RuntimeError("meta['board'] is missing")
    return K1, D1, K2, D2, R, t, meta, board_meta, tuple(imsize)

def get_aruco_dictionary(dict_name: str):
    # meta 예: '4x4_50', '5x5_100' 등
    name = dict_name.lower()
    mapping = {
        "4x4_50":  cv2.aruco.DICT_4X4_50,
        "4x4_100": cv2.aruco.DICT_4X4_100,
        "5x5_50":  cv2.aruco.DICT_5X5_50,
        "5x5_100": cv2.aruco.DICT_5X5_100,
        "6x6_50":  cv2.aruco.DICT_6X6_50,
        "6x6_100": cv2.aruco.DICT_6X6_100,
        "7x7_50":  cv2.aruco.DICT_7X7_50,
        "7x7_100": cv2.aruco.DICT_7X7_100,
        "apriltag_36h11": cv2.aruco.DICT_APRILTAG_36h11,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dict_name in meta: {dict_name}")
    return cv2.aruco.getPredefinedDictionary(mapping[name])

def build_charuco(board_meta: dict, aruco_dict):
    squaresX = int(board_meta["squaresX"])
    squaresY = int(board_meta["squaresY"])
    squareLength = float(board_meta["squareLength"])
    markerLength = float(board_meta["markerLength"])
    # OpenCV는 unit 자유. 여기선 미터 단위로 생성(후 solvePnP도 미터 기준이 됨)
    board = cv2.aruco.CharucoBoard((squaresX, squaresY), squareLength, markerLength, aruco_dict)
    return board

def stereo_rectify_from_calib(K1, D1, K2, D2, R, t, imsize, alpha=0.25):
    w, h = imsize
    flags = 0
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1, D1, K2, D2, (w, h), R, t,
        flags=flags, alpha=alpha, newImageSize=(w, h)
    )
    return R1, R2, P1, P2, Q, roi1, roi2

def load_rectify_maps(npz_path: str):
    d = np.load(npz_path)
    map1x = d["map1x"]; map1y = d["map1y"]
    # 우맵도 들어있겠지만, 여기선 좌영상만 필요
    return map1x, map1y

def rodrigues_to_R(rvec):
    R, _ = cv2.Rodrigues(rvec)
    return R

def average_rotations_so3(Rs: List[np.ndarray]) -> np.ndarray:
    # simple chordal L2 mean on SO(3) via SVD (polar decomposition)
    M = np.zeros((3,3), dtype=np.float64)
    for R in Rs:
        M += R
    U, _, Vt = np.linalg.svd(M)
    Rm = U @ Vt
    if np.linalg.det(Rm) < 0:
        # ensure det=+1
        U[:, -1] *= -1
        Rm = U @ Vt
    return Rm

def draw_hud(img, text_lines, color=(0,255,0)):
    y = 24
    for line in text_lines:
        cv2.putText(img, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        y += 22

# ---------------------------
# Pose estimation on rectified left
# ---------------------------

def estimate_pose_charuco_on_rectified(
    img_rect_bgr: np.ndarray,
    board,
    K_rect: np.ndarray,
    min_charuco: int = 12
) -> Tuple[bool, np.ndarray, np.ndarray, dict]:
    gray = cv2.cvtColor(img_rect_bgr, cv2.COLOR_BGR2GRAY)
    parameters = cv2.aruco.DetectorParameters()
    aruco_dict = board.getDictionary()

    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return False, None, None, {"n_markers": 0, "n_charuco": 0}

    # Refine (optional): improves corner results on Charuco
    # corners, ids, _ = cv2.aruco.refineDetectedMarkers(gray, board, corners, ids, rejected)

    # Charuco interpolation
    retval, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(
        markerCorners=corners,
        markerIds=ids,
        image=gray,
        board=board
    )
    n_char = int(retval) if isinstance(retval, (np.floating, float)) else (len(ch_ids) if ch_ids is not None else 0)
    if ch_ids is None or ch_corners is None or n_char < min_charuco:
        return False, None, None, {"n_markers": len(ids), "n_charuco": n_char}

    # estimate pose (Charuco-specific)
    ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
        charucoCorners=ch_corners,
        charucoIds=ch_ids,
        board=board,
        cameraMatrix=K_rect,
        distCoeffs=None,
        rvec=None,
        tvec=None
    )

    if not ok:
        return False, None, None, {"n_markers": len(ids), "n_charuco": n_char}

    R = rodrigues_to_R(rvec)
    t = tvec.reshape(3)
    return True, R, t, {"n_markers": len(ids), "n_charuco": n_char}

# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Estimate T_C<-B (board->C_rect) on rectified left image.")
    ap.add_argument("--left-id", type=int, required=True)
    ap.add_argument("--right-id", type=int, required=True)  # 열지 않지만 해상도 설정 체크용으로 받음
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--calib", type=str, required=True, help="stereo_pairs/calib_charuco_stereo.npz")
    ap.add_argument("--rectify-maps", type=str, required=True, help="stereo_pairs/rectify_maps.npz (needs map1x,map1y)")
    ap.add_argument("--alpha", type=float, default=0.25, help="stereoRectify alpha to derive K_rect (0..1)")
    ap.add_argument("--shots", type=int, default=8, help="number of successful captures after countdown")
    ap.add_argument("--min-charuco", type=int, default=12)
    ap.add_argument("--out", type=str, default="T_C_from_B.npz")
    ap.add_argument("--log-json", type=str, default="T_C_from_B_log.json")
    ap.add_argument("--backend", type=str, default="dshow", choices=["dshow","msmf","auto"])
    args = ap.parse_args()

    # Load calib & meta
    K1, D1, K2, D2, R_st, t_st, meta, board_meta, imsize = load_calib_with_meta(args.calib)
    aruco_dict = get_aruco_dictionary(board_meta["dict_name"])
    board = build_charuco(board_meta, aruco_dict)

    # Derive rectified K for left from stereoRectify (distortion ~ 0)
    R1, R2, P1, P2, Q, roi1, roi2 = stereo_rectify_from_calib(K1, D1, K2, D2, R_st, t_st, imsize, alpha=args.alpha)
    K_rect = P1[:, :3]

    # Load rectify maps (left)
    map1x, map1y = load_rectify_maps(args.rectify_maps)

    # Open left camera
    if args.backend == "dshow":
        cap = cv2.VideoCapture(args.left_id, cv2.CAP_DSHOW)
    elif args.backend == "msmf":
        cap = cv2.VideoCapture(args.left_id, cv2.CAP_MSMF)
    else:
        cap = cv2.VideoCapture(args.left_id)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS,          args.fps)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    if not cap.isOpened():
        raise RuntimeError("Left camera open failed.")

    win = "C_rect Charuco Pose (Space=Start, ESC=Quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    armed = False
    start_after = None
    shots_goal = max(1, args.shots)
    collected_R = []
    collected_t = []
    log_items = []

    while True:
        ok, frame = cap.read()
        if not ok:
            cv2.waitKey(5)
            continue

        # Remap to rectified left (C_rect)
        img_rect = cv2.remap(frame, map1x, map1y, interpolation=cv2.INTER_LINEAR)
        vis = img_rect.copy()

        # HUD
        hud = []
        if not armed:
            hud.append("Press SPACE to arm, then 3-sec countdown.")
        else:
            if start_after is not None:
                remain = max(0.0, start_after - time.time())
                if remain > 0:
                    hud.append(f"Armed. Capturing starts in {remain:.1f}s...")
                else:
                    hud.append(f"Capturing... [{len(collected_R)}/{shots_goal}]")
            else:
                hud.append(f"Capturing... [{len(collected_R)}/{shots_goal}]")

        draw_hud(vis, hud)

        # If armed and countdown elapsed -> try capture
        if armed and start_after is not None and time.time() >= start_after:
            ok_pose, R_cb, t_cb, stats = estimate_pose_charuco_on_rectified(
                vis, board, K_rect, min_charuco=args.min_charuco
            )
            # Overlay detection stats
            cv2.putText(vis, f"markers={stats.get('n_markers',0)} charuco={stats.get('n_charuco',0)}",
                        (12, vis.shape[0]-16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2, cv2.LINE_AA)

            if ok_pose:
                collected_R.append(R_cb)
                collected_t.append(t_cb)
                log_items.append({"ok": True, **stats})
                # Small pause to avoid duplicate near-identical frames
                time.sleep(0.05)
            else:
                log_items.append({"ok": False, **stats})

            # Done?
            if len(collected_R) >= shots_goal:
                break

        # Show
        cv2.imshow(win, vis)
        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord('q')):  # ESC/q
            break
        if key == 32 and not armed:  # SPACE
            armed = True
            start_after = time.time() + 3.0  # countdown 3s

    cap.release()
    cv2.destroyAllWindows()

    if len(collected_R) == 0:
        print("No successful captures; nothing saved.")
        return

    # Average pose
    R_avg = average_rotations_so3(collected_R)
    t_avg = np.mean(np.stack(collected_t, axis=0), axis=0)

    # --- SO(3) 투영(수치오차 제거) + 진단 ---
    # (average_rotations_so3 내부에서도 det=+1 보정하지만, 한 번 더 안전망)
    U, _, Vt = np.linalg.svd(R_avg); R_avg = U @ Vt
    if np.linalg.det(R_avg) < 0:
        U[:, -1] *= -1
        R_avg = U @ Vt

    detR = float(np.linalg.det(R_avg))
    ortho_err = float(np.linalg.norm(R_avg.T @ R_avg - np.eye(3), ord='fro'))
    zdot = float(R_avg[:,2] @ np.array([0.0,0.0,1.0]))

    print(f"[OK] (diagnostics) det(R)={detR: .6f}  ||R^TR - I||={ortho_err: .3e}  R[:,2]·+Z={zdot: .6f}")
    print(f"[OK] (sanity) t_avg.z={t_avg[2]: .6f}  (should be > 0)")


    # Save outputs
    out_npz = Path(args.out)
    np.savez(out_npz, R=R_avg.astype(np.float64), t=t_avg.astype(np.float64))
    print(f"[OK] Saved T_C_from_B.npz -> {out_npz.resolve()}")
    print("R (3x3):\n", R_avg)
    print("t (m): ", t_avg)

    # Log JSON
    log = {
        "timestamp": time.time(),
        "shots_goal": shots_goal,
        "shots_success": len(collected_R),
        "min_charuco": args.min_charuco,
        "imsize": list(imsize),
        "alpha_used_for_Krect": args.alpha,
        "stats": log_items,
        # --- 추가 메타 (정체성/진단) ---
        "C_ref": "left_rectified",         # 기준 카메라 (정렬된 좌영상)
        "convention_C": "opencv",          # C 좌표계는 OpenCV 카메라 좌표 (x-right, y-down, z-forward)
        "sanity": {
            "detR": detR,
            "ortho_err": ortho_err,
            "Rz_dot_Z": zdot,
            "tz": float(t_avg[2])
        }

    }
    with open(args.log_json, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    print(f"[OK] Saved log -> {Path(args.log_json).resolve()}")

if __name__ == "__main__":
    main()
