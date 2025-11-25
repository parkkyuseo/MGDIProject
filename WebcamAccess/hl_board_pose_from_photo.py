# hl_board_pose_from_photo.py
# Input: --photo <png>  --pose <json from HoloLens>  --board-meta-from <stereo calib npz or json>
# Output: T_H_from_B.npz with {"R": R_HB, "t": t_HB}
import argparse, json, numpy as np, cv2
from pathlib import Path

# --- coordinate convention helpers ---
S_CV2UNITY = np.diag([1.0, -1.0, 1.0])  # y-down(OpenCV) -> y-up(Unity)
S_UNITY2CV = S_CV2UNITY                 # same numeric matrix

def load_board_meta(from_npz: str):
    d = np.load(from_npz, allow_pickle=True)
    meta_raw = d["meta"]
    if isinstance(meta_raw, np.ndarray): meta_raw = meta_raw.item()
    meta = json.loads(meta_raw) if isinstance(meta_raw, str) else meta_raw
    bm = meta["board"]
    return {
        "squaresX": int(bm["squaresX"]),
        "squaresY": int(bm["squaresY"]),
        "squareLength": float(bm["squareLength"]),
        "markerLength": float(bm["markerLength"]),
        "dict_name": bm["dict_name"]
    }

def get_dict(name: str):
    name = name.lower()
    mp = {
        "4x4_50": cv2.aruco.DICT_4X4_50,
        "4x4_100": cv2.aruco.DICT_4X4_100,
        "5x5_50": cv2.aruco.DICT_5X5_50,
        "5x5_100": cv2.aruco.DICT_5X5_100,
        "6x6_50": cv2.aruco.DICT_6X6_50,
        "6x6_100": cv2.aruco.DICT_6X6_100,
        "apriltag_36h11": cv2.aruco.DICT_APRILTAG_36h11,
    }
    return cv2.aruco.getPredefinedDictionary(mp[name])

def build_charuco(bm, ar_dict):
    return cv2.aruco.CharucoBoard(
        (bm["squaresX"], bm["squaresY"]),
        bm["squareLength"], bm["markerLength"], ar_dict
    )

def _as_mat3(x, name):
    arr = np.array(x, dtype=np.float64)
    if arr.shape == (3, 3): return arr
    if arr.ndim == 1 and arr.size == 9: return arr.reshape(3, 3)
    raise KeyError(f"{name} must be 3x3 or length-9, got shape {arr.shape}")

def json_load_pose(pose_json: str):
    with open(pose_json, "r", encoding="utf-8") as f:
        blob = json.load(f)
    data = blob.get("data", blob)

    packs = []
    if any(k in data for k in ("R_HC", "t_HC", "K")): packs.append(data)
    for k in ("pose", "camera", "cam"):
        if isinstance(data.get(k), dict): packs.append(data[k])
    if any(k in blob for k in ("R_HC", "t_HC", "K")): packs.append(blob)

    last_err = None
    for pack in packs:
        try:
            R_HC = _as_mat3(pack["R_HC"], "R_HC")
            t_HC = np.array(pack["t_HC"], dtype=np.float64).reshape(3)
            K    = _as_mat3(pack["K"], "K")
            dist = np.array(pack.get("dist", []), dtype=np.float64).reshape(-1)
            return R_HC, t_HC, K, dist, data  # return 'data' for input_convention lookup
        except Exception as e:
            last_err = e
            continue
    raise KeyError(f"pose.json missing usable R_HC/t_HC/K. last error: {last_err}")

def rodrigues_to_R(rvec):
    R, _ = cv2.Rodrigues(rvec)
    return R

def compose_THB(R_HC_cv, t_HC, R_CB, t_CB):
    # T_H<-B = T_H<-C(cv) * T_C(cv)<-B
    R_HB = R_HC_cv @ R_CB
    t_HB = t_HC + R_HC_cv @ t_CB
    return R_HB, t_HB

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photo", required=True)
    ap.add_argument("--pose", required=True)
    ap.add_argument("--board-meta-from", required=True, help="stereo_pairs/calib_charuco_stereo.npz")
    ap.add_argument("--out", default="T_H_from_B.npz")
    ap.add_argument("--min-charuco", type=int, default=12)
    args = ap.parse_args()

    bm = load_board_meta(args.board_meta_from)
    ar_dict = get_dict(bm["dict_name"])
    board = build_charuco(bm, ar_dict)

    R_HC, t_HC, K, dist, pose_data = json_load_pose(args.pose)

    # Determine input convention for HL pose (default: unity)
    pose_conv = str(pose_data.get("input_convention", "unity")).lower()
    if pose_conv not in ("unity", "opencv"): pose_conv = "unity"

    # Convert HL pose to OpenCV camera convention if needed
    if pose_conv == "unity":
        R_HC_cv = R_HC @ S_UNITY2CV
        used_conv = "unity->opencv"
    else:
        R_HC_cv = R_HC.copy()
        used_conv = "opencv"
    print(f"[OK] pose loaded; ||t_HC||={np.linalg.norm(t_HC):.3f}  input_convention={pose_conv}  used={used_conv}")

    # Read HL photo
    img = cv2.imread(args.photo, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read photo: {args.photo}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h_img, w_img = gray.shape[:2]

    # Quick sanity for K vs image center
    cx, cy = float(K[0, 2]), float(K[1, 2])
    if not (abs(cx - 0.5 * w_img) < 0.15 * w_img and abs(cy - 0.5 * h_img) < 0.15 * h_img):
        print(f"[WARN] principal point far from image center: (cx,cy)=({cx:.1f},{cy:.1f}) vs ({w_img},{h_img})")

    # Detect markers
    params = cv2.aruco.DetectorParameters()
    det = cv2.aruco.ArucoDetector(ar_dict, params)
    corners, ids, _ = det.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        raise RuntimeError("No aruco markers detected in the photo")

    # ---- Robust Charuco interpolation (handles both API signatures) ----
    res = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
    ch_corners = None
    ch_ids = None
    n_char = 0

    # unify across OpenCV versions
    if isinstance(res, tuple):
        if len(res) == 3:
            a, b, c = res
            # If first is scalar (retval), then (retval, corners, ids)
            if np.isscalar(a) or (hasattr(a, "shape") and a.shape == ()):
                retval = float(a)
                ch_corners = b
                ch_ids = c
                n_char = int(retval)
            else:
                # Assume (corners, ids, _) signature
                ch_corners = a
                ch_ids = b
                n_char = 0 if ch_ids is None else int(len(ch_ids))
        elif len(res) == 2:
            # Some builds might return (corners, ids)
            ch_corners, ch_ids = res
            n_char = 0 if ch_ids is None else int(len(ch_ids))
    else:
        raise RuntimeError("Unexpected return type from interpolateCornersCharuco()")

    # basic validity
    if ch_ids is None or ch_corners is None or n_char < args.min_charuco:
        # try fallback: compute length from arrays (if retval unreliable)
        try:
            n_char = min(len(ch_corners), len(ch_ids))
        except Exception:
            n_char = 0
        if n_char < args.min_charuco:
            raise RuntimeError(f"Not enough charuco corners: {n_char} < {args.min_charuco}")

    # enforce equal length (safety for OpenCV assertion)
    if len(ch_corners) != len(ch_ids):
        m = min(len(ch_corners), len(ch_ids))
        ch_corners = ch_corners[:m]
        ch_ids     = ch_ids[:m]

    # Pose on HL photo (Cam(cv) <- B)
    ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
        ch_corners, ch_ids, board, K, dist, rvec=None, tvec=None
    )
    if not ok:
        raise RuntimeError("estimatePoseCharucoBoard failed on HL photo")
    R_CB = rodrigues_to_R(rvec)
    t_CB = tvec.reshape(3)

    # Compose T_H<-B with HL pose already in OpenCV camera convention
    R_HB, t_HB = compose_THB(R_HC_cv, t_HC, R_CB, t_CB)
    np.savez(args.out, R=R_HB.astype(np.float64), t=t_HB.astype(np.float64))
    print(f"[OK] Saved T_H_from_B.npz -> {Path(args.out).resolve()}")
    print("R_HB:\n", R_HB)
    print("t_HB (m):", t_HB)

if __name__ == "__main__":
    main()
