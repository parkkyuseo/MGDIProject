# hl_anchor_capture_server.py (safe-exit version)
import argparse, socket, struct, json, os, time, sys
from datetime import datetime
from pathlib import Path

import numpy as np
import cv2

def recv_exact(conn, n, timeout=5.0):
    conn.settimeout(timeout)
    b = bytearray()
    while len(b) < n:
        chunk = conn.recv(n - len(b))
        if not chunk:
            raise ConnectionError("socket closed")
        b.extend(chunk)
    return bytes(b)

def save_npz_rt(path, R, t):
    np.savez(path, R=R.astype(np.float64), t=t.astype(np.float64))

def load_npz_rt(path):
    d = np.load(path)
    return d["R"].reshape(3,3), d["t"].reshape(3)

def rodrigues_to_R(rvec):
    R, _ = cv2.Rodrigues(rvec); return R

def get_aruco_dict(name):
    mp = {
        "4x4_50":  cv2.aruco.DICT_4X4_50,
        "4x4_100": cv2.aruco.DICT_4X4_100,
        "5x5_50":  cv2.aruco.DICT_5X5_50,
        "5x5_100": cv2.aruco.DICT_5X5_100,
        "6x6_50":  cv2.aruco.DICT_6X6_50,
        "6x6_100": cv2.aruco.DICT_6X6_100,
        "apriltag_36h11": cv2.aruco.DICT_APRILTAG_36h11,
    }
    return cv2.aruco.getPredefinedDictionary(mp[name])

def load_board_meta(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    meta_raw = d["meta"]; 
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

def estimate_THB_from_photo(photo_path, pose_json_path, board_calib_npz, min_char=12):
    img = cv2.imread(str(photo_path), cv2.IMREAD_COLOR)
    if img is None: raise RuntimeError(f"read fail: {photo_path}")
    with open(pose_json_path, "r", encoding="utf-8") as f:
        env = json.load(f)
    data = env.get("data", env)
    R_HC = np.array(data["R_HC"], dtype=np.float64)
    t_HC = np.array(data["t_HC"], dtype=np.float64).reshape(3)
    K = np.array(data["K"], dtype=np.float64).reshape(3,3)
    dist = np.array(data.get("dist", []), dtype=np.float64).reshape(-1)

    bm = load_board_meta(board_calib_npz)
    ar_dict = get_aruco_dict(bm["dict_name"])
    board = cv2.aruco.CharucoBoard((bm["squaresX"], bm["squaresY"]),
                                   bm["squareLength"], bm["markerLength"], ar_dict)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    det = cv2.aruco.ArucoDetector(ar_dict, cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(gray)
    if ids is None or len(ids)==0:
        raise RuntimeError("No aruco markers in photo")

    retval, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
    nchar = int(retval) if isinstance(retval, (np.floating, float)) else (len(ch_ids) if ch_ids is not None else 0)
    if ch_ids is None or ch_corners is None or nchar < min_char:
        raise RuntimeError(f"not enough charuco corners: {nchar} < {min_char}")

    ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
        ch_corners, ch_ids, board, K, dist if dist.size>0 else None, rvec=None, tvec=None
    )
    if not ok: raise RuntimeError("estimatePoseCharucoBoard failed")
    R_CB = rodrigues_to_R(rvec)
    t_CB = tvec.reshape(3)

    R_HB = R_HC @ R_CB
    t_HB = t_HC + R_HC @ t_CB
    return R_HB, t_HB

def invert_RT(R,t):
    Rinv = R.T; tinv = - Rinv @ t; return Rinv, tinv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="anchor_caps")
    ap.add_argument("--port", type=int, default=19610)
    ap.add_argument("--board-calib", type=str, default="")
    ap.add_argument("--TCB", type=str, default="")
    ap.add_argument("--once", action="store_true", help="handle one connection then exit")
    args = ap.parse_args()

    out_base = Path(args.out_dir); out_base.mkdir(parents=True, exist_ok=True)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", args.port)); srv.listen(1)
    srv.settimeout(0.3)  # accept 타임아웃
    print(f"[SRV] listening TCP on 0.0.0.0:{args.port}  (press 'q' or ESC to quit)")

    try:
        import msvcrt
        def key_quit():
            if msvcrt.kbhit():
                ch = msvcrt.getch()
                if not ch: return False
                c = ch.decode(errors="ignore").lower()
                return c == 'q' or ch == b'\x1b'
            return False
    except ImportError:
        def key_quit(): return False  # 비윈도우 대체

    try:
        while True:
            # 키로 종료
            if key_quit():
                print("[SRV] bye."); break

            try:
                conn, addr = srv.accept()
            except socket.timeout:
                continue

            with conn:
                print(f"[SRV] connection from {addr}")
                # 길이+데이터 받기
                jlen = struct.unpack("<I", recv_exact(conn, 4))[0]
                jbuf = recv_exact(conn, jlen)
                plen = struct.unpack("<I", recv_exact(conn, 4))[0]
                pbuf = recv_exact(conn, plen)

                tsdir = datetime.utcnow().strftime("cap_%Y%m%d_%H%M%S")
                outdir = out_base / tsdir
                outdir.mkdir(parents=True, exist_ok=True)

                pose_json_path = outdir / "pose.json"
                photo_path = outdir / "photo.png"
                with open(pose_json_path, "wb") as f: f.write(jbuf)
                with open(photo_path, "wb") as f: f.write(pbuf)
                print(f"[SRV] saved {pose_json_path.name} ({len(jbuf)}B), {photo_path.name} ({len(pbuf)}B)")

                # 선택: T_H<-B
                thb_path = None
                if args.board_calib:
                    try:
                        R_HB, t_HB = estimate_THB_from_photo(photo_path, pose_json_path, args.board_calib)
                        thb_path = outdir / "T_H_from_B.npz"
                        save_npz_rt(thb_path, R_HB, t_HB)
                        print(f"[SRV] saved {thb_path.name}")
                    except Exception as e:
                        print(f"[SRV] T_H<-B failed: {e}")

                # 선택: T_HC
                if thb_path and args.TCB:
                    try:
                        R_HB, t_HB = load_npz_rt(thb_path)
                        R_CB, t_CB = load_npz_rt(args.TCB)
                        R_BC, t_BC = invert_RT(R_CB, t_CB)
                        R_HC = R_HB @ R_BC
                        t_HC = t_HB + R_HB @ t_BC
                        thc_path = outdir / "T_HC.npz"
                        save_npz_rt(thc_path, R_HC, t_HC)
                        print(f"[SRV] saved {thc_path.name}")
                    except Exception as e:
                        print(f"[SRV] T_HC compose failed: {e}")

                print("[SRV] done for this connection.")
                if args.once:
                    print("[SRV] --once: exiting.")
                    break

    except KeyboardInterrupt:
        print("[SRV] bye (KeyboardInterrupt).")
    finally:
        srv.close()

if __name__ == "__main__":
    main()
