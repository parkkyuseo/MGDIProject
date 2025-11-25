# hl_sender.py
# PC -> HoloLens : 3D hand (C_rect) -> (H) via T_HC, UDP JSON
import argparse, json, socket, time, numpy as np
import hand3d_dyn as h3d

import sys, time
try:
    import msvcrt  # Windows만
    _HAS_MSVCRT = True
except ImportError:
    _HAS_MSVCRT = False

def load_transform_npz(path):
    d = np.load(path)
    R = d["R"].reshape(3,3).astype(np.float64)
    t = d["t"].reshape(3).astype(np.float64)
    return R, t

def apply_T(R, t, X):  # X: (3,) or (N,3)
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1: return (R @ X) + t
    return (X @ R.T) + t  # faster for batch

def _should_quit():
    # Ctrl+C는 이미 try/except KeyboardInterrupt로 처리
    # q 키로도 종료 (Windows 콘솔)
    if _HAS_MSVCRT and msvcrt.kbhit():
        ch = msvcrt.getch()
        try:
            c = ch.decode('utf-8').lower()
        except:
            c = ''
        if c == 'q':
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left-id", type=int, required=True)
    ap.add_argument("--right-id", type=int, required=True)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--calib", type=str, required=True, help="stereo_pairs/calib_charuco_stereo.npz")
    ap.add_argument("--transform", type=str, required=True, help="stereo_pairs/T_HC.npz")
    ap.add_argument("--udp", type=str, required=True, help="ip:port (HoloLens)")
    ap.add_argument("--hand", type=str, default="right")
    ap.add_argument("--min-conf", type=float, default=0.35)
    ap.add_argument("--detect-scale", type=float, default=1.0)
    ap.add_argument("--detect-every", type=int, default=1)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    ip, port = args.udp.split(":")
    port = int(port)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # init tracker
    inst = h3d.initialize(
        calib_path=args.calib,
        cam_left=args.left_id, cam_right=args.right_id,
        width=args.width, height=args.height, fps=args.fps,
        detect_scale=args.detect_scale, detect_every=args.detect_every,
        alpha=0.30, min_conf=args.min_conf
    )

    R_HC, t_HC = load_transform_npz(args.transform)
    if not args.quiet:
        print("[HL-SENDER] loaded T_HC from:", args.transform)

    last_ok = None
    t0 = time.time()
    try:
        while True:
            res = h3d.get_minimal_hand3d()
            if not res or "wrist" not in res or "palm" not in res or "index_tip" not in res:
                continue
            conf = res.get("conf", {})
            ptsC = np.array([res["wrist"], res["palm"], res["index_tip"]], dtype=np.float64)  # (3,3)

            # confidence gate: if any too low, hold last
            if min([conf.get("wrist",1), conf.get("palm",1), conf.get("index_tip",1)]) < args.min_conf:
                if last_ok is None: continue
                ptsH = last_ok
            else:
                ptsH = apply_T(R_HC, t_HC, ptsC)  # (3,3)
                last_ok = ptsH

            payload = {
                "hand": args.hand,
                "ts": int(time.time()*1000),
                "points": [
                    {"name":"wrist","x":float(ptsH[0,0]),"y":float(ptsH[0,1]),"z":float(ptsH[0,2])},
                    {"name":"palm","x":float(ptsH[1,0]),"y":float(ptsH[1,1]),"z":float(ptsH[1,2])},
                    {"name":"index_tip","x":float(ptsH[2,0]),"y":float(ptsH[2,1]),"z":float(ptsH[2,2])},
                ]
            }
            sock.sendto(json.dumps(payload).encode("utf-8"), (ip, port))

            if _should_quit():
                print("[HL-SENDER] quit by key 'q'")
                break

            # 너무 타이트하면 Ctrl+C가 안 먹을 수 있음 → 아주 짧게 양보
            time.sleep(0.001)


            if not args.quiet and (time.time()-t0) > 1.0:
                print("[HL-SENDER] sent at ~{} Hz".format(args.fps))
                t0 = time.time()

    except KeyboardInterrupt:
        pass
    finally:
        sock.close()

if __name__ == "__main__":
    main()
