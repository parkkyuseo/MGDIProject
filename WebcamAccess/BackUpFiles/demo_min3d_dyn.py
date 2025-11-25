#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI demo for minimal stereo hand 3D extraction (dynamic rectify version).
- Uses engine's rectified previews and cached 2D landmarks (no extra Mediapipe here)
- Shows landmark overlays + epipolar guides (lightweight)
- Logs Z samples (throttled)
- Optional UDP JSON streaming
- Added: periodic hand_status heartbeat, optional invalid-frame emission
"""
from __future__ import annotations
import argparse
import time
import socket
import json
import numpy as np
import cv2
import sys, os
import hashlib

import hand3d_dyn as h3d  # 동적 정렬맵 버전

# ------------------ UDP helper ------------------
class UDPSender:
    def __init__(self, spec: str | None):
        self.sock = None
        self.addr = None
        if spec:
            host, port = spec.split(":")
            self.addr = (host, int(port))
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, payload: dict):
        if self.sock is None:
            return
        data = json.dumps(payload).encode('utf-8')
        self.sock.sendto(data, self.addr)

# ------------------ Drawing helpers ------------------
POINT_IDS = [0, 5, 9, 13, 17, 8]
COLORS = {
    0: (0, 255, 0),     # wrist
    5: (255, 200, 0),   # index_mcp
    9: (255, 200, 0),   # middle_mcp
    13: (255, 200, 0),  # ring_mcp
    17: (255, 200, 0),  # pinky_mcp
    8: (0, 200, 255),   # index_tip
}

def draw_points(img: np.ndarray, det: dict, draw_epi: bool = True):
    ys_for_epi = []
    for k, (u, v, c) in det.items():
        cv2.circle(img, (int(u), int(v)), 3, COLORS.get(k, (255, 0, 0)), -1)
        if k in (0, 8):
            ys_for_epi.append(int(v))
    if draw_epi and ys_for_epi:
        h, w = img.shape[:2]
        for y in ys_for_epi:
            cv2.line(img, (0, y), (w - 1, y), (80, 80, 80), 1, cv2.LINE_AA)

def annotate_status(img, label, npts, fps=None, zval=None):
    y = 24
    cv2.putText(img, f"{label} | pts={npts}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)
    y += 24
    if fps is not None:
        cv2.putText(img, f"FPS~{fps:.1f}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2, cv2.LINE_AA)
    if zval is not None:
        y += 24
        cv2.putText(img, f"Z(palm)={zval:.3f} m", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)

# ---- T_H<-C loader (inline) ----
def _quat_to_R(x, y, z, w):
    q = np.array([x, y, z, w], dtype=np.float64)
    q /= np.linalg.norm(q)
    x, y, z, w = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w),   2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w),   1-2*(x*x+y*y)]
    ], dtype=np.float64)

def load_thc_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rq = data["R_quat"]; tt = data["t"]
    R = _quat_to_R(float(rq["x"]), float(rq["y"]), float(rq["z"]), float(rq["w"]))
    t = np.array([float(tt["x"]), float(tt["y"]), float(tt["z"])], dtype=np.float64)
    return R, t, data

# ---- Input convention helpers ----
S_CV2UNITY = np.diag([1.0, -1.0, 1.0])  # y-down(OpenCV) -> y-up(Unity/HoloLens)

def apply_input_conv(Xc: np.ndarray, input_conv: str) -> np.ndarray:
    Xc = np.asarray(Xc, dtype=np.float64).reshape(3,)
    if input_conv == "opencv":
        return S_CV2UNITY @ Xc
    return Xc

def thc_hash_from_json(meta: dict) -> str:
    rq = meta.get("R_quat", {})
    tt = meta.get("t", {})
    s  = f'{rq.get("x",0):.9f},{rq.get("y",0):.9f},{rq.get("z",0):.9f},{rq.get("w",0):.9f}|{tt.get("x",0):.9f},{tt.get("y",0):.9f},{tt.get("z",0):.9f}'
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


# ------------------ Main ------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--left-id', type=int, default=0)
    ap.add_argument('--right-id', type=int, default=1)
    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--height', type=int, default=720)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--calib', type=str, required=True)

    # engine tuning
    ap.add_argument('--alpha', type=float, default=0.25)
    ap.add_argument('--min-conf', type=float, default=0.5)
    ap.add_argument('--jump-thresh-m', type=float, default=0.10)
    ap.add_argument('--max-interp-ms', type=int, default=100)
    ap.add_argument('--max-pair-dt-ms', type=int, default=15)
    ap.add_argument('--detect-scale', type=float, default=0.5,
                    help='Run Mediapipe on downscaled frames (e.g., 0.5 → 1/2).')
    ap.add_argument('--detect-every', type=int, default=2,
                    help='Run Mediapipe every Nth frame (e.g., 2 → every other frame).')

    # demo options
    ap.add_argument('--show', action='store_true')
    ap.add_argument('--no-epi', action='store_true', help='Hide epipolar guide lines')
    ap.add_argument('--vis-scale', type=float, default=0.75, help='preview downscale for display only')
    ap.add_argument('--z-log-interval', type=float, default=0.5, help='seconds between Z logs')

    ap.add_argument('--quiet', action='store_true', help='suppress console logs except essential state changes')

    # transform/stream options
    ap.add_argument('--thc', type=str, default=None, help='path to ./calib/T_HC.json')
    ap.add_argument('--udp', type=str, default=None, help='ip:port (e.g., 192.168.50.212:33333)')
    ap.add_argument('--hz',  type=float, default=0.0, help='max send rate (0 = no limit)')
    ap.add_argument('--dry-run', action='store_true', help='do not send UDP; print transformed XYZ')
    ap.add_argument('--input-conv', type=str, default='auto', choices=['auto','opencv','unity'],
                    help='how to interpret incoming C coords: auto=use JSON if present else opencv')

    # NEW: diagnostics/robustness
    ap.add_argument('--status-every', type=float, default=0.2,
                    help='seconds between hand_status heartbeats (min ~0.05s)')
    ap.add_argument('--emit-invalid', action='store_true',
                    help='also send hand_points with valid=false when output is invalid')

    # [PATCH][STATS] 진단 리포트 옵션
    ap.add_argument('--diag-report-sec', type=float, default=0.0,
                    help='seconds between diagnostic summaries to stdout (0=off)')
    ap.add_argument('--diag-final', action='store_true',
                    help='print final diagnostic summary at exit')


    args = ap.parse_args()

    # Initialize engine ...
    inst = h3d.initialize(
        calib_path=args.calib,
        cam_left=args.left_id,
        cam_right=args.right_id,
        width=args.width,
        height=args.height,
        fps=args.fps,
        alpha=args.alpha,
        min_conf=args.min_conf,
        jump_thresh_m=args.jump_thresh_m,
        max_interp_ms=args.max_interp_ms,
        max_pair_dt_ms=args.max_pair_dt_ms,
        detect_scale=args.detect_scale,
        detect_every=args.detect_every,
    )

    # === 초기 1회: 실제 프레임 크기로 맵 맞추고 프리뷰 스태시 심기 ===
    time.sleep(0.1)
    try:
        camL_open = getattr(inst.camL.cap, "isOpened", lambda: False)()
        camR_open = getattr(inst.camR.cap, "isOpened", lambda: False)()
        okL, rawL = (inst.camL.cap.read() if camL_open else (False, None))
        okR, rawR = (inst.camR.cap.read() if camR_open else (False, None))

        if okL and okR and rawL is not None and rawR is not None:
            h, w = rawL.shape[:2]
            inst.force_rebuild_maps(w, h)
            print(f"[boot] forced maps to {w}x{h}")

            if rawL.ndim == 2: rawL = cv2.cvtColor(rawL, cv2.COLOR_GRAY2BGR)
            if rawR.ndim == 2: rawR = cv2.cvtColor(rawR, cv2.COLOR_GRAY2BGR)
            rl0, rr0 = inst._rectify_pair(rawL, rawR)
            with inst._prev_lock:
                inst._prevL, inst._prevR = rl0, rr0
                if not hasattr(inst, "_prev_count"): inst._prev_count = 0
                inst._prev_count += 1
                inst._prev_last_ts = int(time.time()*1000)
            print(f"[boot] seeded preview stash with {w}x{h}")
    except Exception as e:
        print(f"[boot] init-seed skipped: {e}")

    R_HC = None; t_HC = None
    input_conv_use = 'opencv'
    json_info = {}

    if args.thc:
        try:
            R_HC, t_HC, meta = load_thc_json(args.thc)
            json_info = meta
            json_conv = str(meta.get('input_convention', 'opencv')).lower()
            if args.input_conv != 'auto':
                input_conv_use = args.input_conv
                src = "CLI"
            else:
                input_conv_use = json_conv
                src = "JSON"
            print(f"[thc] loaded T_H<-C | ts={meta.get('timestamp','')} | note={meta.get('note','')}")
            print(f"[conv] input_convention={input_conv_use} (source={src})")
        except Exception as e:
            print(f"[ERROR] T_HC load failed: {e}")
            sys.exit(1)
    else:
        print("[WARN] --thc not provided; sending C_rect coords as-is (no H transform)")
        if args.input_conv != 'auto':
            input_conv_use = args.input_conv
            print(f"[conv] input_convention={input_conv_use} (source=CLI; no THC)")
        else:
            input_conv_use = 'opencv'
            print(f"[conv] input_convention=opencv (default; no THC)")

    udp = UDPSender(args.udp)
    last_log_t = 0.0
    t_prev = time.time()
    fps_ema = None

    thc_info_sent = False
    frame_id = 0
    last_status_t = 0.0  # NEW: heartbeat timer

    win_title = f"rectified L|R  {args.width}x{args.height}  detect_scale={args.detect_scale}  every={args.detect_every}"

    prev_stereo_ok = None
    prev_nL = prev_nR = None
    prev_out_valid = None

    if args.show:
        cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)

    # boot wait ...
    boot_deadline = time.time() + 8.0
    boot_ok = False
    while time.time() < boot_deadline:
        if args.show:
            cv2.waitKey(1)
        frames = inst.get_preview_frames()
        if frames is not None:
            boot_ok = True
            break
        time.sleep(0.03)

    if not boot_ok:
        print("[ERROR] No preview frames within 8s. Check camera IDs/ports/backends.")
        try:
            dbg = inst.get_debug_state()
            print(f"[diag] preview stash count={dbg['prev_count']}  last_ms={dbg['prev_last_ms']}  "
                  f"mapL={dbg['mapL_size']} mapR={dbg['mapR_size']}")
        except Exception as e:
            print(f"[diag] preview debug failed: {e}")
        try: inst.stop()
        except: pass
        try:
            if args.show:
                cv2.waitKey(1); cv2.destroyAllWindows()
        except: pass
        sys.exit(1)

    send_interval = 0.0 if (args.hz is None or args.hz <= 0.0) else (1.0 / float(args.hz))
    last_send_time = 0.0

    try:
        # [PATCH][STATS]
        next_diag_t = (time.time() + float(args.diag_report_sec)) if float(getattr(args, 'diag_report_sec', 0.0)) > 0.0 else None

        while True:
            now = time.time()
            dt = now - t_prev
            t_prev = now
            if dt > 0:
                fps_inst = 1.0 / dt
                fps_ema = fps_inst if fps_ema is None else (0.9 * fps_ema + 0.1 * fps_inst)

            frames = inst.get_preview_frames()
            if frames is None:
                time.sleep(0.005)
                if args.show:
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord('q'), ord('Q')):
                        break
                # even if no frames, still send heartbeat
                if (now - last_status_t) >= max(0.05, float(args.status_every)):
                    last_status_t = now
                    status = {
                        "type": "hand_status",
                        "ts": int(time.time()*1000),
                        "nL": 0, "nR": 0,
                        "out_valid": False,
                        "conf": {},
                        "fps": float(fps_ema) if fps_ema is not None else None
                    }
                    if args.dry_run or not args.udp: print(json.dumps(status))
                    else: udp.send(status)
                continue

            rl, rr = frames
            rl = rl.copy(); rr = rr.copy()

            last2d = inst.get_last_2d()
            nL = nR = 0
            dL = dR = {}
            if last2d is not None:
                dL, dR = last2d
                nL, nR = len(dL), len(dR)
                if args.show:
                    draw_points(rl, dL, draw_epi=not args.no_epi)
                    draw_points(rr, dR, draw_epi=not args.no_epi)

            out = h3d.get_minimal_hand3d()
            stereo_ok = (nL >= 6 and nR >= 6 and out is not None)

            # NEW: send thc_info once if available, regardless of validity
            if (args.udp and (R_HC is not None) and (t_HC is not None) and (not thc_info_sent)):
                def f6(x): return float(f"{x:.6f}")
                meta_payload = {
                    "type": "thc_info",
                    "ts_send": int(time.time() * 1000),
                    "thc_hash": thc_hash_from_json(json_info) if json_info else "",
                    "input_convention": str(input_conv_use),
                    "anchor_id": json_info.get("anchor_id", None) if json_info else None
                }
                rq = json_info.get("R_quat", None)
                if rq:
                    meta_payload["R_quat"] = {"x": float(rq["x"]), "y": float(rq["y"]),
                                              "z": float(rq["z"]), "w": float(rq["w"])}
                meta_payload["t"] = {"x": f6(t_HC[0]), "y": f6(t_HC[1]), "z": f6(t_HC[2])}
                if args.dry_run: print(json.dumps(meta_payload))
                else: udp.send(meta_payload)
                thc_info_sent = True
                print(f"[thc] using {args.thc} hash={meta_payload['thc_hash']}")

            # ---- send hand_points (valid or invalid) ----
            sent_any_points = False
            if out is not None:
                conf = out.get("conf", {})
                good = (conf.get("wrist",0.0) >= args.min_conf and
                        conf.get("palm",0.0)  >= args.min_conf and
                        conf.get("index_tip",0.0) >= args.min_conf)
                if good:
                    def f6(x): return float(f"{x:.6f}")
                    ptsC = {
                        "wrist":     tuple(out["wrist"]),
                        "palm":      tuple(out["palm"]),
                        "index_tip": tuple(out["index_tip"]),
                    }

                    cam_u = {}
                    ptsH = {}

                    if R_HC is not None:
                        for k, v in ptsC.items():
                            Xc = np.array(v, dtype=np.float64)
                            Xc_u = apply_input_conv(Xc, input_conv_use)
                            cam_u[k] = Xc_u.copy()
                            Xh = R_HC @ Xc_u + t_HC
                            ptsH[k] = (f6(Xh[0]), f6(Xh[1]), f6(Xh[2]))
                    else:
                        for k, v in ptsC.items():
                            Xc = np.array(v, dtype=np.float64)
                            Xc_u = apply_input_conv(Xc, input_conv_use)
                            cam_u[k] = Xc_u.copy()
                            ptsH[k] = (f6(Xc_u[0]), f6(Xc_u[1]), f6(Xc_u[2]))

                    frame_id += 1
                    payload = {
                        "type": "hand_points",
                        "frame_id": frame_id,
                        "hand": "right",
                        "ts": int(time.time() * 1000),
                        "points": [
                            {"name":"wrist",     "x": ptsH["wrist"][0], "y": ptsH["wrist"][1], "z": ptsH["wrist"][2]},
                            {"name":"palm",      "x": ptsH["palm"][0],  "y": ptsH["palm"][1],  "z": ptsH["palm"][2]},
                            {"name":"index_tip", "x": ptsH["index_tip"][0], "y": ptsH["index_tip"][1], "z": ptsH["index_tip"][2]},
                        ],
                        "palm_cz":  float(f"{cam_u['palm'][2]:.6f}"),
                        "wrist_cz": float(f"{cam_u['wrist'][2]:.6f}"),
                        "valid": True
                    }

                    if args.dry_run or not args.udp:
                        print(json.dumps(payload))
                    else:
                        if (send_interval == 0.0) or ((now - last_send_time) >= send_interval):
                            udp.send(payload)
                            last_send_time = now
                    sent_any_points = True
                else:
                    # out exists but confidence gate failed
                    if args.emit_invalid:
                        frame_id += 1
                        payload = {
                            "type": "hand_points",
                            "frame_id": frame_id,
                            "hand": "right",
                            "ts": int(time.time() * 1000),
                            "valid": False
                        }
                        if args.dry_run or not args.udp:
                            print(json.dumps(payload))
                        else:
                            if (send_interval == 0.0) or ((now - last_send_time) >= send_interval):
                                udp.send(payload); last_send_time = now
                        sent_any_points = True
            else:
                # out is None
                if args.emit_invalid:
                    frame_id += 1
                    payload = {
                        "type": "hand_points",
                        "frame_id": frame_id,
                        "hand": "right",
                        "ts": int(time.time() * 1000),
                        "valid": False
                    }
                    if args.dry_run or not args.udp:
                        print(json.dumps(payload))
                    else:
                        if (send_interval == 0.0) or ((now - last_send_time) >= send_interval):
                            udp.send(payload); last_send_time = now
                    sent_any_points = True

            # ---------- STATUS HEARTBEAT (always) ----------
            if (now - last_status_t) >= max(0.05, float(args.status_every)):
                last_status_t = now
                status = {
                    "type": "hand_status",
                    "ts": int(time.time()*1000),
                    "nL": nL, "nR": nR,
                    "out_valid": bool(out is not None and sent_any_points),
                    "conf": (out.get("conf", {}) if out else {}),
                    "fps": float(fps_ema) if fps_ema is not None else None
                }
                if args.dry_run or not args.udp: print(json.dumps(status))
                else: udp.send(status)

            # [PATCH][STATS] periodic diagnostic summary
            if next_diag_t is not None and time.time() >= next_diag_t:
                try:
                    st = inst.get_stats(reset=False)
                    summary = {
                        "type": "diag_summary",
                        "ts": int(time.time()*1000),
                        "counts": st.get("counts", {}),
                        "palm_pts_hist": st.get("palm_pts_hist", [])
                    }
                    print(json.dumps(summary))
                except Exception as e:
                    print(f"[diag] stats failed: {e}")
                next_diag_t = time.time() + float(args.diag_report_sec)


            # ---------- 콘솔 로그(quiet 지원) ----------
            if stereo_ok and (now - last_log_t > args.z_log_interval) and not getattr(args, 'quiet', False):
                try:
                    print(f"Z(m) palm: {out['palm'][2]: .3f} | wrist: {out['wrist'][2]: .3f} | tip: {out['index_tip'][2]: .3f}")
                except Exception:
                    pass
                last_log_t = now

            if (prev_nL, prev_nR, prev_out_valid, prev_stereo_ok) != (nL, nR, out is not None, stereo_ok) and not getattr(args, 'quiet', False):
                print(f"[state] nL={nL} nR={nR} out_valid={out is not None} stereo_ok={stereo_ok}")
                prev_nL, prev_nR, prev_out_valid, prev_stereo_ok = nL, nR, (out is not None), stereo_ok

            # ---------- 시각화 ----------
            if args.show:
                vis = np.hstack([rl, rr])
                h_, w_ = rl.shape[:2]
                left_roi  = vis[:, :w_]
                right_roi = vis[:, w_:]

                zval = (out['palm'][2] if out is not None else None)
                annotate_status(left_roi,  "LEFT",  nL, fps=fps_ema, zval=zval)
                annotate_status(right_roi, "RIGHT", nR, fps=None,   zval=zval)

                banner = "STEREO OK" if stereo_ok else ("LEFT only" if (nL >= 6 and nR == 0) else
                                                        ("RIGHT only" if (nR >= 6 and nL == 0) else "NO HANDS"))
                color = (0,200,0) if banner=="STEREO OK" else ((0,200,255) if "only" in banner else (0,0,255))
                cv2.putText(vis, banner, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)

                if args.vis_scale and args.vis_scale != 1.0:
                    vis = cv2.resize(vis, None, fx=args.vis_scale, fy=args.vis_scale, interpolation=cv2.INTER_AREA)

                cv2.imshow(win_title, vis)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord('d'), ord('D')):
                    try:
                        rs = inst.get_runtime_state()
                        print(f"[diag] camL_age={rs['camL_age_ms']}ms  camR_age={rs['camR_age_ms']}ms  "
                              f"stash_age={rs['stash_age_ms']}ms  stash_count={rs['stash_count']}")
                    except Exception as e:
                        print("[diag] runtime_state failed:", e)
                if key in (27, ord('q'), ord('Q')):
                    break

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C caught. Shutting down...")

    finally:
        try: inst.stop()
        except Exception: pass
        # [PATCH][STATS] final diagnostic summary
        try:
            if getattr(args, 'diag_final', False):
                st = inst.get_stats(reset=False)
                final = {
                    "type": "diag_final",
                    "ts": int(time.time()*1000),
                    "counts": st.get("counts", {}),
                    "palm_pts_hist": st.get("palm_pts_hist", [])
                }
                print(json.dumps(final))
        except Exception as e:
            print(f"[diag] final stats failed: {e}")

        try:
            if args.show:
                cv2.waitKey(1); cv2.destroyAllWindows()
        except Exception: pass
        time.sleep(0.05)

if __name__ == '__main__':
    main()
