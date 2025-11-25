#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic-rectification version of minimal stereo hand 3D extractor.
- Uses only calibration params (K1,D1,K2,D2,R,t)
- Builds rectify maps (R1,R2,P1,P2,Q) on the fly for the requested resolution
- MediaPipe runs on downscaled frames (detect_scale) and/or every Nth frame (detect_every) for speed
- Public API: get_minimal_hand3d()

Coordinates: meters in rectified left-camera rig frame (C_rect).
"""

from __future__ import annotations
import time
import threading
import queue
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from collections import defaultdict  # [PATCH][STATS]

import numpy as np
import cv2
import platform

try:
    import mediapipe as mp
    MP_AVAILABLE = True
except Exception:
    MP_AVAILABLE = False


# ==========================
# Data containers
# ==========================
@dataclass
class Frame:
    img: np.ndarray
    ts_ms: int

@dataclass
class TriRes:
    xyz: Optional[np.ndarray]  # (3,)
    conf: float


# ==========================
# MediaPipe wrapper (single hand, light model)
# ==========================
class MPHandDetector:
    def __init__(self, min_det_conf: float = 0.5, min_track_conf: float = 0.5):
        if not MP_AVAILABLE:
            raise RuntimeError("mediapipe not available. pip install mediapipe")
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            model_complexity=1,      # light model
            max_num_hands=1,
            min_detection_confidence=min_det_conf,
            min_tracking_confidence=min_track_conf,
        )
        self._ids = [0, 5, 9, 13, 17, 8]  # wrist + 4x MCP + index_tip

    def detect(self, bgr: np.ndarray) -> Dict[int, Tuple[float, float, float]]:
        """Return dict[id] = (u_px, v_px, conf). Conf ~= handedness score."""
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False  # minor speed-up
        res = self._hands.process(rgb)
        out: Dict[int, Tuple[float, float, float]] = {}
        if not res.multi_hand_landmarks:
            return out
        lm = res.multi_hand_landmarks[0]
        conf = 1.0
        if res.multi_handedness and len(res.multi_handedness) > 0:
            try:
                conf = float(res.multi_handedness[0].classification[0].score)
            except Exception:
                conf = 1.0
        for idx in self._ids:
            p = lm.landmark[idx]
            u = float(p.x) * w
            v = float(p.y) * h
            out[idx] = (u, v, conf)
        return out


# ==========================
# Camera thread (backend fallback)
# ==========================
class CamThread:
    def __init__(self, cam_id: int, width: int, height: int, fps: int):
        self.cam_id = cam_id
        self.width, self.height, self.fps = width, height, fps
        self.q: "queue.Queue[Frame]" = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self.cap = None
        self._open_camera_with_fallbacks()

        # capture runtime state
        self._last_ts_ms = 0
        self._last_ok_ms = 0
        self._frame_count = 0

        # non-destructive latest() snapshot
        self._last: Optional[Frame] = None
        self._last_lock = threading.Lock()

        self.th = threading.Thread(target=self._loop, daemon=True)

    def _open_camera_with_fallbacks(self):
        """
        Windows: DSHOW first (stable format negotiation), CAP_ANY fallback.
        macOS:  AVFOUNDATION
        Linux:  V4L2
        Windows often falls back to YUY2 if MJPG is hard; reduce res/FPS to save bandwidth.
        Keep only the newest frame via BUFFERSIZE=1 if available.
        """
        import platform, time
        sysname = platform.system().lower()

        if 'windows' in sysname or 'win' in sysname:
            backends = [getattr(cv2, 'CAP_DSHOW', 700), getattr(cv2, 'CAP_ANY', 0)]
        elif 'darwin' in sysname or 'mac' in sysname:
            backends = [getattr(cv2, 'CAP_AVFOUNDATION', 1200), getattr(cv2, 'CAP_ANY', 0)]
        else:
            backends = [getattr(cv2, 'CAP_V4L2', 200), getattr(cv2, 'CAP_ANY', 0)]

        W, H, FPS = int(self.width), int(self.height), int(self.fps)
        last_err = None

        for be in backends:
            try:
                cap = cv2.VideoCapture(self.cam_id, be)
                if not cap.isOpened():
                    cap.release()
                    continue

                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  W)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
                cap.set(cv2.CAP_PROP_FPS,          FPS)
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass

                if 'windows' in sysname or 'win' in sysname:
                    try:
                        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                    except Exception:
                        pass

                time.sleep(0.05)
                ok, _ = cap.read()
                if not ok:
                    cap.release()
                    continue

                if 'windows' in sysname or 'win' in sysname:
                    rw = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    rh = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    rf = cap.get(cv2.CAP_PROP_FPS)
                    if (rf == 0.0) or (int(rw) != W or int(rh) != H):
                        cap.release()
                        cap = cv2.VideoCapture(self.cam_id, be)
                        if not cap.isOpened():
                            continue
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  960)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
                        cap.set(cv2.CAP_PROP_FPS,          24)
                        try:
                            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        except Exception:
                            pass
                        time.sleep(0.05)
                        ok, _ = cap.read()
                        if not ok:
                            cap.release()
                            continue

                self.cap = cap
                return

            except Exception as e:
                last_err = e
                try:
                    cap.release()
                except Exception:
                    pass
                continue

        raise RuntimeError(f"Failed to open camera {self.cam_id} with any backend. Last error: {last_err}")

    def start(self):
        self.th.start()

    def stop(self):
        self._stop.set()
        self.th.join(timeout=1.5)
        if self.cap:
            self.cap.release()

    def _loop(self):
        # warm-up: auto exposure/focus settle and flush buffer
        warm = 0
        while warm < 10 and not self._stop.is_set():
            ok, _ = self.cap.read()
            if ok:
                warm += 1
            else:
                time.sleep(0.01)

        fail_count = 0
        while not self._stop.is_set():
            ok, img = self.cap.read()
            now_ms = int(time.time() * 1000)

            if not ok or img is None:
                fail_count = min(fail_count + 1, 1000)
                if fail_count >= 50:   # ~0.25–0.5s continuous failure → reopen
                    self._reopen_camera()
                    fail_count = 0
                time.sleep(0.005)
                continue

            # normal frame: update state + keep latest non-destructive snapshot
            self._frame_count += 1
            self._last_ok_ms = now_ms
            ts_ms = now_ms
            fr = Frame(img=img, ts_ms=ts_ms)

            # keep latest snapshot
            with self._last_lock:
                self._last = fr
                self._last_ts_ms = ts_ms

            # compatibility queue (keep newest only)
            if self.q.full():
                try:
                    self.q.get_nowait()
                except Exception:
                    pass
            self.q.put(fr)

            fail_count = 0

    def latest(self) -> Optional[Frame]:
        # return latest snapshot without consuming
        with self._last_lock:
            fr = self._last
        if fr is not None:
            return fr
        # shortly wait up to 20ms on boot
        t0 = time.time()
        while (time.time() - t0) < 0.02:
            time.sleep(0.001)
            with self._last_lock:
                fr = self._last
            if fr is not None:
                return fr
        return None

    def _reopen_camera(self):
        try:
            self.cap.release()
        except:
            pass
        time.sleep(0.03)
        be = getattr(cv2, 'CAP_DSHOW', 700) if platform.system().lower().startswith('win') else 0
        cap = cv2.VideoCapture(self.cam_id, be)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS,          self.fps)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except:
            pass
        time.sleep(0.05)
        self.cap = cap
        print(f"[watchdog] reopened cam{self.cam_id}")

    def get_state(self):
        return {
            "last_ts_ms": int(self._last_ts_ms),
            "last_ok_ms": int(self._last_ok_ms),
            "frames": int(self._frame_count),
        }


# ==========================
# Core: StereoHand3D (dynamic rectify)
# ==========================
class StereoHand3D:
    def __init__(
        self,
        calib_path: str,
        cam_ids: Tuple[int, int] = (0, 1),
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        min_conf: float = 0.5,
        alpha: float = 0.25,
        jump_thresh_m: float = 0.10,
        max_interp_ms: int = 100,
        max_pair_dt_ms: int = 15,
        enable_mediapipe: bool = True,
        detect_scale: float = 0.5,     # run MP on downscaled frames
        detect_every: int = 2,         # run MP every Nth frame
    ):
        # thresholds
        self.min_conf = float(min_conf)
        self.alpha = float(alpha)
        self.jump_thresh_m = float(jump_thresh_m)
        self.max_interp_ms = int(max_interp_ms)
        self.max_pair_dt_ms = int(max_pair_dt_ms)
        self.fps = int(fps)

        # load calibration params (no prebuilt rectify maps)
        cal = np.load(calib_path)
        self.K1, self.D1 = cal['K1'], cal['D1']
        self.K2, self.D2 = cal['K2'], cal['D2']
        self.R,  self.t  = cal['R'],  cal['t']   # baseline encoded in t

        # requested capture size (driver may snap to near mode like 960x544)
        self.width, self.height = int(width), int(height)

        # initial maps set to None; build on first frame size
        self.map1x = self.map1y = self.map2x = self.map2y = None
        self.R1 = self.R2 = self.P1 = self.P2 = self.Q = None

        # cameras
        self.camL = CamThread(cam_ids[0], self.width, self.height, fps)
        self.camR = CamThread(cam_ids[1], self.width, self.height, fps)

        # detector (separate instances for L/R)
        self.detL = MPHandDetector(self.min_conf, self.min_conf) if enable_mediapipe else None
        self.detR = MPHandDetector(self.min_conf, self.min_conf) if enable_mediapipe else None

        # EMA & last valid/interp
        self._ema = {'wrist': None, 'palm': None, 'index_tip': None}
        self._last_valid = None
        self._last_valid_ts = 0

        # preview stash for demo
        self._prev_lock = threading.Lock()
        self._prevL = None
        self._prevR = None

        # share 2D for overlay / reuse
        self._last2d_lock = threading.Lock()
        self._last2dL = None
        self._last2dR = None

        # speed options
        self.detect_scale = float(detect_scale)
        self.detect_every = int(detect_every)
        self._fidx = 0

        # [PATCH][A] frame-pair 기반 detect 스케줄링용 시퀀스/타임스탬프
        self._pair_seq = 0
        self._last_pair_ts = -1

        # output cache
        self._latest_out = None
        self._lock = threading.Lock()

        # worker thread control
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._worker, daemon=True)

        # boot/diag counters
        self._prev_count = 0
        self._prev_last_ts = 0
        self._dbg_last = 0.0

        # L/R auto-swap detector accumulators
        self.lr_swapped = False           # whether L/R is swapped (diagnostic)
        self._lr_check_done = False       # whether check completed
        self._lr_accum = []               # sign(uL-uR) samples

        # sanity-check gate to avoid re-entry
        self._rect_sign_checked  = False

        # [PATCH][STATS] 진단 집계
        self._stats_lock = threading.Lock()
        self._stats = defaultdict(int)
        self._palm_pts_hist = [0]*6  # palm 3D 평균에 참여한 점 개수(0~5) 히스토그램

    # [PATCH][STATS] 내부 카운터 헬퍼/조회
    def _stats_inc(self, key: str, v: int = 1):
        with self._stats_lock:
            self._stats[key] = int(self._stats.get(key, 0)) + int(v)

    def _stats_hist_palm(self, n: int):
        if 0 <= n <= 5:
            with self._stats_lock:
                self._palm_pts_hist[n] += 1

    def get_stats(self, reset: bool = False) -> dict:
        with self._stats_lock:
            counts = dict(self._stats)
            hist = list(self._palm_pts_hist)
            if reset:
                self._stats.clear()
                self._palm_pts_hist = [0]*6
        return {"counts": counts, "palm_pts_hist": hist}


    # ---------- rectify map builder ----------
    def _build_rect_maps(self, width: int, height: int):
        size = (int(width), int(height))  # (w,h)
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            self.K1, self.D1, self.K2, self.D2, size, self.R, self.t,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
        )
        map1x, map1y = cv2.initUndistortRectifyMap(self.K1, self.D1, R1, P1, size, cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(self.K2, self.D2, R2, P2, size, cv2.CV_32FC1)
        # store
        self.R1, self.R2, self.P1, self.P2, self.Q = R1, R2, P1, P2, Q
        self.map1x, self.map1y = map1x, map1y
        self.map2x, self.map2y = map2x, map2y

        # one-time sign sanity check (may trigger full rebuild)
        try:
            self._sanity_rect_sign()
        except Exception:
            pass

    # ---------- lifecycle ----------
    def start(self):
        self.camL.start(); self.camR.start()
        self._th.start()

    def stop(self):
        self._stop.set()
        self._th.join(timeout=1.5)
        self.camL.stop(); self.camR.stop()

    # ---------- helpers ----------
    def _rectify_pair(self, imgL: np.ndarray, imgR: np.ndarray):
        rl = cv2.remap(imgL, self.map1x, self.map1y, interpolation=cv2.INTER_LINEAR)
        rr = cv2.remap(imgR, self.map2x, self.map2y, interpolation=cv2.INTER_LINEAR)
        return rl, rr

    def _reset_ema(self):
        self._ema = {'wrist': None, 'palm': None, 'index_tip': None}
        self._last_valid = None
        self._last_valid_ts = 0

    def _pick_synced(self, fL: Optional[Frame], fR: Optional[Frame]):
        if abs(fL.ts_ms - fR.ts_ms) <= self.max_pair_dt_ms:
            return (fL, fR)

        frame_interval_s = 1.0 / max(1, getattr(self, 'fps', 30))
        budget_s = max(0.04, frame_interval_s)  # ≥40ms
        deadline = time.perf_counter() + budget_s

        while time.perf_counter() < deadline:
            if fL.ts_ms < fR.ts_ms:
                fL2 = self.camL.latest()
                if fL2 is not None:
                    fL = fL2
            else:
                fR2 = self.camR.latest()
                if fR2 is not None:
                    fR = fR2

            if abs(fL.ts_ms - fR.ts_ms) <= self.max_pair_dt_ms:
                return (fL, fR)

            time.sleep(0.0015)

        return None

    def force_rebuild_maps(self, width: int, height: int):
        self._build_rect_maps(int(width), int(height))
        print(f"[info] force_rebuild_maps -> {width}x{height}")

    @staticmethod
    def invert_RT(R: np.ndarray, t: np.ndarray):
        """Inverse of [R|t]: R_inv = R^T, t_inv = -R^T t."""
        R_inv = R.T
        t_vec = t.reshape(3, 1)
        t_inv = - R_inv @ t_vec
        return R_inv, t_inv

    def _apply_full_lr_swap_and_rebuild(self, frame_w: int, frame_h: int):
        """
        Root fix when L/R were mismatched:
          - Swap K1/D1 ↔ K2/D2
          - Invert (R,t) to flip camera order baseline
          - Rebuild maps/projectors for current size
          - Reset EMA/state
        """
        # 1) swap intrinsics/distortions
        self.K1, self.K2 = self.K2, self.K1
        self.D1, self.D2 = self.D2, self.D1

        # 2) invert baseline
        self.R, self.t = StereoHand3D.invert_RT(self.R, self.t)

        # 3) rebuild
        # prevent re-entrant sanity during rebuild; set checked flag after fix
        prev_checked = self._rect_sign_checked
        self._rect_sign_checked = True
        self._build_rect_maps(int(frame_w), int(frame_h))
        self._rect_sign_checked = True  # keep as checked

        # 4) reset EMA and LR flags
        self._reset_ema()
        self.lr_swapped = False
        self._lr_check_done = True
        self._lr_accum = []

        print(f"[lr-fix] FULL swap applied, rebuilt maps for {frame_w}x{frame_h}")
        try:
            print(f"[rectify] P1[0,3]={self.P1[0,3]:.3f}  P2[0,3]={self.P2[0,3]:.3f}")
        except Exception:
            pass

    # raw triangulation with provided P1/P2 and uvL/uvR
    def _triangulate_points_raw(self, P1, P2, uvL, uvR):
        pL = np.array(uvL, dtype=np.float32).reshape(2,1)
        pR = np.array(uvR, dtype=np.float32).reshape(2,1)
        X4 = cv2.triangulatePoints(P1, P2, pL, pR)
        w  = float(X4[3,0])
        if abs(w) < 1e-6:
            return None
        return (X4[:3,0]/w).astype(np.float32)

    # one-time check: ensure Z>0 for positive disparity; if not, apply full fix
    def _sanity_rect_sign(self):
        if self._rect_sign_checked:
            return
        # mark as checked early to avoid recursion on rebuild
        self._rect_sign_checked = True
        if self.P1 is None or self.P2 is None:
            return

        try:
            h = int(self.map1x.shape[0]); w = int(self.map1x.shape[1])
        except Exception:
            h, w = int(self.height), int(self.width)

        u0 = w * 0.5
        v0 = h * 0.5
        d  = 5.0  # positive disparity hypothesis
        uvL = (u0, v0)
        uvR = (u0 - d, v0)
        X = self._triangulate_points_raw(self.P1, self.P2, uvL, uvR)
        if X is None:
            print("[sanity] triangulation failed; keeping current L/R.")
            return

        if float(X[2]) > 0.0:
            print("[sanity] rect chain OK (Z>0) with (L,R).")
        else:
            print("[sanity] rect chain flipped (Z<0). Applying full L/R fix...")
            self._apply_full_lr_swap_and_rebuild(w, h)

    def _detect_both(self, rl, rr):
        if self.detL is None or self.detR is None:
            return {}, {}
        s = float(self.detect_scale)
        if s != 1.0:
            rls = cv2.resize(rl, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            rrs = cv2.resize(rr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            dL = self.detL.detect(rls)
            dR = self.detR.detect(rrs)
            if dL: dL = {k:(u/s, v/s, c) for k,(u,v,c) in dL.items()}
            if dR: dR = {k:(u/s, v/s, c) for k,(u,v,c) in dR.items()}
            return dL, dR
        else:
            return self.detL.detect(rl), self.detR.detect(rr)

    def _triangulate_points(self, uvL, uvR) -> Optional[np.ndarray]:
        pL = np.array(uvL, dtype=np.float32).reshape(2,1)
        pR = np.array(uvR, dtype=np.float32).reshape(2,1)
        X4 = cv2.triangulatePoints(self.P1, self.P2, pL, pR)  # 4x1
        w = float(X4[3,0])
        if abs(w) < 1e-6:
            return None
        X = (X4[:3,0] / w).astype(np.float32)
        return X

    # accumulate sign(uL-uR) to detect runtime L/R mismatch; apply full fix once decided
    def _accum_lr_sign(self, dL: dict, dR: dict):
        if self._lr_check_done:
            return
        try:
            samples = []
            for k in (0,5,9,13,17,8):
                if k in dL and k in dR:
                    uL = float(dL[k][0]); uR = float(dR[k][0])
                    disp = uL - uR
                    if abs(disp) > 0.5:
                        samples.append(1.0 if disp > 0.0 else -1.0)
            if samples:
                self._lr_accum.extend(samples)

            if len(self._lr_accum) >= 20:
                med = float(np.median(self._lr_accum))
                self.lr_swapped = (med < 0.0)
                print(f"[lr-check] median(sign(uL-uR))={med:.1f} -> lr_swapped={self.lr_swapped}")
                self._lr_check_done = True

                if self.lr_swapped:
                    try:
                        h = int(self.map1x.shape[0]); w = int(self.map1x.shape[1])
                    except Exception:
                        h, w = int(self.height), int(self.width)
                    print("[lr-check] Applying full L/R fix based on runtime evidence...")
                    self._apply_full_lr_swap_and_rebuild(w, h)
        except Exception:
            pass

    def _triangulate_set(self, dL: Dict[int,Tuple[float,float,float]], dR: Dict[int,Tuple[float,float,float]]):
        out: Dict[int, TriRes] = {}
        ids = [0,5,9,13,17,8]
        for i in ids:
            if i in dL and i in dR:
                uvL = (dL[i][0], dL[i][1])
                uvR = (dR[i][0], dR[i][1])
                X = self._triangulate_points(uvL, uvR)
                if X is not None:
                    conf = min(dL[i][2], dR[i][2])
                    out[i] = TriRes(X, conf)
                else:
                    out[i] = TriRes(None, 0.0)
            else:
                out[i] = TriRes(None, 0.0)
        return out

    def _compute_palm_center(self, tri: Dict[int, TriRes]) -> TriRes:
        keys = [0,5,9,13,17]
        pts, confs = [], []
        for k in keys:
            if k in tri and tri[k].xyz is not None:
                pts.append(tri[k].xyz); confs.append(tri[k].conf)
        if len(pts) < 3:
            return TriRes(None, 0.0)
        pts = np.stack(pts, axis=0)
        return TriRes(pts.mean(axis=0), float(np.mean(confs)))

    def _ema_apply(self, name: str, raw: Optional[np.ndarray]) -> Optional[np.ndarray]:
        prev = self._ema.get(name)
        if raw is None:
            return prev
        if prev is None:
            self._ema[name] = raw
            return raw

        # immediate accept if Z sign flips (avoid freeze across front/back flip)
        try:
            if float(prev[2]) * float(raw[2]) < 0.0:
                self._ema[name] = raw
                return raw
        except Exception:
            pass

        # [PATCH][B] jump 게이트 → 스텝 클램프: 큰 점프도 최대치만큼은 따라가게
        delta = raw - prev
        dist = float(np.linalg.norm(delta))
        if dist > self.jump_thresh_m:
            raw = prev + (delta * (self.jump_thresh_m / dist))

        out = (1.0 - self.alpha) * prev + self.alpha * raw
        self._ema[name] = out
        return out

    def _package_output(self, tri: Dict[int, TriRes], palm: TriRes):
        if tri[0].xyz is None or tri[8].xyz is None or palm.xyz is None:
            return None
        return {
            'wrist': tuple(map(float, tri[0].xyz)),
            'palm':  tuple(map(float, palm.xyz)),
            'index_tip': tuple(map(float, tri[8].xyz)),
            'conf': {
                'wrist': float(tri[0].conf),
                'palm':  float(palm.conf),
                'index_tip': float(tri[8].conf),
            }
        }

    def _interpolate_if_needed(self, cur_out: Optional[Dict], ts_ms: int):
        if cur_out is not None:
            self._last_valid = cur_out
            self._last_valid_ts = ts_ms
            return cur_out
        if self._last_valid is None:
            return None
        if ts_ms - self._last_valid_ts > self.max_interp_ms:
            return None
        return self._last_valid

    def get_runtime_state(self):
        now_ms = int(time.time()*1000)
        L = self.camL.get_state() if hasattr(self.camL, "get_state") else {}
        R = self.camR.get_state() if hasattr(self.camR, "get_state") else {}
        with self._prev_lock:
            pc = int(getattr(self, "_prev_count", 0))
            pts = int(getattr(self, "_prev_last_ts", 0))
        return {
            "camL_age_ms": (now_ms - L.get("last_ok_ms", now_ms)),
            "camR_age_ms": (now_ms - R.get("last_ok_ms", now_ms)),
            "stash_age_ms": (now_ms - pts if pts else None),
            "stash_count": pc,
        }

    # ---------- worker ----------
    def _worker(self):
        if not hasattr(self, '_dbg_last'):
            self._dbg_last = 0.0
        for name in ('_cnt_none', '_cnt_probe_fail', '_cnt_rect_prev_fail', '_cnt_rect_sync_fail'):
            if not hasattr(self, name):
                setattr(self, name, 0)

        while not self._stop.is_set():
            try:
                # 1) latest frames
                fL = self.camL.latest(); fR = self.camR.latest()
                if fL is None or fR is None:
                    self._cnt_none += 1
                    self._stats_inc('stall_latest_none')  # [PATCH][STATS]
                    if time.time() - self._dbg_last > 1.0:
                        print(f"[stall] latest None: cnt={self._cnt_none}")
                        self._dbg_last = time.time()
                    time.sleep(0.002)
                    continue

                # ---------------------------
                # [preview] ensure map size matches; rebuild if needed
                # ---------------------------
                try:
                    hL, wL = fL.img.shape[:2]
                    hR, wR = fR.img.shape[:2]

                    need_rebuild = (
                        getattr(self, 'map1x', None) is None or getattr(self, 'map2x', None) is None or
                        self.map1x.shape[0] != hL or self.map1x.shape[1] != wL or
                        self.map2x.shape[0] != hR or self.map2x.shape[1] != wR
                    )
                    if need_rebuild:
                        self._build_rect_maps(wL, hL)
                        print(f"[info] rebuilt rectify maps for preview {wL}x{hL}")
                except Exception as e:
                    self._cnt_probe_fail += 1
                    print(f"[ERROR] size probe/rebuild failed (preview): {e} (cnt={self._cnt_probe_fail})")
                    time.sleep(0.01)
                    continue

                # channel guard
                imgL = fL.img; imgR = fR.img
                if imgL.ndim == 2: imgL = cv2.cvtColor(imgL, cv2.COLOR_GRAY2BGR)
                if imgR.ndim == 2: imgR = cv2.cvtColor(imgR, cv2.COLOR_GRAY2BGR)

                # always rectify in (L,R) order — maps match cameras
                try:
                    rl_preview, rr_preview = self._rectify_pair(imgL, imgR)
                except Exception as e:
                    self._cnt_rect_prev_fail += 1
                    try:
                        m1 = (self.map1x.shape[1], self.map1x.shape[0]); m2 = (self.map2x.shape[1], self.map2x.shape[0])
                    except Exception:
                        m1 = m2 = None
                    print(f"[WARN] rectify failed (preview): {e} | frameL={imgL.shape[:2]} frameR={imgR.shape[:2]} mapL={m1} mapR={m2} (cnt={self._cnt_rect_prev_fail})")
                    with self._prev_lock:
                        self._prevL = imgL
                        self._prevR = imgR
                        if not hasattr(self, '_prev_count'): self._prev_count = 0
                        if not hasattr(self, '_prev_last_ts'): self._prev_last_ts = 0
                        self._prev_count += 1
                        self._prev_last_ts = int(time.time()*1000)
                    time.sleep(0.01)
                    continue

                with self._prev_lock:
                    self._prevL = rl_preview
                    self._prevR = rr_preview
                    if not hasattr(self, '_prev_count'): self._prev_count = 0
                    if not hasattr(self, '_prev_last_ts'): self._prev_last_ts = 0
                    self._prev_count += 1
                    self._prev_last_ts = int(time.time()*1000)

                # ---------------------------
                # [3D] build synced pair (reconfirm map size)
                # ---------------------------
                pair = self._pick_synced(fL, fR)
                if pair is None:
                    self._stats_inc('sync_fail')  # [PATCH][STATS]
                    continue
                fL, fR = pair
                self._stats_inc('frames_3d')     # [PATCH][STATS] 3D 파이프라인 진입 프레임

                try:
                    hL2, wL2 = fL.img.shape[:2]
                    hR2, wR2 = fR.img.shape[:2]
                    need_rebuild2 = (
                        self.map1x.shape[0] != hL2 or self.map1x.shape[1] != wL2 or
                        self.map2x.shape[0] != hR2 or self.map2x.shape[1] != wR2
                    )
                    if need_rebuild2:
                        self._build_rect_maps(wL2, hL2)
                        print(f"[info] rebuilt rectify maps for synced {wL2}x{hL2}")
                except Exception as e:
                    print(f"[ERROR] size probe/rebuild failed (synced): {e}")
                    time.sleep(0.01)
                    continue

                imgL2 = fL.img; imgR2 = fR.img
                if imgL2.ndim == 2: imgL2 = cv2.cvtColor(imgL2, cv2.COLOR_GRAY2BGR)
                if imgR2.ndim == 2: imgR2 = cv2.cvtColor(imgR2, cv2.COLOR_GRAY2BGR)

                # rectify (always L,R; if chain was flipped, a full fix has rebuilt maps already)
                try:
                    rl, rr = self._rectify_pair(imgL2, imgR2)
                except Exception as e:
                    self._cnt_rect_sync_fail += 1
                    try:
                        m1 = (self.map1x.shape[1], self.map1x.shape[0]); m2 = (self.map2x.shape[1], self.map2x.shape[0])
                    except Exception:
                        m1 = m2 = None
                    print(f"[WARN] rectify failed (synced): {e} | frameL={imgL2.shape[:2]} frameR={imgR2.shape[:2]} mapL={m1} mapR={m2} (cnt={self._cnt_rect_sync_fail})")
                    time.sleep(0.01)
                    continue

                # [PATCH][A] 프레임 페어 기반 detect 스케줄링
                pair_ts = int(max(fL.ts_ms, fR.ts_ms))
                if pair_ts != self._last_pair_ts:
                    self._pair_seq += 1
                    self._last_pair_ts = pair_ts

                # 2D detection (downscale / frame skip)
                self._fidx += 1  # (유지; 더 이상 스케줄링엔 사용하지 않음)
                do_detect = (self._pair_seq % max(1, self.detect_every) == 0)  # [PATCH][A]

                if do_detect:
                    self._stats_inc('detect_runs')
                    dL, dR = self._detect_both(rl, rr)
                    with self._last2d_lock:
                        self._last2dL = dict(dL)
                        self._last2dR = dict(dR)
                        self._accum_lr_sign(dL, dR)
                else:
                    self._stats_inc('detect_reuse')
                    with self._last2d_lock:
                        dL = {} if self._last2dL is None else dict(self._last2dL)
                        dR = {} if self._last2dR is None else dict(self._last2dR)

                # [PATCH][STATS2] L/R miss 및 히스토그램
                ids_req = [0,5,9,13,17,8]
                presentL = set(dL.keys()); presentR = set(dR.keys())
                nL = len(presentL); nR = len(presentR)
                self._stats_inc(f'hist2d_L_{nL}')
                self._stats_inc(f'hist2d_R_{nR}')

                # 랜드마크별 miss 카운트
                for i in ids_req:
                    if i not in presentL: self._stats_inc(f'miss2d_L_{i}')
                    if i not in presentR: self._stats_inc(f'miss2d_R_{i}')

                # wrist/tip 동시/단독 miss 분류
                for tag,i in [('wrist',0), ('tip',8)]:
                    Lh = (i in presentL); Rh = (i in presentR)
                    if not Lh and not Rh: self._stats_inc(f'miss2d_{tag}_both')
                    elif not Lh and Rh:   self._stats_inc(f'miss2d_{tag}_left_only')
                    elif Lh and not Rh:   self._stats_inc(f'miss2d_{tag}_right_only')


                # 3D triangulation & palm
                tri  = self._triangulate_set(dL, dR)
                palm = self._compute_palm_center(tri)

                # [PATCH][STATS] palm 평균에 쓰인 3D 점 개수 및 2D/tri/conf 상태 스냅샷
                palm_keys = [0,5,9,13,17]
                palm_pts_n = sum(1 for k in palm_keys if (k in tri and tri[k].xyz is not None))
                self._stats_hist_palm(palm_pts_n)

                wrist_has2d = (0 in dL and 0 in dR)
                tip_has2d   = (8 in dL and 8 in dR)
                wrist_tri_ok_pre = (tri[0].xyz is not None)
                tip_tri_ok_pre   = (tri[8].xyz is not None)
                palm_ok_pre      = (palm.xyz is not None)

                wrist_conf_pre = float(tri[0].conf)
                tip_conf_pre   = float(tri[8].conf)
                palm_conf_pre  = float(palm.conf)

                wrist_conf_fail = (wrist_tri_ok_pre and wrist_conf_pre < self.min_conf)  # [PATCH][STATS]
                tip_conf_fail   = (tip_tri_ok_pre   and tip_conf_pre   < self.min_conf)  # [PATCH][STATS]
                palm_conf_fail  = (palm_ok_pre      and palm_conf_pre  < self.min_conf)  # [PATCH][STATS]

                # confidence gating
                if tri[0].conf < self.min_conf:
                    tri[0] = TriRes(None, 0.0)
                if tri[8].conf < self.min_conf:
                    tri[8] = TriRes(None, 0.0)
                if palm.conf < self.min_conf:
                    palm = TriRes(None, 0.0)

                # EMA
                tri0  = self._ema_apply('wrist',     tri[0].xyz)
                palmF = self._ema_apply('palm',      palm.xyz)
                tipF  = self._ema_apply('index_tip', tri[8].xyz)

                tri_out  = {
                    0: TriRes(tri0,  tri[0].conf if tri0  is not None else 0.0),
                    8: TriRes(tipF,  tri[8].conf if tipF  is not None else 0.0),
                }
                palm_out = TriRes(palmF, palm.conf if palmF is not None else 0.0)

                cur = self._package_output(tri_out, palm_out)
                

                # [PATCH][STATS] 패키징 직전 유효성 판정(EMA 이후가 아닌, 게이트 반영 기준)
                pre_pack_wrist_ok = (tri[0].xyz is not None)
                pre_pack_tip_ok   = (tri[8].xyz is not None)
                pre_pack_palm_ok  = (palm.xyz   is not None)
                pre_ok = (pre_pack_wrist_ok and pre_pack_tip_ok and pre_pack_palm_ok)

                if pre_ok:
                    self._stats_inc('frames_pre_ok')
                else:
                    self._stats_inc('frames_invalid')

                    # non-exclusive components
                    if not wrist_has2d: self._stats_inc('comp_wrist_missing2d')
                    if not tip_has2d:   self._stats_inc('comp_tip_missing2d')
                    if palm_pts_n < 3:  self._stats_inc('comp_palm_pts_lt3')
                    if wrist_conf_fail: self._stats_inc('comp_wrist_conf_gate')
                    if tip_conf_fail:   self._stats_inc('comp_tip_conf_gate')
                    if palm_conf_fail:  self._stats_inc('comp_palm_conf_gate')
                    if wrist_has2d and not wrist_tri_ok_pre: self._stats_inc('comp_wrist_tri_fail')
                    if tip_has2d   and not tip_tri_ok_pre:   self._stats_inc('comp_tip_tri_fail')

                    # primary(단일 분류; 우선순위)
                    if   not wrist_has2d: code = 'MISS2D_WRIST'
                    elif not tip_has2d:   code = 'MISS2D_TIP'
                    elif palm_pts_n < 3:  code = 'PALM_PTS_LT3'
                    elif wrist_conf_fail: code = 'CONF_WRIST'
                    elif tip_conf_fail:   code = 'CONF_TIP'
                    elif palm_conf_fail:  code = 'CONF_PALM'
                    elif not wrist_tri_ok_pre: code = 'TRI_WRIST'
                    elif not tip_tri_ok_pre:   code = 'TRI_TIP'
                    else: code = 'OTHER'
                    self._stats_inc(f'primary_{code}')

                cur = self._interpolate_if_needed(cur, fL.ts_ms)

                # [PATCH][C] 출력 메타 추가: ts_src_ms / detected_now / interpolated
                if cur is not None:
                    interpolated = (cur is self._last_valid)
                    ts_src_ms = int(self._last_valid_ts if interpolated else pair_ts)
                    cur = dict(cur)
                    cur['ts_src_ms'] = ts_src_ms
                    cur['detected_now'] = bool(do_detect)
                    cur['interpolated'] = bool(interpolated)

                # [PATCH][STATS] 보간/최종 출력 집계
                if cur is not None:
                    self._stats_inc('frames_output')
                    if cur.get('interpolated', False):
                        self._stats_inc('frames_interpolated')
                else:
                    self._stats_inc('frames_output_none')


                with self._lock:
                    self._latest_out = cur

            except Exception as e:
                import traceback
                print("[FATAL] worker loop exception:", e)
                traceback.print_exc()
                time.sleep(0.01)
                continue

    def get_debug_state(self):
        with self._prev_lock:
            pc = int(getattr(self, "_prev_count", 0))
            pts = int(getattr(self, "_prev_last_ts", 0))
            m1 = None if getattr(self, "map1x", None) is None else (self.map1x.shape[1], self.map1x.shape[0])
            m2 = None if getattr(self, "map2x", None) is None else (self.map2x.shape[1], self.map2x.shape[0])
        return {"prev_count": pc, "prev_last_ms": pts, "mapL_size": m1, "mapR_size": m2}

    # ---------- public getters ----------
    def get(self) -> Optional[Dict]:
        with self._lock:
            return None if self._latest_out is None else dict(self._latest_out)

    def get_preview_frames(self):
        with self._prev_lock:
            if self._prevL is None or self._prevR is None:
                return None
            return self._prevL, self._prevR

    def get_last_2d(self):
        with self._last2d_lock:
            if self._last2dL is None or self._last2dR is None:
                return None
            return dict(self._last2dL), dict(self._last2dR)


# ==========================
# Module-level singleton & API
# ==========================
_instance: Optional[StereoHand3D] = None

def initialize(
    calib_path: str,
    rect_path: str = "",          # kept for compatibility; ignored
    cam_left: int = 0,
    cam_right: int = 1,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    alpha: float = 0.25,
    min_conf: float = 0.5,
    jump_thresh_m: float = 0.10,
    max_interp_ms: int = 100,
    max_pair_dt_ms: int = 15,
    detect_scale: float = 0.5,
    detect_every: int = 2,
):
    """Initialize singleton engine with dynamic rectify maps."""
    global _instance
    if _instance is not None:
        return _instance
    _instance = StereoHand3D(
        calib_path=calib_path,
        cam_ids=(cam_left, cam_right),
        width=width, height=height, fps=fps,
        min_conf=min_conf, alpha=alpha,
        jump_thresh_m=jump_thresh_m,
        max_interp_ms=max_interp_ms, max_pair_dt_ms=max_pair_dt_ms,
        enable_mediapipe=True,
        detect_scale=detect_scale,
        detect_every=detect_every,
    )
    _instance.start()
    return _instance


def get_minimal_hand3d() -> Optional[dict]:
    """
    Returns:
      {
        "wrist": (x, y, z),      # meters, in rectified rig frame C_rect
        "palm":  (x, y, z),
        "index_tip": (x, y, z),
        "conf": { "wrist": float, "palm": float, "index_tip": float }  # 0~1,
        # [PATCH][C] 아래 메타 필드가 추가됩니다.
        # "ts_src_ms": int,       # 이 출력이 기반한 소스 프레임 시각(ms)
        # "detected_now": bool,   # 이번 반복에서 실제 2D 검출을 돌렸는지
        # "interpolated": bool    # 이번 출력이 보간(hold-last)인지
      }
    or None (invalid frame).
    """
    global _instance
    if _instance is None:
        return None
    return _instance.get()


if __name__ == "__main__":
    print("This module provides get_minimal_hand3d(). Use your demo script to run the preview.")
