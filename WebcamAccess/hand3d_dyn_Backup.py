#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic-rectification version of minimal stereo hand 3D extractor.
- Uses only calibration params (K1,D1,K2,D2,R,t)
- Builds rectify maps (R1,R2,P1,P2,Q) on the fly for the requested resolution
- MediaPipe runs on downscaled frames (detect_scale) and/or every Nth frame (detect_every) for speed
- Public API: get_minimal_hand3d()

Coordinates: meters in rectified left-camera rig frame (C_rect).

Added:
- [PATCH][A] Frame-pair-based detect scheduling
- [PATCH][B] EMA step-clamp for large jumps
- [PATCH][C] Output meta (ts_src_ms / detected_now / interpolated)
- [PATCH][MONO] Mono fallback using last disparity for short gaps (<= max_interp_ms)
- [PATCH][GESTURE-ANGLE] Angle-based Open/Closed using 21pt 3D + hysteresis
"""

from __future__ import annotations
import time
import threading
import queue
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Union
from pathlib import Path 
import hashlib

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
# MediaPipe wrapper (single hand, full 21 points)
# ==========================
class MPHandDetector:
    def __init__(self, min_det_conf: float = 0.5, min_track_conf: float = 0.5, model_complexity: int = 1, max_hands: int = 1):
        if not MP_AVAILABLE:
            raise RuntimeError("mediapipe not available. pip install mediapipe")
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            model_complexity=int(model_complexity),         # 0 or 1
            max_num_hands=int(max_hands),
            min_detection_confidence=float(min_det_conf),
            min_tracking_confidence=float(min_track_conf),
        )
        # 21 landmarks: 0..20
        self._ids = list(range(21))

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
        is_win = sysname.startswith('win')
        is_mac = sysname.startswith('darwin') or 'mac' in sysname

        if is_win:
            backends = [getattr(cv2, 'CAP_DSHOW', 700), getattr(cv2, 'CAP_ANY', 0)]
        elif is_mac:
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

                if is_win:
                    try:
                        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                    except Exception:
                        pass

                time.sleep(0.05)
                ok, _ = cap.read()
                if not ok:
                    cap.release()
                    continue

                if is_win:
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
        self.th.join(timeout=1.5)  # cleaned: do not store return value (join() returns None)
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass

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
        # improved: consistent backend choice and Windows MJPG FOURCC on reopen
        try:
            self.cap.release()
        except:
            pass
        time.sleep(0.03)
        sysname = platform.system().lower()
        is_win = sysname.startswith('win')
        be = getattr(cv2, 'CAP_DSHOW', 700) if is_win else getattr(cv2, 'CAP_ANY', 0)
        cap = cv2.VideoCapture(self.cam_id, be)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS,          self.fps)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except:
            pass
        if is_win:
            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            except Exception:
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
        # --- MP params (optional; fall back to min_conf when None)
        mp_det_conf: Optional[float] = None,
        mp_track_conf: Optional[float] = None,
        mp_model_complexity: int = 1,
        mp_max_hands: int = 1,
        rect_alpha: float = 0.0,
        # --- [GESTURE-ANGLE] options (repurpose gesture_* flags; no model needed)
        gesture_enable: bool = False,
        gesture_model: Optional[Union[str, bytes]] = None,  # ignored
        gesture_every: int = 1,
        gesture_min_score: float = 0.6,  # ignored; keep for compatibility
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
        # enforce t shape (3,1) for consistency across the pipeline
        tvec = np.array(self.t).reshape(3, -1)
        self.t = tvec if tvec.shape == (3, 1) else tvec[:, :1]

        # requested capture size (driver may snap to near mode like 960x544)
        self.width, self.height = int(width), int(height)

        # initial maps set to None; build on first frame size
        self.map1x = self.map1y = self.map2x = self.map2y = None
        self.R1 = self.R2 = self.P1 = self.P2 = self.Q = None

        # cameras
        self.camL = CamThread(cam_ids[0], self.width, self.height, fps)
        self.camR = CamThread(cam_ids[1], self.width, self.height, fps)

        # MP thresholds (None => use self.min_conf)
        self.mp_det_conf   = float(mp_det_conf)   if mp_det_conf   is not None else self.min_conf
        self.mp_track_conf = float(mp_track_conf) if mp_track_conf is not None else self.min_conf
        self.mp_model_complexity = int(mp_model_complexity)
        if self.mp_model_complexity not in (0, 1):
            self.mp_model_complexity = 1
        self.mp_max_hands = int(mp_max_hands)

        # detector (separate instances for L/R)
        self.detL = MPHandDetector(
            min_det_conf=self.mp_det_conf,
            min_track_conf=self.mp_track_conf,
            model_complexity=self.mp_model_complexity,
            max_hands=self.mp_max_hands
        ) if enable_mediapipe else None
        self.detR = MPHandDetector(
            min_det_conf=self.mp_det_conf,
            min_track_conf=self.mp_track_conf,
            model_complexity=self.mp_model_complexity,
            max_hands=self.mp_max_hands
        ) if enable_mediapipe else None

        # [GESTURE-ANGLE] state / hysteresis (no Tasks, no model)
        self._gesture_enable     = bool(gesture_enable)
        self._gesture_every      = int(gesture_every)
        self._gesture_last       = None   # {"name": str, "score": float, "src": "angles"}
        self._open_cnt           = 0
        self._close_cnt          = 0
        # 히스테리시스: 해제(Open)는 보수적으로, 그랩(Closed)은 관대하게
        self._OPEN_K             = 2      # 연속 N프레임
        self._OPEN_THRESH        = 0.65   # 해제 임계(각도 기반 score)
        self._CLOSE_K            = 1      # 연속 N프레임
        self._CLOSE_THRESH       = 0.60   # 그랩 임계(각도 기반 score)

        # EMA & last valid/interp
        self._ema = {'wrist': None, 'palm': None, 'index_tip': None}
        self._last_valid = None
        self._last_valid_ts = 0

        # [PATCH][MONO] store last disparity per landmark for mono fallback
        self._last_disp = {i: None for i in range(21)}

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
        self._cur_pair_ts_ms = 0  # [PATCH][MONO] current pair timestamp

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

        self._prev_pre_ok = True   # [PATCH][FORCE] 직전 반복에서 3D가 실제 생성되었는지

        self.rect_alpha = float(rect_alpha)

    # ---------- rectify map builder ----------
    def _build_rect_maps(self, width: int, height: int):
        size = (int(width), int(height))  # (w,h)
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            self.K1, self.D1, self.K2, self.D2, size, self.R, self.t,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=self.rect_alpha
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
        # keep (3,1)
        self.t = np.array(self.t).reshape(3, 1)

        # 3) rebuild
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
            for k in range(21):
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

    # [PATCH][MONO] Triangulate with mono fallback (uses last disparity for short gaps)
    def _triangulate_set(self, dL: Dict[int,Tuple[float,float,float]], dR: Dict[int,Tuple[float,float,float]]):
        out: Dict[int, TriRes] = {}
        ids = list(range(21))

        # allow mono fallback only within max_interp_ms since last valid output
        now_ts = int(getattr(self, '_cur_pair_ts_ms', 0))
        allow_mono = (
            self._last_valid is not None and
            now_ts > 0 and
            (now_ts - int(self._last_valid_ts)) <= int(self.max_interp_ms)
        )

        for i in ids:
            hasL = (i in dL); hasR = (i in dR)

            if hasL and hasR:
                uvL = (float(dL[i][0]), float(dL[i][1]))
                uvR = (float(dR[i][0]), float(dR[i][1]))
                X = self._triangulate_points(uvL, uvR)
                if X is not None:
                    try:
                        self._last_disp[i] = float(uvL[0] - uvR[0])
                    except Exception:
                        pass
                    conf = float(min(dL[i][2], dR[i][2]))
                    out[i] = TriRes(X, conf)
                else:
                    out[i] = TriRes(None, 0.0)

            elif allow_mono and (self._last_disp.get(i) is not None):
                # synthesize the missing view using last disparity (rectified: same v)
                disp = float(self._last_disp[i])

                if hasL and not hasR:
                    uL = float(dL[i][0]); v = float(dL[i][1])
                    uR = uL - disp
                    X = self._triangulate_points((uL, v), (uR, v))
                    conf = float(dL[i][2]) * 0.90
                    out[i] = TriRes(X, conf if X is not None else 0.0)

                elif hasR and not hasL:
                    uR = float(dR[i][0]); v = float(dR[i][1])
                    uL = uR + disp
                    X = self._triangulate_points((uL, v), (uR, v))
                    conf = float(dR[i][2]) * 0.90
                    out[i] = TriRes(X, conf if X is not None else 0.0)

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

        # [PATCH][B] step clamp
        delta = raw - prev
        dist = float(np.linalg.norm(delta))
        if dist > self.jump_thresh_m:
            raw = prev + (delta * (self.jump_thresh_m / dist))

        out = (1.0 - self.alpha) * prev + self.alpha * raw
        self._ema[name] = out
        return out

    # [PATCH][ROI] 직전 2D로부터 ROI bbox 산출
    def _bbox_from_2d(self, d: dict, w: int, h: int, margin_scale: float = 1.6):
        if not d:
            return None
        us = [float(u) for (u,_,_) in d.values()]
        vs = [float(v) for (_,v,_) in d.values()]
        umin, umax = max(0.0, min(us)), min(float(w-1), max(us))
        vmin, vmax = max(0.0, min(vs)), min(float(h-1), max(vs))
        cx, cy = (umin+umax)/2.0, (vmin+vmax)/2.0
        bw, bh = max(umax-umin, 60.0), max(vmax-vmin, 60.0)
        bw *= margin_scale; bh *= margin_scale
        x0 = int(max(0, cx - bw/2)); y0 = int(max(0, cy - bh/2))
        x1 = int(min(w, cx + bw/2)); y1 = int(min(h, cy + bh/2))
        if x1 - x0 < 10 or y1 - y0 < 10:
            return None
        return (x0, y0, x1, y1)

    # [PATCH][ROI] ROI에서 재검출하고, 좌표를 원 프레임 기준으로 복원
    def _detect_in_roi(self, det: MPHandDetector, img: np.ndarray, roi):
        if det is None or roi is None:
            return {}
        x0, y0, x1, y1 = roi
        crop = img[y0:y1, x0:x1]
        if crop.size == 0:
            return {}
        d = det.detect(crop)
        if not d:
            return {}
        return {k: (u + x0, v + y0, c) for k, (u, v, c) in d.items()}

    def _package_output(self, tri: Dict[int, TriRes], palm: TriRes):
        if tri.get(0, TriRes(None,0)).xyz is None or tri.get(8,TriRes(None,0)).xyz is None or palm.xyz is None:
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

    # ---------- [GESTURE-ANGLE] helpers ----------
    @staticmethod
    def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Optional[float]:
        if a is None or b is None or c is None:
            return None
        BA = a - b; BC = c - b
        den = (np.linalg.norm(BA) * np.linalg.norm(BC) + 1e-8)
        if den <= 1e-7:
            return None
        v = np.clip(float(np.dot(BA, BC) / den), -1.0, 1.0)
        return float(np.degrees(np.arccos(v)))

    def _gesture_from_tri(self, tri: Dict[int, TriRes]) -> Optional[dict]:
        """Return {'name','score'} based on finger curl angles & palm ratio, or None if insufficient."""
        # indices
        fingers = {"index":(5,6,7,8),"middle":(9,10,11,12),"ring":(13,14,15,16),"little":(17,18,19,20)}
        curl = []
        have_any = False
        for (mcp,pip,dip,tip) in fingers.values():
            a = tri.get(mcp, TriRes(None,0)).xyz
            b = tri.get(pip, TriRes(None,0)).xyz
            c = tri.get(dip, TriRes(None,0)).xyz
            th = self._angle_deg(a,b,c)
            curl.append(th)
            have_any |= (th is not None)
        if not have_any:
            return None

        # palm center & ratio (optional but helpful)
        palm_keys = [0,5,9,13,17]
        palm_pts = [tri[k].xyz for k in palm_keys if (k in tri and tri[k].xyz is not None)]
        ratio = None
        if len(palm_pts) >= 3:
            pc = np.mean(np.stack(palm_pts,0), axis=0)
            try:
                tips = [tri[i].xyz for i in [8,12,16,20] if tri.get(i,TriRes(None,0)).xyz is not None]
                mcps = [tri[i].xyz for i in [5,9,13,17] if tri.get(i,TriRes(None,0)).xyz is not None]
                if len(tips) >= 3 and len(mcps) >= 3:
                    tips_d  = np.mean([np.linalg.norm(t - pc) for t in tips])
                    mcps_d  = np.mean([np.linalg.norm(m - pc) for m in mcps]) + 1e-8
                    ratio = float(tips_d / mcps_d)
            except Exception:
                ratio = None

        # counts
        bent     = sum([1 for th in curl if (th is not None and th <= 70.0)])
        straight = sum([1 for th in curl if (th is not None and th >= 110.0)])

        # score heuristics
        if bent >= 3:
            score = min(0.95, 0.60 + 0.07*(bent-3))
            return {"name":"Closed_Fist", "score":score, "dbg":{"curl":curl, "ratio":ratio}}
        if straight >= 3 and (ratio is None or ratio >= 1.45):
            extra = 0.0 if ratio is None else max(0.0, min(0.2, 0.15*(ratio-1.45)/0.10))
            score = min(0.95, 0.62 + 0.06*(straight-3) + extra)
            return {"name":"Open_Palm", "score":score, "dbg":{"curl":curl, "ratio":ratio}}

        return {"name":"None", "score":0.0, "dbg":{"curl":curl, "ratio":ratio}}

    def _update_gesture_hysteresis(self, cand: Optional[dict], ts_ms: int) -> Optional[dict]:
        """Apply hysteresis; None or weak evidence does not flip state."""
        if cand is None:
            # decay a bit, but don't change state
            self._open_cnt  = max(0, self._open_cnt - 1)
            self._close_cnt = max(0, self._close_cnt - 1)
            return None

        name = cand.get("name","None")
        score = float(cand.get("score", 0.0))

        decided = None
        if name == "Closed_Fist" and score >= self._CLOSE_THRESH:
            self._close_cnt = min(self._close_cnt + 1, self._CLOSE_K)
            self._open_cnt = 0
            if self._close_cnt >= self._CLOSE_K:
                decided = {"name":"Closed_Fist","score":score,"src":"angles"}
        elif name == "Open_Palm" and score >= self._OPEN_THRESH:
            self._open_cnt = min(self._open_cnt + 1, self._OPEN_K)
            self._close_cnt = 0
            if self._open_cnt >= self._OPEN_K:
                decided = {"name":"Open_Palm","score":score,"src":"angles"}
        else:
            # weak/None → hold
            self._open_cnt  = max(0, self._open_cnt - 1)
            self._close_cnt = max(0, self._close_cnt - 1)

        if decided is not None:
            prev = (self._gesture_last.get("name") if self._gesture_last else None)
            self._gesture_last = dict(decided)
            if decided["name"] != prev:
                # 간단 로그 (Top-K가 없으니 score만)
                print(f"[glog][state] -> {decided['name']} (score={decided['score']:.2f})")
            return decided
        return None

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
                    continue
                fL, fR = pair

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

                # rectify
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

                self._cur_pair_ts_ms = pair_ts  # [PATCH][MONO]

                # 2D detection (downscale / frame skip)
                self._fidx += 1
                do_detect = (self._pair_seq % max(1, self.detect_every) == 0)

                # [PATCH][FORCE] 직전 프레임 실패 → 이번엔 강제 검출
                if not getattr(self, '_prev_pre_ok', True):
                    do_detect = True

                if do_detect:
                    dL, dR = self._detect_both(rl, rr)

                    # ROI retry helper (partial detection triggers retry)
                    def _needs_roi_retry(d: dict, min_count: int = 6) -> bool:
                        return (not d) or (len(d) < min_count)

                    # [PATCH][ROI] 재시도
                    if _needs_roi_retry(dL):
                        with self._last2d_lock:
                            lastL = {} if self._last2dL is None else dict(self._last2dL)
                        h_, w_ = rl.shape[:2]
                        roi1 = self._bbox_from_2d(lastL, w_, h_, 1.6)
                        if roi1:
                            dL = self._detect_in_roi(self.detL, rl, roi1)
                        if _needs_roi_retry(dL):
                            roi2 = self._bbox_from_2d(lastL, w_, h_, 2.2)
                            if roi2:
                                dL = self._detect_in_roi(self.detL, rl, roi2)

                    if _needs_roi_retry(dR):
                        with self._last2d_lock:
                            lastR = {} if self._last2dR is None else dict(self._last2dR)
                        h_, w_ = rr.shape[:2]
                        roi1 = self._bbox_from_2d(lastR, w_, h_, 1.6)
                        if roi1:
                            dR = self._detect_in_roi(self.detR, rr, roi1)
                        if _needs_roi_retry(dR):
                            roi2 = self._bbox_from_2d(lastR, w_, h_, 2.2)
                            if roi2:
                                dR = self._detect_in_roi(self.detR, rr, roi2)

                    with self._last2d_lock:
                        self._last2dL = dict(dL)
                        self._last2dR = dict(dR)
                        self._accum_lr_sign(dL, dR)
                else:
                    with self._last2d_lock:
                        dL = {} if self._last2dL is None else dict(self._last2dL)
                        dR = {} if self._last2dR is None else dict(self._last2dR)

                # 3D triangulation (21pts) & palm
                tri  = self._triangulate_set(dL, dR)
                palm = self._compute_palm_center(tri)

                # confidence gating for minimal outputs
                if tri.get(0,TriRes(None,0)).conf < self.min_conf:
                    tri[0] = TriRes(None, 0.0)
                if tri.get(8,TriRes(None,0)).conf < self.min_conf:
                    tri[8] = TriRes(None, 0.0)
                if palm.conf < self.min_conf:
                    palm = TriRes(None, 0.0)

                # EMA for 3 keypoints (output smoothing)
                tri0  = self._ema_apply('wrist',     tri[0].xyz)
                palmF = self._ema_apply('palm',      palm.xyz)
                tipF  = self._ema_apply('index_tip', tri[8].xyz)

                tri_out  = {
                    0: TriRes(tri0,  tri[0].conf if tri0  is not None else 0.0),
                    8: TriRes(tipF,  tri[8].conf if tipF  is not None else 0.0),
                }
                palm_out = TriRes(palmF, palm.conf if palmF is not None else 0.0)

                # === joints_c 추가: 21개 관절 C_rect 3D + conf ===
                joints_c = []
                for i in range(21):
                    r = tri.get(i, TriRes(None,0.0))
                    if r.xyz is not None:
                        x,y,z = map(float, r.xyz)
                        joints_c.append((x,y,z,float(r.conf)))
                    else:
                        joints_c.append((None,None,None,0.0))

                cur = self._package_output(tri_out, palm_out)
                if cur is not None:
                    cur = dict(cur)
                    cur["joints_c"] = joints_c  # 엔진 내부(C_rect) 좌표계

                # [GESTURE-ANGLE] 업데이트 (detect 실행된 프레임에서만)
                if self._gesture_enable and do_detect and (self._pair_seq % max(1, self._gesture_every) == 0):
                    cand = self._gesture_from_tri(tri)
                    _ = self._update_gesture_hysteresis(cand, pair_ts)

                # [PATCH][FORCE] 다음 루프 강제검출 여부
                self._prev_pre_ok = bool(cur is not None)

                pre_interp_none = (cur is None)
                cur = self._interpolate_if_needed(cur, fL.ts_ms)

                # (옵션) 현재 안정 제스처를 출력 객체에 포함
                if cur is not None and self._gesture_last:
                    cur = dict(cur)
                    cur["gesture"] = dict(self._gesture_last)  # {"name","score","src":"angles"}

                # [PATCH][C] meta
                if cur is not None:
                    interpolated = bool(pre_interp_none)
                    ts_src_ms = int(self._last_valid_ts if interpolated else pair_ts)
                    cur["ts_src_ms"]  = ts_src_ms
                    cur["detected_now"] = bool(do_detect)
                    cur["interpolated"] = bool(interpolated)

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
_instance: Optional['StereoHand3D'] = None  # fixed: correct engine type hint

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
    mp_det_conf: Optional[float] = None,
    mp_track_conf: Optional[float] = None,
    mp_model_complexity: int = 1,
    mp_max_hands: int = 1,
    rect_alpha: float = 0.0,
    # [GESTURE-ANGLE] passthrough (model ignored)
    gesture_enable: bool = False,
    gesture_model: str = "",
    gesture_every: int = 1,
    gesture_min_score: float = 0.6,
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
        mp_det_conf=mp_det_conf,
        mp_track_conf=mp_track_conf,
        mp_model_complexity=mp_model_complexity,
        mp_max_hands=mp_max_hands,
        rect_alpha=rect_alpha,
        # [GESTURE-ANGLE]
        gesture_enable=gesture_enable,
        gesture_model=gesture_model,
        gesture_every=gesture_every,
        gesture_min_score=gesture_min_score,
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
        "conf": { "wrist": float, "palm": float, "index_tip": float },  # 0~1
        # meta:
        # "ts_src_ms": int, "detected_now": bool, "interpolated": bool
        # gesture (angle-based, optional):
        # "gesture": { "name": "Closed_Fist|Open_Palm", "score": float, "src": "angles" }
        # joints_c:
        # "joints_c": list of 21 tuples (x,y,z,conf); if unavailable -> (None,None,None,0.0)
      }
    or None (invalid frame).
    """
    global _instance
    if _instance is None:
        return None
    return _instance.get()


if __name__ == "__main__":
    print("This module provides get_minimal_hand3d(). Use your demo script to run the preview.")
