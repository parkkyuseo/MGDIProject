#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stereo_capture_charuco.py (patched)
- ChArUco 기반 스테레오 캡처(좌/우 자동 판별, 자동/수동 저장)
- 캡처 종료 후 스테레오 보정(내부 K,D 및 외부 R,t,E,F) 수행하여 .npz로 저장

패치 요약:
1) 커버리지 계산 개선: 호모그래피 기반 보드 외곽 폴리곤 면적 사용(실패 시 안전 폴백)
2) CharucoDetector 반환 시그니처 호환 처리 + DetectorParameters 소폭 튜닝
3) 최소 변경 원칙: 기존 로직/CLI 대부분 그대로 유지
"""

import os
import sys
import time
import glob
import json
import math
import argparse
import numpy as np
from collections import deque, Counter

import cv2

# ------------------------------- Utils & Config -------------------------------

def disable_opencl():
    """OpenCL 비활성화(환경에 따라 UMat 경로로 빠지며 속도/호환 이슈 방지)."""
    os.environ["OPENCV_OPENCL_DEVICE"] = "disabled"
    os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"
    os.environ["OPENCV_OPENCL"] = "disabled"
    try:
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass


ARUCO_DICT_NAME_TO_ENUM = {
    "4x4_50": cv2.aruco.DICT_4X4_50,
    "4x4_100": cv2.aruco.DICT_4X4_100,
    "4x4_250": cv2.aruco.DICT_4X4_250,
    "4x4_1000": cv2.aruco.DICT_4X4_1000,
    "5x5_50": cv2.aruco.DICT_5X5_50,
    "5x5_100": cv2.aruco.DICT_5X5_100,
    "5x5_250": cv2.aruco.DICT_5X5_250,
    "5x5_1000": cv2.aruco.DICT_5X5_1000,
    "6x6_50": cv2.aruco.DICT_6X6_50,
    "6x6_100": cv2.aruco.DICT_6X6_100,
    "6x6_250": cv2.aruco.DICT_6X6_250,
    "6x6_1000": cv2.aruco.DICT_6X6_1000,
    "7x7_50": cv2.aruco.DICT_7X7_50,
    "7x7_100": cv2.aruco.DICT_7X7_100,
    "7x7_250": cv2.aruco.DICT_7X7_250,
    "7x7_1000": cv2.aruco.DICT_7X7_1000,
    "apriltag_36h11": cv2.aruco.DICT_APRILTAG_36h11,  # 참고용
}

def aruco_dictionary_from_name(name: str):
    if name not in ARUCO_DICT_NAME_TO_ENUM:
        raise ValueError(f"--charuco-dict '{name}' not supported.")
    return cv2.aruco.getPredefinedDictionary(ARUCO_DICT_NAME_TO_ENUM[name])

def make_charuco_board(sx, sy, square_len, marker_len, aruco_dict):
    board = None
    if hasattr(cv2.aruco, "CharucoBoard_create"):
        # OpenCV 구버전 스타일
        board = cv2.aruco.CharucoBoard_create(sx, sy, square_len, marker_len, aruco_dict)
    else:
        CharucoBoard = getattr(cv2.aruco, "CharucoBoard", None)
        if CharucoBoard is not None and hasattr(CharucoBoard, "create"):
            # OpenCV 신버전 스타일
            board = CharucoBoard.create(sx, sy, square_len, marker_len, aruco_dict)
        elif CharucoBoard is not None:
            # 혹시 더 구버전이면 생성자 직접 호출
            board = CharucoBoard((sx, sy), square_len, marker_len, aruco_dict)
    if board is None:
        raise RuntimeError("이 OpenCV 빌드에는 CharucoBoard API가 없습니다.")
    return board

def open_cam(idx, width, height, fps, fourcc_str, buffersize=2, backend="auto"):
    # --- Windows 백엔드 선택 ---
    api_pref = 0
    if backend == "dshow":
        api_pref = cv2.CAP_DSHOW
    elif backend == "msmf":
        api_pref = cv2.CAP_MSMF

    cap = cv2.VideoCapture(idx, api_pref) if api_pref != 0 else cv2.VideoCapture(idx)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera index {idx} (backend={backend})")

    # FOURCC
    if fourcc_str:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc_str))

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    if fps is not None:
        cap.set(cv2.CAP_PROP_FPS, int(fps))

    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, buffersize)
    except Exception:
        pass
    return cap

def reopen_if_needed(cap, idx, width, height, fps, fourcc_str):
    if cap is None or not cap.isOpened():
        try:
            return open_cam(idx, width, height, fps, fourcc_str)
        except Exception:
            return None
    return cap

def next_index(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    lefts = sorted(glob.glob(os.path.join(save_dir, "left_*.png")))
    if not lefts:
        return 0
    # 파일명에서 정수 인덱스 추출
    import re
    mx = -1
    for p in lefts:
        m = re.search(r"left_(\d+)\.png$", p)
        if m:
            mx = max(mx, int(m.group(1)))
    return mx + 1 if mx >= 0 else 0

# ------------------------------- ChArUco Detect -------------------------------

def detect_charuco(gray, aruco_dict, board, detector_params=None, refine=True):
    """
    Try new API first (CharucoDetector/ArucoDetector), then fall back to
    detectMarkers + interpolateCornersCharuco. Return:
      (ok, charuco_corners, charuco_ids, (raw_corners, raw_ids, rejected))

    [PATCH] DetectorParameters 기본값 약간 튜닝, CharucoDetector 반환 시그니처 호환 처리.
    """
    ar = cv2.aruco

    # --- DetectorParameters 준비(+ 소폭 튜닝) ---
    if detector_params is None:
        detector_params = getattr(ar, "DetectorParameters", lambda: None)()
        try:
            # 안전 범위 내에서 소폭 완화 (작게 보일 때 검출력 향상)
            if hasattr(detector_params, "cornerRefinementMethod") and hasattr(ar, "CORNER_REFINE_SUBPIX"):
                detector_params.cornerRefinementMethod = ar.CORNER_REFINE_SUBPIX
            if hasattr(detector_params, "minMarkerPerimeterRate"):
                detector_params.minMarkerPerimeterRate = 0.02  # default≈0.03
            if hasattr(detector_params, "adaptiveThreshConstant"):
                detector_params.adaptiveThreshConstant = 7
        except Exception:
            pass

    corners_raw = ids_raw = rejected = None
    cc = ci = None
    ok = False

    # 1) New: CharucoDetector (버전별 반환 시그니처 호환)
    ChDet = getattr(ar, "CharucoDetector", None)
    if ChDet is not None:
        try:
            det = ChDet(board)  # 기본 파라미터 사용
            out = det.detectBoard(gray)
            # 가능한 형태들 처리:
            # (cc, ci, mC, mI) or (ret, cc, ci) or (cc, ci)
            ret_flag = True
            if isinstance(out, tuple):
                if len(out) >= 4:
                    _cc, _ci = out[0], out[1]
                elif len(out) == 3:
                    ret_flag, _cc, _ci = out
                elif len(out) == 2:
                    _cc, _ci = out
                else:
                    _cc = _ci = None
            else:
                _cc = _ci = None

            if _cc is not None and _ci is not None:
                cc = np.asarray(_cc, dtype=np.float32)
                ci = np.asarray(_ci, dtype=np.int32)
                if len(cc) >= 6:  # 최소 코너수
                    return True, cc, ci, (None, None, None)
        except Exception:
            # CharucoDetector 실패 시 폴백
            pass

    # 2) Fallback: ArucoDetector or detectMarkers
    ArDet = getattr(ar, "ArucoDetector", None)
    if ArDet is not None:
        try:
            corners_raw, ids_raw, rejected = ArDet(aruco_dict, detector_params).detectMarkers(gray)
        except Exception:
            corners_raw, ids_raw, rejected = ar.detectMarkers(gray, aruco_dict)
    else:
        corners_raw, ids_raw, rejected = ar.detectMarkers(gray, aruco_dict)

    if ids_raw is None or len(ids_raw) == 0:
        return False, None, None, (corners_raw, ids_raw, rejected)

    if refine and hasattr(ar, "refineDetectedMarkers"):
        try:
            rej = [] if rejected is None else rejected
            ar.refineDetectedMarkers(gray, board, corners_raw, ids_raw, rej)
            rejected = rej
        except Exception:
            pass

    try:
        ret, cc, ci = ar.interpolateCornersCharuco(
            markerCorners=corners_raw, markerIds=ids_raw, image=gray, board=board
        )
        if ret and cc is not None and len(cc) >= 6:
            ok = True
    except Exception:
        ok = False

    return ok, (None if cc is None else cc.astype(np.float32)), \
           (None if ci is None else ci.astype(np.int32)), \
           (corners_raw, ids_raw, rejected)

def coverage_percent(charuco_corners, w, h):
    """
    (원본) 코너 바운딩 박스 면적 / 전체 프레임 면적(%)
    ※ 내부 코너 기반이라 실제 보드 면적보다 작게 추정됨.
    """
    if charuco_corners is None or len(charuco_corners) == 0:
        return 0.0
    pts = charuco_corners.reshape(-1, 2)
    x_min, y_min = pts[:,0].min(), pts[:,1].min()
    x_max, y_max = pts[:,0].max(), pts[:,1].max()
    box_area = max(0.0, (x_max - x_min) * (y_max - y_min))
    full = float(w) * float(h)
    return (box_area / full) * 100.0 if full > 0 else 0.0

# ------------------------- [PATCH] Coverage Helpers ---------------------------

def coverage_percent_charuco_board(cc, w, h, sx, sy):
    """
    내부 코너 바운딩박스 기반 값에 보드 외곽 한 칸(half-cell)을 보정하는 스케일 계수 적용.
    scale = (sx/(sx-1)) * (sy/(sy-1))
    """
    raw = coverage_percent(cc, w, h)
    if sx <= 1 or sy <= 1:
        return raw
    scale = (sx / (sx - 1.0)) * (sy / (sy - 1.0))
    return min(100.0, raw * scale)

def coverage_via_minrect(cc, w, h):
    """
    내부 코너의 회전 바운딩박스(minAreaRect) 면적으로 커버리지 계산(축정렬보다 과소추정 완화).
    """
    if cc is None or len(cc) == 0:
        return 0.0
    pts = cc.reshape(-1,2).astype(np.float32)
    rect = cv2.minAreaRect(pts)
    box  = cv2.boxPoints(rect)
    area = cv2.contourArea(box)
    return (area / float(w*h)) * 100.0

def coverage_via_homography(cc, ci, board, w, h, squares_x, squares_y, square_len):
    """
    Charuco 내부 코너 ↔ 보드 평면좌표로 호모그래피 추정 후,
    '보드 외곽 사각형'을 화면으로 투영해 면적으로 커버리지 계산.
    반환: (coverage_percent, projected_polygon[4x2] or None)
    """
    if cc is None or ci is None or len(ci) < 4:
        return 0.0, None
    # 1) 보드 내부 코너의 2D 보드좌표(미터)
    obj3d = board_corner_3d_from_ids(board, ci)  # (N,3)
    if obj3d is None or len(obj3d) != len(ci):
        return 0.0, None
    obj2d = obj3d[:, :2].astype(np.float32)      # (N,2)
    img2d = cc.reshape(-1, 2).astype(np.float32) # (N,2)

    H, mask = cv2.findHomography(obj2d, img2d, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if H is None:
        return 0.0, None

    # 2) 보드 외곽 4점(내부 그리드 x:[0..(sx-1)], y:[0..(sy-1)]에 half-cell 확장)
    # 2) 보드 외곽 4점(내부 코너 min/max에서 '정확히 한 칸' 확장)
    x_min_obj = float(obj2d[:, 0].min())
    x_max_obj = float(obj2d[:, 0].max())
    y_min_obj = float(obj2d[:, 1].min())
    y_max_obj = float(obj2d[:, 1].max())
    s = float(square_len)
    left, right  = x_min_obj - s, x_max_obj + s
    top,  bottom = y_min_obj - s, y_max_obj + s

    board_poly = np.array([
        [left,  top,    1.0],
        [right, top,    1.0],
        [right, bottom, 1.0],
        [left,  bottom, 1.0],
    ], dtype=np.float32).T  # 3x4


    proj = (H @ board_poly)
    proj = (proj[:2] / proj[2]).T.astype(np.float32)  # 4x2
    area = cv2.contourArea(proj)
    cov = (area / float(w*h)) * 100.0
    return float(max(0.0, min(100.0, cov))), proj

def coverage_estimate(cc, ci, board, w, h, sx, sy, square_len):
    """
    [권장] 커버리지 추정 종합 함수:
      1) 호모그래피 기반 외곽 폴리곤 면적
      2) 실패 시 minAreaRect
      3) 그래도 애매하면 내부코너 바운딩박스 × 보정계수
    """
    cov, poly = coverage_via_homography(cc, ci, board, w, h, sx, sy, square_len)
    if cov > 0.0 and poly is not None:
        return cov, poly
    # fallback #1
    cov_mr = coverage_via_minrect(cc, w, h)
    if cov_mr > 0.0:
        # 내부코너 기반이므로 보정계수 곱해 근사
        adj = coverage_percent_charuco_board(cc, w, h, sx, sy)
        # minrect, adj 중 더 큰 값을 채택(너무 보수적 추정 방지)
        return float(min(100.0, max(cov_mr, adj))), None
    # fallback #2
    return coverage_percent_charuco_board(cc, w, h, sx, sy), None

# ------------------------------- L/R Auto Decide ------------------------------

def lr_sign_from_pair(cL, cR):
    """
    좌/우 프레임에서 검출된 코너들의 평균 x 좌표 차를 이용한 부호 판단.
    sign = mean_x_L - mean_x_R
    음수면 SWAP 필요하다고 가정(기존 로직과 호환)
    """
    xL = cL.reshape(-1, 2)[:,0].mean()
    xR = cR.reshape(-1, 2)[:,0].mean()
    return xL - xR

def decide_swap_majority(samples, thresh=0.0):
    """
    여러 프레임에서 측정된 sign들의 다수결로 SWAP 여부 결정.
    sign < thresh → SWAP 필요(True)
    """
    if not samples:
        return False, 0
    votes = [("swap" if s < thresh else "keep") for s in samples]
    cnt = Counter(votes)
    swap = cnt["swap"] > cnt["keep"]
    confidence = abs(cnt["swap"] - cnt["keep"])
    return swap, confidence

# ------------------------- Geometry helpers (restored) ------------------------
def centroid_of_corners(charuco_corners):
    """Return (cx, cy) of detected Charuco corners."""
    pts = charuco_corners.reshape(-1, 2)
    c = pts.mean(axis=0)
    return float(c[0]), float(c[1])

def angle_deg_fitline(charuco_corners):
    """
    Estimate dominant orientation via cv2.fitLine.
    Return angle (deg) in [-90, 90].
    """
    pts = charuco_corners.reshape(-1, 2).astype(np.float32)
    if len(pts) < 2:
        return 0.0
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    ang = math.degrees(math.atan2(float(vy), float(vx)))
    if ang > 90:
        ang -= 180
    if ang < -90:
        ang = 180
    return ang

# ------------------------------- Drawing / HUD --------------------------------

def draw_charuco_overlay(img, charuco_corners, color=(0,255,0)):
    if charuco_corners is None:
        return img
    pts = charuco_corners.reshape(-1, 2).astype(int)
    for p in pts:
        cv2.circle(img, (int(p[0]), int(p[1])), 3, color, -1)
    return img

def put_hud(panel, text_lines, org=(10, 25)):
    x, y = org
    for i, line in enumerate(text_lines):
        cv2.putText(panel, line, (x, y + i*22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

# ------------------------------- Save / IO ------------------------------------

def save_pair(save_dir, idx, frameL, frameR):
    os.makedirs(save_dir, exist_ok=True)
    left_path  = os.path.join(save_dir, f"left_{idx:04d}.png")
    right_path = os.path.join(save_dir, f"right_{idx:04d}.png")
    ok1 = cv2.imwrite(left_path, frameL)
    ok2 = cv2.imwrite(right_path, frameR)
    return ok1 and ok2, left_path, right_path

# ------------------------------- Calibration ----------------------------------

def board_corner_3d_from_ids(board, charuco_ids):
    """
    Charuco Board의 각 코너 id에 대응하는 3D 점을 추출.
    OpenCV Python 바인딩: board.chessboardCorners 또는 getChessboardCorners() 사용.
    """
    try:
        # OpenCV 버전에 따라 속성/메서드가 다를 수 있음
        corners3d = np.array(board.chessboardCorners, dtype=np.float32)
    except AttributeError:
        corners3d = np.array(board.getChessboardCorners(), dtype=np.float32)
    ids = charuco_ids.reshape(-1)
    return corners3d[ids, :]  # (N,3)

def calibrate_mono_charuco(image_paths, aruco_dict, board):
    """
    각 이미지에서 charucoCorners/Ids를 추출하여 calibrateCameraCharuco로 내부파라미터 추정.
    반환: K, D, rvecs, tvecs, all_charuco_corners(list), all_charuco_ids(list), used_count
    """
    all_corners = []
    all_ids = []
    imsize = None
    used = 0
    for p in image_paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if imsize is None:
            imsize = (img.shape[1], img.shape[0])

        ok, cc, ci, _ = detect_charuco(img, aruco_dict, board)
        if ok:
            all_corners.append(cc)
            all_ids.append(ci)
            used += 1

    if used < 3:
        raise RuntimeError("Not enough charuco detections for mono calibration.")

    # calibrateCameraCharuco expects lists of corners/ids per image
    flags = 0
    ret, K, D, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        charucoCorners=all_corners,
        charucoIds=all_ids,
        board=board,
        imageSize=imsize,
        cameraMatrix=None,
        distCoeffs=None,
        flags=flags
    )
    return ret, K, D, rvecs, tvecs, all_corners, all_ids, imsize, used

def build_stereo_correspondences(pair_paths, aruco_dict, board):
    """
    스테레오 보정용 対応점 구성.
    - 각 pair(left_i, right_i)에서 charuco 디텍션
    - 좌/우 공통 ID 교집합만 취득
    - objectPoints(해당 ID의 3D 점), imagePoints1/2(2D)
    반환: objpoints(list of Nx3), imgpoints1(list Nx2), imgpoints2(list Nx2), image_size
    """
    objpoints = []
    imgpoints1 = []
    imgpoints2 = []
    imsize = None

    used = 0
    dropped = 0

    for left_path, right_path in pair_paths:
        imgL = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
        imgR = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
        if imgL is None or imgR is None:
            dropped += 1
            continue

        if imsize is None:
            imsize = (imgL.shape[1], imgL.shape[0])

        okL, cL, idL, _ = detect_charuco(imgL, aruco_dict, board)
        okR, cR, idR, _ = detect_charuco(imgR, aruco_dict, board)
        if not (okL and okR):
            dropped += 1
            continue

        # 좌/우 공통 ID로 정렬
        idsL = idL.reshape(-1)
        idsR = idR.reshape(-1)
        setL = set(int(x) for x in idsL)
        setR = set(int(x) for x in idsR)
        common = sorted(list(setL.intersection(setR)))
        if len(common) < 6:
            # 대응점이 너무 적으면 패스
            dropped += 1
            continue

        # 인덱스 매칭
        idxL = [np.where(idsL == cid)[0][0] for cid in common]
        idxR = [np.where(idsR == cid)[0][0] for cid in common]

        ptsL = cL.reshape(-1, 2)[idxL]  # (Nc,2)
        ptsR = cR.reshape(-1, 2)[idxR]  # (Nc,2)
        obj  = board_corner_3d_from_ids(board, np.array(common, dtype=np.int32))  # (Nc,3)

        objpoints.append(obj.astype(np.float32))
        imgpoints1.append(ptsL.astype(np.float32))
        imgpoints2.append(ptsR.astype(np.float32))
        used += 1

    return objpoints, imgpoints1, imgpoints2, imsize, used, dropped

def put_lr_badge(img, text):
    cv2.putText(img, text, (12, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 3, cv2.LINE_AA)


def stereo_calibrate_charuco(save_dir, aruco_dict, board, calib_out, args=None):
    """
    저장된 pair 이미지로부터
      1) 좌/우 개별 Charuco monocular 보정
      2) 공통 ID 대응점 기반 stereoCalibrate(R,t,E,F)
      3) 결과 .npz 저장 (보드/딕셔너리 메타는 안전 getter + CLI 값으로 기록)
    """
    left_paths  = sorted(glob.glob(os.path.join(save_dir, "left_*.png")))
    right_paths = sorted(glob.glob(os.path.join(save_dir, "right_*.png")))
    if len(left_paths) == 0 or len(left_paths) != len(right_paths):
        raise RuntimeError("Saved pairs are insufficient or mismatched.")

    # --------- Monocular calibration (Charuco) ----------
    mono_err1, K1, D1, r1, t1, all_c1, all_i1, imsize1, mono_used1 = calibrate_mono_charuco(left_paths,  aruco_dict, board)
    mono_err2, K2, D2, r2, t2, all_c2, all_i2, imsize2, mono_used2 = calibrate_mono_charuco(right_paths, aruco_dict, board)

    if imsize1 != imsize2:
        raise RuntimeError(f"Left/Right image sizes differ: {imsize1} vs {imsize2}")

    # --------- Stereo correspondences (ID 교집합) ----------
    pairs = list(zip(left_paths, right_paths))
    objpoints, imgpoints1, imgpoints2, imsize, used_pairs, dropped_pairs = build_stereo_correspondences(
        pairs, aruco_dict, board
    )
    if used_pairs < 3:
        raise RuntimeError(f"Not enough valid stereo pairs after ID matching (used={used_pairs}).")

    # --------- Stereo calibration ----------
    flags = cv2.CALIB_FIX_INTRINSIC  # 모노 K,D를 고정하고 R,t,E,F 추정
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 200, 1e-9)
    stereo_err, K1_out, D1_out, K2_out, D2_out, R, T, E, F = cv2.stereoCalibrate(
        objectPoints=objpoints,
        imagePoints1=imgpoints1,
        imagePoints2=imgpoints2,
        cameraMatrix1=K1,
        distCoeffs1=D1,
        cameraMatrix2=K2,
        distCoeffs2=D2,
        imageSize=imsize,
        flags=flags,
        criteria=criteria
    )

    # --------- R 정규직교화(수치 오차 보정) ----------
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R[:, -1] *= -1

    # --------- 메타 안전 수집 (빌드 차이 대응) ----------
    def _get_board_size(b, fb):
        try:
            return tuple(b.getChessboardSize())
        except Exception:
            return fb

    def _get_square_len(b, fb):
        try:
            return float(b.getSquareLength())
        except Exception:
            return fb

    def _get_marker_len(b, fb):
        try:
            return float(b.getMarkerLength())
        except Exception:
            return fb

    sx = args.squares_x if args and hasattr(args, "squares_x") else 0
    sy = args.squares_y if args and hasattr(args, "squares_y") else 0
    sq = args.square_length if args and hasattr(args, "square_length") else 0.0
    mk = args.marker_length if args and hasattr(args, "marker_length") else 0.0
    dict_name = args.charuco_dict if args and hasattr(args, "charuco_dict") else "unknown"

    b_size = _get_board_size(board, (sx, sy))
    b_sq   = _get_square_len(board, sq)
    b_mk   = _get_marker_len(board, mk)

    meta = {
        "board": {
            "squaresX": int(b_size[0]),
            "squaresY": int(b_size[1]),
            "squareLength": float(b_sq),
            "markerLength": float(b_mk),
            "dict_name": str(dict_name)
        },
        "imsize": imsize,
        "mono_reproj_err_L": float(mono_err1),
        "mono_reproj_err_R": float(mono_err2),
        "stereo_reproj_err": float(stereo_err),
        "used_pairs": int(used_pairs),
        "dropped_pairs": int(dropped_pairs)
    }

    # --------- 저장 ----------
    out_path = os.path.join(save_dir, calib_out)
    np.savez_compressed(
        out_path,
        K1=K1_out, D1=D1_out, K2=K2_out, D2=D2_out, R=R, t=T, E=E, F=F,
        meta=json.dumps(meta)
    )

    # --------- 요약 ----------
    print("=== Stereo Calibration Summary ===")
    print(f"Image size          : {imsize}")
    print(f"Mono (L) reproj err : {mono_err1:.4f} (used {mono_used1})")
    print(f"Mono (R) reproj err : {mono_err2:.4f} (used {mono_used2})")
    print(f"Stereo reproj err   : {stereo_err:.6f}")
    print(f"Used stereo pairs   : {used_pairs}, dropped: {dropped_pairs}")
    print(f"Saved               : {out_path}")

    return out_path, {
        "mono_err_L": mono_err1,
        "mono_err_R": mono_err2,
        "stereo_err": stereo_err,
        "used_pairs": used_pairs,
        "dropped_pairs": dropped_pairs
    }

# ------------------------------- Main Capture Loop -----------------------------

def main():
    disable_opencl()

    ap = argparse.ArgumentParser(description="ChArUco-based stereo capture + auto L/R + auto-save + stereo calibration")
    # 카메라 I/O
    ap.add_argument("--left-id", type=int, default=0)
    ap.add_argument("--right-id", type=int, default=1)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--fourcc", type=str, default="MJPG")  # MJPG|YUYV|...

    # 보드/딕셔너리
    ap.add_argument("--charuco-dict", type=str, default="5x5_1000", choices=list(ARUCO_DICT_NAME_TO_ENUM.keys()))
    ap.add_argument("--squares-x", type=int, default=7, help="number of chessboard squares in X (columns)")
    ap.add_argument("--squares-y", type=int, default=5, help="number of chessboard squares in Y (rows)")
    ap.add_argument("--square-length", type=float, default=0.024, help="square length in meters")
    ap.add_argument("--marker-length", type=float, default=0.018, help="marker length in meters")

    # 자동 저장/품질
    ap.add_argument("--min-interval", type=float, default=0.6)
    ap.add_argument("--coverage-min", type=float, default=1.0)
    ap.add_argument("--coverage-max", type=float, default=99.0)
    ap.add_argument("--centroid-thresh", type=float, default=10.0)
    ap.add_argument("--angle-thresh", type=float, default=4.0)
    ap.add_argument("--aggressive-auto", action="store_true", help="포즈 변화 조건 무시(간격+커버리지만)")

    ap.add_argument("--pairs", type=int, default=0, help="N>0 이면 해당 쌍 수 저장 후 자동 종료")

    # 보정/출력
    ap.add_argument("--save-dir", type=str, default="./stereo_pairs")
    ap.add_argument("--calibrate-now", action="store_true", help="종료 시 자동 보정 수행")
    ap.add_argument("--calib-out", type=str, default="calib_charuco_stereo.npz")
    ap.add_argument("--dry-run", action="store_true", help="검출/표시만(저장/보정 안함)")

    # L/R 자동 판별 개선
    ap.add_argument("--lr-sample-n", type=int, default=10, help="초기 L/R 판별용 샘플 프레임 수")

    # 성능/견고성
    ap.add_argument("--threaded-capture", action="store_true", help="(옵션) 추후 구현용 placeholder")

    ap.add_argument("--min-charuco", type=int, default=10, help="minimum charuco corners per view to accept a frame")

    ap.add_argument("--backend", type=str, default="auto",
                    choices=["auto","dshow","msmf"],
                    help="Windows video backend: auto, dshow(DirectShow), msfm(Media Foundation)")
    
    args = ap.parse_args()

    # --- 준비 ---
    aruco_dict = aruco_dictionary_from_name(args.charuco_dict)
    board = make_charuco_board(args.squares_x, args.squares_y, args.square_length, args.marker_length, aruco_dict)

    capL = open_cam(args.left_id, args.width, args.height, args.fps, args.fourcc, backend=args.backend)
    capR = open_cam(args.right_id, args.width, args.height, args.fps, args.fourcc, backend=args.backend)

    # 웜업
    t0 = time.time()
    while time.time() - t0 < 0.8:
        capL.read()
        capR.read()

    # 인덱스
    idx = next_index(args.save_dir)

    # 상태
    armed = False
    auto_on = True
    swap_frames = False
    last_save_t = 0.0
    last_centroid = None
    last_angle = None
    saved_pairs = 0

    # L/R 판별 샘플링
    lr_samples = []

    # UI
    win = "ChArUco Stereo Capture"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    # 메인 루프
    while True:
        retL, frameL = capL.read()
        retR, frameR = capR.read()

        if not retL:
            capL = reopen_if_needed(capL, args.left_id, args.width, args.height, args.fps, args.fourcc)
            continue
        if not retR:
            capR = reopen_if_needed(capR, args.right_id, args.width, args.height, args.fps, args.fourcc)
            continue

        # L/R 스왑 결정(초기 N 프레임 다수결)
        grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

        okL, cL, idL, dbgL = detect_charuco(grayL, aruco_dict, board)
        okR, cR, idR, dbgR = detect_charuco(grayR, aruco_dict, board)

        if len(lr_samples) < args.lr_sample_n and okL and okR:
            lr_samples.append(lr_sign_from_pair(cL, cR))
            if len(lr_samples) == args.lr_sample_n:
                swap_frames, conf = decide_swap_majority(lr_samples)
                # 초기 판별 실패시(샘플이 부족) → 루프 중에도 계속 시도 가능
        # 스왑 적용
        if swap_frames:
            frameL, frameR = frameR, frameL
            grayL, grayR = grayR, grayL
            okL, cL, idL, dbgL, okR, cR, idR, dbgR = okR, cR, idR, dbgR, okL, cL, idL, dbgL

        # 디텍션 오버레이
        visL = frameL.copy()
        visR = frameR.copy()
        
        put_lr_badge(visL, f"L (K1) <- idx{args.left_id}")
        put_lr_badge(visR, f"R (K2) <- idx{args.right_id}")

        if okL:
            draw_charuco_overlay(visL, cL, (0,255,0))
        if okR:
            draw_charuco_overlay(visR, cR, (0,255,0))

        # ------------------ [PATCH] 커버리지/포즈/폴리곤 오버레이 ------------------
        h, w = grayL.shape[:2]
        if okL:
            covL, polyL = coverage_estimate(cL, idL, board, w, h, args.squares_x, args.squares_y, args.square_length)
            if polyL is not None:
                cv2.polylines(visL, [polyL.astype(int)], True, (0,255,255), 2)
        else:
            covL, polyL = 0.0, None

        if okR:
            covR, polyR = coverage_estimate(cR, idR, board, w, h, args.squares_x, args.squares_y, args.square_length)
            if polyR is not None:
                cv2.polylines(visR, [polyR.astype(int)], True, (0,255,255), 2)
        else:
            covR, polyR = 0.0, None

        covMean = (covL + covR) * 0.5
        # ---------------------------------------------------------------------

        # 커버리지 외 보조 지표(기존 로직 유지)
        angleNow = None
        centNow  = None
        if okL and okR:
            # 두 카메라 모두 성공 시 평균 기반 지표
            angleL = angle_deg_fitline(cL)
            angleR = angle_deg_fitline(cR)
            angleNow = 0.5 * (angleL + angleR)

            cxL, cyL = centroid_of_corners(cL)
            cxR, cyR = centroid_of_corners(cR)
            centNow = ((cxL + cxR) * 0.5, (cyL + cyR) * 0.5)

        # -------------------- 자동 저장 판정 (첫 저장 예외 + 안정 로그 친화) --------------------
        status_line = "HOLD"
        now = time.time()
        can_save = False

        if okL and okR:
            # 공통 게이트 플래그
            cov_ok = (args.coverage_min <= covMean <= args.coverage_max)
            dt_ok  = (now - last_save_t) >= args.min_interval

            # 코너 수 (없으면 0)
            nL = len(cL.reshape(-1, 2)) if cL is not None else 0
            nR = len(cR.reshape(-1, 2)) if cR is not None else 0

            # 포즈 게이트:
            #  - aggressive-auto면 항상 True
            #  - "첫 저장"이거나 기준이 없으면(True) 첫 저장 통과
            #  - 그 외에는 centroid/angle 임계값 검사
            pose_ok_flag = True
            if not args.aggressive_auto:
                if saved_pairs == 0 or last_centroid is None or last_angle is None:
                    pose_ok_flag = True
                else:
                    d_cent = 0.0
                    d_ang  = 0.0
                    if centNow is not None and last_centroid is not None:
                        d_cent = math.hypot(centNow[0] - last_centroid[0],
                                            centNow[1] - last_centroid[1])
                    if angleNow is not None and last_angle is not None:
                        d_ang = abs(angleNow - last_angle)
                        if d_ang > 90:  # wrap
                            d_ang = 180 - d_ang
                    pose_ok_flag = (d_cent >= args.centroid_thresh) or (d_ang >= args.angle_thresh)

            # 최종 판정
            if cov_ok and dt_ok and pose_ok_flag:
                if nL < args.min_charuco or nR < args.min_charuco:
                    status_line = f"HOLD: few charuco (L={nL}, R={nR})"
                    can_save = False
                else:
                    status_line = f"OK dt={now-last_save_t:.2f}s cov={covMean:.1f}%"
                    if not args.dry_run and armed and auto_on:
                        can_save = True
            else:
                reasons = []
                if not cov_ok:      reasons.append("coverage")
                if not dt_ok:       reasons.append("interval")
                if not pose_ok_flag and not args.aggressive_auto:
                    reasons.append("pose")
                status_line = "HOLD: " + ", ".join(reasons) if reasons else "HOLD"
        else:
            miss = []
            if not okL: miss.append("L no charuco")
            if not okR: miss.append("R no charuco")
            status_line = "HOLD: " + ", ".join(miss)
        # --------------------------------------------------------------------


        # 수동 저장 키 체크는 아래 키 이벤트에서 처리

        # 패널 합성
        panel = np.hstack([visL, visR])

        # HUD
        text = [
            f"ARMED={armed}  AUTO={auto_on}  SWAP={swap_frames}",
            f"coverage L={covL:.1f}% R={covR:.1f}% mean={covMean:.1f}%",
            f"angle={angleNow:.1f} deg" if angleNow is not None else "angle=NA",
            f"pairs={saved_pairs}" + (f"/{args.pairs}" if args.pairs > 0 else ""),
            status_line
        ]
        put_hud(panel, text, (10,25))

        cv2.imshow(win, panel)
        key = cv2.waitKey(1) & 0xFF

        # 키 처리
        if key in (27, ord('q')):  # ESC, q
            break
        elif key == 32:  # Space → 3초 카운트다운 후 ARMED
            # HUD 카운트다운
            for t in range(3, 0, -1):
                tmp = panel.copy()
                put_hud(tmp, [f"ARMING... {t}s"], (10, panel.shape[0]-30))
                cv2.imshow(win, tmp)
                cv2.waitKey(250)
                cv2.waitKey(250)
                cv2.waitKey(250)
                cv2.waitKey(250)
            armed = True
            last_save_t = time.time()
        elif key == ord('a'):
            auto_on = not auto_on
        elif key in (ord('s'), ord('f')):  # 수동 저장(armed + 양쪽 검출 성공)
            if not args.dry_run and armed and okL and okR:
                ok, lp, rp = save_pair(args.save_dir, idx, frameL, frameR)
                if ok:
                    saved_pairs += 1
                    idx += 1
                    last_save_t = time.time()
                    last_centroid = centNow if centNow is not None else last_centroid
                    last_angle = angleNow if angleNow is not None else last_angle
        # 자동 저장 실행
        if can_save:
            ok, lp, rp = save_pair(args.save_dir, idx, frameL, frameR)
            if ok:
                saved_pairs += 1
                idx += 1
                last_save_t = now
                last_centroid = centNow if centNow is not None else last_centroid
                last_angle = angleNow if angleNow is not None else last_angle

        # 목표 쌍 도달 시 자동 종료
        if args.pairs > 0 and saved_pairs >= args.pairs:
            break

    # 종료
    capL.release()
    capR.release()
    cv2.destroyAllWindows()

    # 캘리브레이션
    if args.calibrate_now and not args.dry_run:
        try:
            out_path, summ = stereo_calibrate_charuco(
                        save_dir=args.save_dir,
                        aruco_dict=aruco_dict,
                        board=board,
                        calib_out=args.calib_out,
                        args=args   # ← pass full args for meta
                    )
        except Exception as e:
            print(f"[Calibration] Failed: {e}")

if __name__ == "__main__":
    main()
