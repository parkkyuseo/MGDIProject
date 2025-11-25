#!/usr/bin/env python3
"""
stereo_postcalib_tool.py
- 입력: calib_charuco_stereo.npz (K1,K2,D1,D2,R,t,meta 포함)
- 출력:
  1) calib_summary.json (baseline/mm, image_size, 보드/오차 요약)
  2) rectify_maps.npz    (map1x,map1y,map2x,map2y,Q,roi1,roi2)
맥/윈도우 공통 사용.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import cv2


def load_meta_safe(arr):
    if arr is None:
        return {}
    try:
        return json.loads(str(arr))
    except Exception:
        return {}


def export_summary(npz_path: Path, out_json: Path) -> dict:
    d = np.load(npz_path, allow_pickle=True)
    T = d["t"]
    baseline_mm = float(np.linalg.norm(T)) * 1000.0
    meta = load_meta_safe(d.get("meta", None))
    summary = {
        "baseline_mm": round(baseline_mm, 3),
        "image_size": meta.get("imsize", None),
        "board": meta.get("board", {}),
        "mono_reproj_err_L": meta.get("mono_reproj_err_L", None),
        "mono_reproj_err_R": meta.get("mono_reproj_err_R", None),
        "stereo_reproj_err": meta.get("stereo_reproj_err", None),
        "calib_npz": str(npz_path.resolve()),
    }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_rectify_maps(npz_path: Path, out_maps: Path, alpha: float = 0.0):
    d = np.load(npz_path, allow_pickle=True)
    K1, K2 = d["K1"], d["K2"]
    D1, D2 = d["D1"], d["D2"]
    R, T = d["R"], d["t"]
    meta = load_meta_safe(d.get("meta", None))
    size = tuple(meta["imsize"])  # (w,h)

    # stereoRectify
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1, D1, K2, D2, size, R, T, alpha=alpha
    )
    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, size, cv2.CV_32FC1)

    np.savez_compressed(
        out_maps,
        map1x=map1x, map1y=map1y,
        map2x=map2x, map2y=map2y,
        Q=Q, roi1=np.array(roi1), roi2=np.array(roi2)
    )
    return dict(Q_shape=Q.shape, roi1=roi1, roi2=roi2)


def main():
    ap = argparse.ArgumentParser(
        description="Export baseline summary JSON and build rectification maps from a stereo calib npz."
    )
    ap.add_argument("--npz", required=True, help="path to calib_charuco_stereo.npz")
    ap.add_argument("--out-json", default=None, help="output calib_summary.json (default: <npz_dir>/calib_summary.json)")
    ap.add_argument("--out-maps", default=None, help="output rectify_maps.npz (default: <npz_dir>/rectify_maps.npz)")
    ap.add_argument("--alpha", type=float, default=0.0, help="stereoRectify alpha (0..1)")
    args = ap.parse_args()

    npz_path = Path(args.npz)
    if not npz_path.exists():
        raise SystemExit(f"[ERR] npz not found: {npz_path}")

    out_json = Path(args.out_json) if args.out_json else (npz_path.parent / "calib_summary.json")
    out_maps = Path(args.out_maps) if args.out_maps else (npz_path.parent / "rectify_maps.npz")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_maps.parent.mkdir(parents=True, exist_ok=True)

    # 1) 요약/베이스라인
    summary = export_summary(npz_path, out_json)
    print(f"baseline (mm): {summary['baseline_mm']}")
    print(f"image_size   : {summary['image_size']}")
    print(f"saved JSON   : {out_json}")

    # 2) rectification 맵
    info = build_rectify_maps(npz_path, out_maps, alpha=args.alpha)
    print(f"saved maps   : {out_maps} (Q={info['Q_shape']}, roi1={tuple(info['roi1'])}, roi2={tuple(info['roi2'])})")


if __name__ == "__main__":
    main()
