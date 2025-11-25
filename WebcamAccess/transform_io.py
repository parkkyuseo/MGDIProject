# -*- coding: utf-8 -*-
"""
transform_io.py
- 보드 포즈 기반 변환행렬 T_H<-C = T_H<-B * (T_C<-B)^-1 을 저장/불러오기
- 저장 포맷 권장: .npz  (키: R(3x3), t(3,))
- 옵션: .json { "R": [[...],[...],[...]], "t": [x,y,z] }

예시:
  save_transform("transforms/T_HC.npz", R, t)
  R2, t2 = load_transform("transforms/T_HC.npz")
  X_H = apply_transform(R2, t2, X_C)
"""

from typing import Tuple
import numpy as np
import json
import os


def save_transform(path: str, R: np.ndarray, t: np.ndarray):
    R = np.asarray(R, dtype=float)
    t = np.asarray(t, dtype=float).reshape(3,)
    if R.shape != (3, 3):
        raise ValueError("R must be 3x3")
    if t.shape != (3,):
        raise ValueError("t must be (3,)")

    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        np.savez(path, R=R, t=t)
    elif ext == ".json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"R": R.tolist(), "t": t.tolist()}, f, ensure_ascii=False, separators=(',', ':'))
    else:
        raise ValueError("지원 확장자: .npz 또는 .json")


def load_transform(path: str) -> Tuple[np.ndarray, np.ndarray]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        data = np.load(path)
        R = data["R"].astype(float)
        t = data["t"].astype(float).reshape(3,)
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        R = np.array(obj["R"], dtype=float).reshape(3, 3)
        t = np.array(obj["t"], dtype=float).reshape(3,)
    else:
        raise ValueError("지원 확장자: .npz 또는 .json")
    return R, t


def apply_transform(R: np.ndarray, t: np.ndarray, X_C: np.ndarray) -> np.ndarray:
    """
    X_H = R @ X_C + t
    """
    R = np.asarray(R, dtype=float).reshape(3, 3)
    t = np.asarray(t, dtype=float).reshape(3,)
    X_C = np.asarray(X_C, dtype=float).reshape(3,)
    return R.dot(X_C) + t
