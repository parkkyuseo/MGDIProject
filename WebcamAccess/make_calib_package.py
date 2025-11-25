# make_calib_package.py
# 한 줄 요약:
#   T_HC.npz -> (선택: 입력 프레임 정렬) -> calib/T_HC.json + summary.txt + test_apply_thc.py
#
# 옵션:
#   --thc <npz>            : 입력 npz (키: R(3x3), t(3,))
#   --outdir <dir>         : 출력 디렉토리(기본 calib)
#   --align <none|z|zx>    : 정렬 모드. none(기본), z, zx   [※ 입력(웹캠) 프레임 우측곱 정렬]
#   --input-conv <opencv|unity>
#                          : 입력 X_C의 좌표계 관례. opencv(기본, x-right,y-down,z-forward) /
#                            unity(x-right,y-up,z-forward). opencv면 런타임에서 S·X_C_cv 선적용 필요
#   --anchor-id <str>      : 앵커 GUID(또는 식별자). ""이면 JSON에 null 저장
#   --note <str>           : 메모(자유 텍스트)
#
# 결과:
#   outdir/T_HC.json, outdir/summary.txt, outdir/test_apply_thc.py
#
# 수식(요약):
#   기본:        X_H = R * X_C + t
#   input=opencv: X_H = R * (S * X_C_cv) + t,  S = diag(1,-1,1)
#
# 정렬(align=z/zx)은 입력 프레임에 대한 우측곱 A를 구성하여 R' = R @ A 로 반영. t는 그대로.

import argparse, json, os, datetime
import numpy as np

# ---------- 선형대수 유틸 ----------

def load_npz_rt(path):
    d = np.load(path)
    R = d["R"].reshape(3,3).astype(np.float64)
    t = d["t"].reshape(3).astype(np.float64)
    return R, t

def orthonormalize_R(R):
    U, _, Vt = np.linalg.svd(R)
    Rm = U @ Vt
    if np.linalg.det(Rm) < 1.0 - 1e-10:
        # 이론상 det=+1이어야 함. 수치 오차로 -1라면 마지막 축 반전
        U[:, -1] *= -1
        Rm = U @ Vt
    return Rm

def _skew(v):
    x, y, z = v
    return np.array([[0,-z, y],[z, 0,-x],[-y, x, 0]], dtype=np.float64)

def rot_axis_angle(axis, theta):
    axis = np.asarray(axis, dtype=np.float64)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.eye(3)
    a = axis / n
    K = _skew(a)
    c, s = np.cos(theta), np.sin(theta)
    return (c*np.eye(3)) + (1-c)*np.outer(a,a) + s*K

def rot_between(a, b):
    """a->b 로 보내는 회전행렬 (둘 다 3D, det=+1)"""
    a = np.asarray(a, dtype=np.float64); a /= (np.linalg.norm(a) + 1e-12)
    b = np.asarray(b, dtype=np.float64); b /= (np.linalg.norm(b) + 1e-12)
    c = float(np.clip(np.dot(a,b), -1.0, 1.0))
    if c > 1.0 - 1e-10:
        return np.eye(3)
    if c < -1.0 + 1e-10:
        # 반평행: a에 수직인 축을 임의로 선택해 180° 회전
        tmp = np.array([1.0,0.0,0.0]) if abs(a[0]) < 0.9 else np.array([0.0,1.0,0.0])
        axis = np.cross(a, tmp); axis /= (np.linalg.norm(axis) + 1e-12)
        return rot_axis_angle(axis, np.pi)
    v = np.cross(a,b)
    s = np.linalg.norm(v)
    K = _skew(v)
    R = np.eye(3) + K + (K@K) * ((1.0 - c) / (s*s + 1e-12))
    return orthonormalize_R(R)

# ---------- 프레임 정렬(입력 우측곱) ----------

def align_R_input_side(R, mode):
    """
    입력(웹캠) 프레임을 R의 '우측곱'으로 정렬.
    mode='none' 그대로,
    'z'  : (R @ A) ez = ez (카메라 +Z가 H +Z로 향하도록)
    'zx' : 위 이후, (R @ A) ex 의 XY평면 yaw를 0으로 (H +X 정렬)
    반환: (R_aligned, A_right)  with  R_aligned = R @ A_right,  t는 변경 없음
    """
    if mode in (None, "none"):
        return orthonormalize_R(R), np.eye(3)

    ez = np.array([0.0,0.0,1.0]); ex = np.array([1.0,0.0,0.0])

    # 1) Z 정렬: A_z s.t. (R @ A_z) ez = ez  =>  A_z: ez -> R^T ez
    A = rot_between(ez, R.T @ ez)
    R1 = R @ A

    if mode == "z":
        return orthonormalize_R(R1), A

    # 2) X 정렬: hx = (R1 ex)의 XY yaw 제거 => C측 z축 회전(=우측 Rot_z)
    hx = R1 @ ex
    hx_proj = hx.copy(); hx_proj[2] = 0.0
    if np.linalg.norm(hx_proj) > 1e-12:
        theta = np.arctan2(hx_proj[1], hx_proj[0])  # XY 평면 각
        A_x = rot_axis_angle(ez, -theta)            # 입력 z축 회전
        A = A @ A_x
        R1 = R @ A

    return orthonormalize_R(R1), A

# ---------- 쿼터니언/파일 출력 ----------

def mat_to_quat_xyzw(R):
    # Quaternion (x,y,z,w), row-major R (Unity 호환)
    m00, m01, m02 = R[0,0], R[0,1], R[0,2]
    m10, m11, m12 = R[1,0], R[1,1], R[1,2]
    m20, m21, m22 = R[2,0], R[2,1], R[2,2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    else:
        if (m00 > m11) and (m00 > m22):
            s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
            w = (m21 - m12) / s
            x = 0.25 * s
            y = (m01 + m10) / s
            z = (m02 + m20) / s
        elif m11 > m22:
            s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
            w = (m02 - m20) / s
            x = (m01 + m10) / s
            y = 0.25 * s
            z = (m12 + m21) / s
        else:
            s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
            w = (m10 - m01) / s
            x = (m02 + m20) / s
            y = (m12 + m21) / s
            z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    q /= (np.linalg.norm(q) + 1e-12)
    return q

def write_thc_json(out_json, R, t, anchor_id, note, input_conv, align_mode):
    ts = datetime.datetime.utcnow().isoformat() + "Z"
    qx, qy, qz, qw = mat_to_quat_xyzw(R)
    data = {
        "version": 2,
        "timestamp": ts,
        "anchor_id": (None if (anchor_id is None or anchor_id.strip()=="") else anchor_id),
        "input_convention": input_conv,   # "opencv" or "unity"
        "align_mode": align_mode,         # "none"|"z"|"zx"
        "R_quat": {"x": float(qx), "y": float(qy), "z": float(qz), "w": float(qw)},
        "t": {"x": float(t[0]), "y": float(t[1]), "z": float(t[2])},
        "note": ("" if note is None else str(note))
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def write_summary(out_txt, R_raw, R_final, t_final, align_mode, note, input_conv):
    lines = []
    lines.append("T_H<-C package summary")
    lines.append("======================\n")
    lines.append("[Units / Convention]")
    lines.append("- Units: meters (m)")
    if input_conv == "opencv":
        lines.append("- Input  X_C : camera (OpenCV: x-right, y-down, z-forward), column-vector")
        lines.append("- Mapping   : X_H = R * (S * X_C) + t,  S = diag(1,-1,1)")
    else:
        lines.append("- Input  X_C : camera (Unity: x-right, y-up, z-forward), column-vector")
        lines.append("- Mapping   : X_H = R * X_C + t")
    lines.append("- Output X_H : HoloLens world (anchor-fixed), column-vector\n")

    lines.append("[Matrix sanity]")
    lines.append(f"- det(R_raw)   = {np.linalg.det(R_raw): .6f}")
    lines.append(f"- det(R_final) = {np.linalg.det(R_final): .6f}  (SO(3) projected)\n")

    lines.append("[Alignment (input-side/right-multiply)]")
    lines.append(f"- align mode   : {align_mode}")
    lines.append("- z  : make (R@A) * +Z_C  ->  +Z_H")
    lines.append("- zx : then yaw within XY so (R@A) * +X_C -> +X_H\n")

    lines.append("[Translation t (m)]")
    lines.append(f"- t = [{t_final[0]: .6f}, {t_final[1]: .6f}, {t_final[2]: .6f}]\n")

    if note:
        lines.append(f"[Note] {note}\n")

    lines.append("[Quick test]")
    lines.append("python ./calib/test_apply_thc.py\n")

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def write_test_py(out_path):
    code = r'''# ./calib/test_apply_thc.py
import json, numpy as np, os

HERE = os.path.dirname(__file__)
JPATH = os.path.join(HERE, "T_HC.json")

def quat_to_R(qx,qy,qz,qw):
    x,y,z,w = qx,qy,qz,qw
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    R = np.array([
        [1-2*(yy+zz),   2*(xy-wz),     2*(xz+wy)],
        [2*(xy+wz),     1-2*(xx+zz),   2*(yz-wx)],
        [2*(xz-wy),     2*(yz+wx),     1-2*(xx+yy)]
    ], dtype=np.float64)
    return R

S_CV2UNITY = np.diag([1.0, -1.0, 1.0])  # y-down -> y-up

def apply(R, t, Xc, input_conv):
    Xc = np.asarray(Xc, dtype=np.float64).reshape(3,)
    if input_conv == "opencv":
        Xc = S_CV2UNITY @ Xc
    return (R @ Xc) + t

def main():
    with open(JPATH, "r", encoding="utf-8") as f:
        J = json.load(f)
    q = J["R_quat"]
    t = np.array([J["t"]["x"], J["t"]["y"], J["t"]["z"]], dtype=np.float64)
    R = quat_to_R(q["x"], q["y"], q["z"], q["w"])
    input_conv = J.get("input_convention", "opencv")

    samples = [
        ("wrist_demo",     [0.00, 0.00, 0.50]),
        ("palm_demo",      [0.05, 0.00, 0.55]),
        ("index_tip_demo", [0.10, -0.02, 0.60]),
    ]
    print("[Loaded] input_convention:", input_conv)
    print("[Loaded] R (3x3):\n", R)
    print("[Loaded] t:", t)

    for name, Xc in samples:
        Xh = apply(R, t, Xc, input_conv)
        print(f"{name:16s}  X_C={np.array(Xc)}  ->  X_H={Xh}")

if __name__ == "__main__":
    main()
'''
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(code)

# ---------- 메인 ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thc", required=True, help="path to stereo_pairs/T_HC.npz")
    ap.add_argument("--outdir", default="calib")
    ap.add_argument("--align", default="none", choices=["none","z","zx"], help="axis alignment mode (input-side/right-multiply)")
    ap.add_argument("--input-conv", default="opencv", choices=["opencv","unity"],
                    help="convention of input X_C (opencv: x-right,y-down,z-forward; unity: x-right,y-up,z-forward)")
    ap.add_argument("--anchor-id", default="", help="Spatial Anchor GUID or any identifier; empty -> null")
    ap.add_argument("--note", default="", help="free-form note")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    R_raw, t_raw = load_npz_rt(args.thc)

    # 기본 R 정규직교화
    R0 = orthonormalize_R(R_raw)

    # (선택) 입력측 정렬: R' = R0 @ A_right, t는 변경 없음
    R_aligned, A_right = align_R_input_side(R0, args.align)

    R_final = orthonormalize_R(R_aligned)
    t_final = t_raw.astype(np.float64)  # 입력 프레임 정렬은 우측곱이므로 t는 그대로

    # JSON
    out_json = os.path.join(args.outdir, "T_HC.json")
    write_thc_json(out_json, R_final, t_final, args.anchor_id, args.note, args.input_conv, args.align)

    # summary
    out_txt = os.path.join(args.outdir, "summary.txt")
    write_summary(out_txt, R_raw, R_final, t_final, args.align, args.note, args.input_conv)

    # test script
    out_test = os.path.join(args.outdir, "test_apply_thc.py")
    write_test_py(out_test)

    print("[OK] wrote:", out_json)
    print("[OK] wrote:", out_txt)
    print("[OK] wrote:", out_test)

if __name__ == "__main__":
    main()
