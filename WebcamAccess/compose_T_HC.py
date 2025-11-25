#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compose_T_HC.py
Compute T_H<-C from T_H<-B and T_C<-B:
    T_H<-C = T_H<-B * (T_C<-B)^-1

- Loads npz files that contain R(3x3) and t(3,)
- (Optional) Right-multiply flips/rotations on the Board frame at H-side:
    --fix-board-z {none,ax,ay}  (ax=Rot_x(pi), ay=Rot_y(pi))
    --fix-board-rot {none,rz90,rz-90,rz180}
- Projects to SO(3) only if orthonormal error is significant (numerical drift)
- Prints diagnostics (det(R), ||R^T R - I||, Z·+Z, XY_score)
- Saves T_HC.npz (R,t) and optional JSON (quat+t)
"""
import argparse, json
import numpy as np
from pathlib import Path

def load_npz_rt(path: str):
    d = np.load(path)
    R = d["R"].reshape(3, 3).astype(np.float64)
    t = d["t"].reshape(3).astype(np.float64)
    return R, t

def invert_RT(R: np.ndarray, t: np.ndarray):
    R_inv = R.T
    t_inv = - R_inv @ t
    return R_inv, t_inv

def compose(R_AB, t_AB, R_BC, t_BC):
    R_AC = R_AB @ R_BC
    t_AC = t_AB + R_AB @ t_BC
    return R_AC, t_AC

def project_to_SO3(R: np.ndarray):
    U, _, Vt = np.linalg.svd(R)
    Rm = U @ Vt
    if np.linalg.det(Rm) < 0:
        U[:, -1] *= -1
        Rm = U @ Vt
    return Rm

def mat_to_quat_xyzw(R: np.ndarray):
    m00,m01,m02 = R[0,0],R[0,1],R[0,2]
    m10,m11,m12 = R[1,0],R[1,1],R[1,2]
    m20,m21,m22 = R[2,0],R[2,1],R[2,2]
    trace = m00+m11+m22
    if trace>0.0:
        s = 0.5/np.sqrt(trace+1.0)
        w = 0.25/s
        x = (m21-m12)*s; y=(m02-m20)*s; z=(m10-m01)*s
    else:
        if (m00>m11) and (m00>m22):
            s = 2.0*np.sqrt(1.0+m00-m11-m22)
            w = (m21-m12)/s; x=0.25*s; y=(m01+m10)/s; z=(m02+m20)/s
        elif m11>m22:
            s = 2.0*np.sqrt(1.0+m11-m00-m22)
            w = (m02-m20)/s; x=(m01+m10)/s; y=0.25*s; z=(m12+m21)/s
        else:
            s = 2.0*np.sqrt(1.0+m22-m00-m11)
            w = (m10-m01)/s; x=(m02+m20)/s; y=(m12+m21)/s; z=0.25*s
    q = np.array([x,y,z,w], dtype=np.float64)
    q /= (np.linalg.norm(q) + 1e-12)
    return q

def rotz90(sign=+1):
    th = np.deg2rad(90 * sign)
    c, s = np.cos(th), np.sin(th)
    return np.array([[ c,-s,0],
                     [ s, c,0],
                     [ 0, 0,1]], dtype=np.float64)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--THB", required=True, help="npz with T_H<-B (keys: R,t)")
    ap.add_argument("--TCB", required=True, help="npz with T_C<-B (keys: R,t)")
    ap.add_argument("--out", default="T_HC.npz", help="output npz path for T_H<-C")
    ap.add_argument("--json-out", default="", help="optional JSON path (quat+t)")

    # 기존 보정(우측곱 flip): none | ax | ay
    ap.add_argument("--fix-board-z", default="none", choices=["none","ax","ay"],
                    help="Right-multiply 180° flip on Board frame at H-side: ax=Rot_x(pi), ay=Rot_y(pi)")
    # 신규: Z축 회전 보정
    ap.add_argument("--fix-board-rot", default="none", choices=["none","rz90","rz-90","rz180"],
                    help="Right-multiply Z-rotation on Board frame at H-side (none|rz90|rz-90|rz180)")

    args = ap.parse_args()

    # Load & normalize inputs
    R_HB, t_HB = load_npz_rt(args.THB)
    R_CB, t_CB = load_npz_rt(args.TCB)
    R_HB = project_to_SO3(R_HB)
    R_CB = project_to_SO3(R_CB)

    # Optional H-side board-frame fixes (right-multiply)
    if args.fix_board_z != "none":
        A = np.diag([1,-1,-1]) if args.fix_board_z == "ax" else np.diag([-1,1,-1])
        R_HB = R_HB @ A

    if args.fix_board_rot != "none":
        if args.fix_board_rot == "rz90":
            A = rotz90(+1)
        elif args.fix_board_rot == "rz-90":
            A = rotz90(-1)
        else:  # rz180
            A = rotz90(+1) @ rotz90(+1)
        R_HB = R_HB @ A

    # Compose
    R_BC, t_BC = invert_RT(R_CB, t_CB)
    R_HC, t_HC = compose(R_HB, t_HB, R_BC, t_BC)

    # Diagnostics before projection
    I = np.eye(3)
    ortho_err = np.linalg.norm(R_HC.T @ R_HC - I, ord='fro')
    det_before = float(np.linalg.det(R_HC))
    ez = np.array([0.0, 0.0, 1.0])
    ex = np.array([1.0, 0.0, 0.0]); ey = np.array([0.0, 1.0, 0.0])
    zdot = float(R_HC[:,2] @ ez)
    xy_score = (abs(R_HC[:,0] @ ex) + abs(R_HC[:,1] @ ey)) - (abs(R_HC[:,0] @ ey) + abs(R_HC[:,1] @ ex))

    if det_before < 0.0:
        print("[WARN] det(R_HC)<0 before projection → possible reflection (axis flip). "
              "Check camera/HoloLens convention unification.")

    if ortho_err > 1e-6:
        R_HC = project_to_SO3(R_HC)
        print(f"[WARN] projected R_HC to SO(3) (||R^T R - I|| was {ortho_err:.2e})")
        ortho_err = np.linalg.norm(R_HC.T @ R_HC - I, ord='fro')

    # Final diagnostics
    det_final = float(np.linalg.det(R_HC))
    zdot_final = float(R_HC[:,2] @ ez)
    xy_score_final = (abs(R_HC[:,0] @ ex) + abs(R_HC[:,1] @ ey)) - (abs(R_HC[:,0] @ ey) + abs(R_HC[:,1] @ ex))

    print(f"[OK] Composed T_H<-C -> {Path(args.out).resolve()}")
    print("det(R_HC)     = ", f"{det_final: .6f}")
    print("||R^T R - I|| = ", f"{ortho_err: .3e}")
    print("Z·+Z          = ", f"{zdot_final:+.6f}")
    print("XY_score      = ", f"{xy_score_final:+.6f}")
    print("R_HC:\n", R_HC)
    print("t_HC (m):", t_HC)

    # Save npz
    np.savez(args.out, R=R_HC.astype(np.float64), t=t_HC.astype(np.float64))

    # (Optional) JSON
    if args.json_out:
        qx, qy, qz, qw = mat_to_quat_xyzw(R_HC)
        data = {
            "version": 2,
            "note": f"T_H<-C composed; fix_z={args.fix_board_z}; fix_rot={args.fix_board_rot}; rotation projected to SO(3) if needed",
            "R_quat": {"x": float(qx), "y": float(qy), "z": float(qz), "w": float(qw)},
            "t": {"x": float(t_HC[0]), "y": float(t_HC[1]), "z": float(t_HC[2])},
            "detR": det_final,
            "ortho_err": ortho_err,
            "Rz_dot_Z": zdot_final,
            "XY_score": xy_score_final,
            "input_convention": "opencv",   # Upstream expects OpenCV camera convention
            "C_ref": "left_rectified"
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[OK] Wrote JSON -> {Path(args.json_out).resolve()}")

if __name__ == "__main__":
    main()
