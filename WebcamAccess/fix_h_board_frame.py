# fix_h_board_frame.py
# Try combinations of Board-frame adjustments on H-side and pick the best by a score:
#   candidates = (Z-rot: none, +90, -90, 180) × (flip: none, Ax, Ay)
# Score = 2*max(0, Z·+Z) + XY_score, where
#   XY_score = (|Ex·ex| + |Ey·ey|) - (|Ex·ey| + |Ey·ex|)
import numpy as np, argparse

def load_rt(p):
    d = np.load(p)
    return d['R'].reshape(3,3).astype(float), d['t'].reshape(3).astype(float)

def inv_rt(R, t):
    Rt = R.T
    return Rt, -Rt @ t

def so3(R):
    U, _, Vt = np.linalg.svd(R)
    Rm = U @ Vt
    if np.linalg.det(Rm) < 0:
        U[:, -1] *= -1
        Rm = U @ Vt
    return Rm

def compose(R_AB, t_AB, R_BC, t_BC):
    return so3(R_AB @ R_BC), t_AB + R_AB @ t_BC

def rotz90(sign=+1):
    th = np.deg2rad(90 * sign); c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]], float)

Ax = np.diag([1, -1, -1])   # Rot_x(pi)
Ay = np.diag([-1, 1, -1])   # Rot_y(pi)

ap = argparse.ArgumentParser()
ap.add_argument("--THB", required=True)  # T_H<-B.npz
ap.add_argument("--TCB", required=True)  # T_C<-B.npz
ap.add_argument("--out", default="T_HC_fixed.npz")
args = ap.parse_args()

R_HB, t_HB = load_rt(args.THB)
R_CB, t_CB = load_rt(args.TCB)
R_HB = so3(R_HB); R_CB = so3(R_CB)

cands_rot = [("none", np.eye(3)), ("rz90", rotz90(+1)), ("rz-90", rotz90(-1)), ("rz180", rotz90(+1) @ rotz90(+1))]
cands_flip= [("none", np.eye(3)), ("Ax", Ax), ("Ay", Ay)]

best_score = -1e9
best_name = None
best_RHC = None
best_tHC = None

ez = np.array([0.0,0.0,1.0]); ex = np.array([1.0,0.0,0.0]); ey = np.array([0.0,1.0,0.0])

for rn, Rz in cands_rot:
    for fn, F in cands_flip:
        R_HB_fix = so3(R_HB @ Rz @ F)   # right-multiply adjustments on H-side
        R_BC, t_BC = inv_rt(R_CB, t_CB)
        R_HC, t_HC = compose(R_HB_fix, t_HB, R_BC, t_BC)

        zdot = float(R_HC[:,2] @ ez)
        xys  = (abs(R_HC[:,0] @ ex) + abs(R_HC[:,1] @ ey)) - (abs(R_HC[:,0] @ ey) + abs(R_HC[:,1] @ ex))
        score = 2.0 * max(0.0, zdot) + xys
        detR = float(np.linalg.det(R_HC))

        print(f"[{rn}+{fn}] score={score:+.3f}  z·Z={zdot:+.3f}  xys={xys:+.3f}  det={detR:+.6f}  t={t_HC}")

        if score > best_score:
            best_score = score
            best_name  = f"{rn}+{fn}"
            best_RHC   = R_HC
            best_tHC   = t_HC

print(f"[BEST] {best_name}  score={best_score:+.3f}")
np.savez(args.out, R=best_RHC.astype(np.float64), t=best_tHC.astype(np.float64))
print(f"[OK] wrote {args.out}")
