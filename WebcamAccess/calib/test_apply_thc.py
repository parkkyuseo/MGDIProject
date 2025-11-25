# ./calib/test_apply_thc.py
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
