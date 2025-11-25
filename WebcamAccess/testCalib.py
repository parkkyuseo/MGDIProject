import json, numpy as np

def quat_to_R(x,y,z,w):
    xx,yy,zz = x*x,y*y,z*z; xy, xz, yz = x*y, x*z, y*z; wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy)],
        [2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx)],
        [2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy)]
    ], dtype=float)

with open("./calib/T_HC.json","r") as f:
    J=json.load(f)
R = quat_to_R(J["R_quat"]["x"],J["R_quat"]["y"],J["R_quat"]["z"],J["R_quat"]["w"])
ez_H = R @ np.array([0,0,1.0])  # 카메라 +Z가 H에서 어디를 가리키는가
print("R[:,2] =", ez_H, "   dot with +Z_H =", ez_H @ np.array([0,0,1.0]))
