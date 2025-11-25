# save as: win_cam_sanity.py
import cv2, sys

backend = cv2.CAP_DSHOW  # 또는 cv2.CAP_MSMF
for idx in range(0, 6):
    cap = cv2.VideoCapture(idx, backend)
    if cap.isOpened():
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"[OK] idx={idx} size={int(w)}x{int(h)}")
        ret, f = cap.read()
        if ret:
            cv2.imshow(f"cam{idx}", f)
            cv2.waitKey(300)
        cap.release()
cv2.destroyAllWindows()
