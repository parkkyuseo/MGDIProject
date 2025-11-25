# check_cam_props.py
import cv2, sys
cam_id = int(sys.argv[1]) if len(sys.argv)>1 else 0
cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW if hasattr(cv2,'CAP_DSHOW') else cv2.CAP_ANY)
print("opened:", cap.isOpened())
print("W,H,FPS:", cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FPS))
fcc = int(cap.get(cv2.CAP_PROP_FOURCC))
print("FOURCC:", "".join([chr((fcc >> 8*i) & 0xFF) for i in range(4)]))
cap.release()
