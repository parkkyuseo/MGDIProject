import cv2
for idx in range(4):
    for api_name, api in [("MSMF", cv2.CAP_MSMF), ("DSHOW", cv2.CAP_DSHOW)]:
        cap = cv2.VideoCapture(idx, api)
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
        ret, frame = cap.read()
        print(f"idx={idx} api={api_name} ret={ret} shape={None if not ret else frame.shape}")
        if ret:
            cv2.imwrite(f"test_idx{idx}_{api_name}.jpg", frame)
        cap.release()
