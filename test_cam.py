import cv2

cap = cv2.VideoCapture(0)
print("opened:", cap.isOpened())
success, frame = cap.read()
print("success:", success)
print("frame:", frame)
cap.release()