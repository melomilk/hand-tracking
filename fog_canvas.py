import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

fog = np.zeros((480, 640), dtype=np.float32)

while True:
    success, frame = cap.read()
    if not success:
        continue

    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('f'):  # press f to add fog
        fog = np.zeros((480, 640), dtype=np.float32)
    elif key == ord('c'):  # press c to clear
        fog = np.ones((480, 640), dtype=np.float32)
        

# create foggy version of current frame
    blurred = cv2.GaussianBlur(frame, (51, 51), 0)
    # add slight white tint to blurred
    white = np.ones_like(frame) * 255
    foggy = cv2.addWeighted(blurred, 0.75, white, 0.25, 0)

    # fog mask — 0 = foggy, 1 = clear
    # for now pressing f makes whole screen foggy, c clears it
    mask_3ch = np.stack([fog] * 3, axis=-1)
    output = (frame * mask_3ch + foggy * (1 - mask_3ch)).astype(np.uint8)

    cv2.imshow("Fog Canvas", output)

cap.release()
cv2.destroyAllWindows()