import cv2
import mediapipe as mp
import time


mp_hands = mp.solutions.hands #gives access to hand detection model
mp_draw = mp.solutions.drawing_utils #gives tools to draw landmarks on screen
prev_time = 0
hands = mp_hands.Hands(
    static_image_mode = False, #video, no single images
    max_num_hands = 2,
    min_detection_confidence = 0.7
)

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    success, frame = cap.read()
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    if not success or frame is None:
        print ("failed to grab frame")
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #mediapipe expects rgb, opencv reads in bgr

    results = hands.process(rgb_frame) #sends frame to mediapipe

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, #the image to draw on
                hand_landmarks, #21 points to draw
                mp_hands.HAND_CONNECTIONS #draws the lines connecting the dots
            )
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Hand Tracking", frame)
    cv2.setWindowProperty("Hand Tracking", cv2.WND_PROP_TOPMOST, 1)

    key = cv2.waitKey(10) & 0xFF
    if key == ord('q') or key == 27:  
        break
    
cap.release()
cv2.destroyAllWindows()