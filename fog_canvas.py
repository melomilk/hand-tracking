import cv2
import numpy as np
import mediapipe as mp

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# fog starts clear
fog = np.ones((480, 640), dtype=np.float32)

# initialize face mesh
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# initialize hands mesh
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

prev_index_pos = None  #in order to make lines smooth (w/out gaps)

while True:
    success, frame = cap.read()
    if not success:
        continue

    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    hand_results = hands.process(rgb_frame)

    # defaults — important so the program doesn't crash
    # if a face or hand isn't detected in a given frame
    is_blowing = False
    is_pinching = False
    index_x, index_y = None, None
    

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]

        upper_lip = landmarks[13]
        lower_lip = landmarks[14]
        left_cheek = landmarks[234]
        right_cheek = landmarks[454]

        upper_y = upper_lip.y * h
        lower_y = lower_lip.y * h
        left_x = left_cheek.x * w
        right_x = right_cheek.x * w

        mouth_open = (lower_y - upper_y) > 15
        face_width = right_x - left_x
        is_blowing = mouth_open and face_width > 200
        mouth_x = (left_x + right_x) / 2
        mouth_y = (upper_y + lower_y) / 2

        cv2.putText(frame, f'mouth open: {mouth_open}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f'blowing: {is_blowing}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f'face width: {int(face_width)}', (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    if is_blowing:
        breath_mask = np.zeros((480, 640), dtype=np.float32)
        cv2.circle(breath_mask, (int(mouth_x), int(mouth_y)), 80, 1.0, thickness=-1)
        breath_mask = cv2.GaussianBlur(breath_mask, (81, 81), 0)
        fog = np.clip(fog - breath_mask * 0.05, 0, 1)

    if hand_results.multi_hand_landmarks:
        hand_landmarks = hand_results.multi_hand_landmarks[0].landmark
        h, w = frame.shape[:2]

        thumb_finger = hand_landmarks[4]
        index_finger = hand_landmarks[8]

        thumb_x, thumb_y = thumb_finger.x * w, thumb_finger.y * h
        index_x, index_y = index_finger.x * w, index_finger.y * h

        distance = ((index_x - thumb_x) ** 2 + (index_y - thumb_y) ** 2) ** 0.5
        is_pinching = distance < 35

        cv2.putText(frame, f'distance: {int(distance)}', (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f'pinching: {is_pinching}', (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if is_pinching and index_x is not None:
        cv2.circle(fog, (int(index_x), int(index_y)), 20, 1, thickness=-1)

    if is_pinching and index_x is not None:
        current_pos = (int(index_x), int(index_y))
        if prev_index_pos is not None:
            cv2.line(fog, prev_index_pos, current_pos, 1, thickness=40)
        prev_index_pos = current_pos
    else:
        prev_index_pos = None 

    # create foggy version
    blurred = cv2.GaussianBlur(frame, (51, 51), 0)
    white = np.ones_like(frame) * 255
    foggy = cv2.addWeighted(blurred, 0.75, white, 0.25, 0)

    mask_3ch = np.stack([fog] * 3, axis=-1)
    output = (frame * mask_3ch + foggy * (1 - mask_3ch)).astype(np.uint8)

    cv2.imshow("Fog Canvas", output)

cap.release()
cv2.destroyAllWindows()