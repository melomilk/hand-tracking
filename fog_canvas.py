import cv2
import numpy as np
import mediapipe as mp

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

fog = np.zeros((480, 640), dtype=np.uint8)  # 0 = no fog, 255 = maximum fog

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
prev_finger_pos = None

def add_fog_blob(fog, center_x, center_y, radius=80, max_density=60):
    h, w = fog.shape
    y_coords, x_coords = np.ogrid[:h, :w]
    
    # calculate distance from center for every pixel
    distance = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    
    # radial gradient: closer to center = more fog added
    mask = np.clip(1 - distance / radius, 0, 1)
    
    # add to fog density, clamp at 255
    fog[:] = np.clip(fog.astype(np.float32) + mask * max_density, 0, 255).astype(np.uint8)

def evaporate_fog(fog, evaporation_rate=1.0):
    fog[:] = np.clip(fog.astype(np.float32) - evaporation_rate, 0, 255).astype(np.uint8)


def clear_fog_swipe(fog, decay_rate=30):
    fog[:] = np.clip(fog.astype(np.float32) - decay_rate, 0, 255).astype(np.uint8)

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

    # if a face or hand isn't detected in a given frame
    is_blowing = False
    is_pinching = False
    is_swiping = False
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
        add_fog_blob(fog, int(mouth_x), int(mouth_y))
        # evaporation runs every frame, outside is_blowing
    evaporate_fog(fog)


    if hand_results.multi_hand_landmarks:
            hand_landmarks = hand_results.multi_hand_landmarks[0].landmark
            h, w = frame.shape[:2]

            thumb_finger = hand_landmarks[4]
            index_finger = hand_landmarks[8]

            thumb_x, thumb_y = thumb_finger.x * w, thumb_finger.y * h
            index_x, index_y = index_finger.x * w, index_finger.y * h

            distance = ((index_x - thumb_x) ** 2 + (index_y - thumb_y) ** 2) ** 0.5
            is_pinching = distance < 35

            # detect open palm — all fingertips above their knuckles
            fingertip_ids = [8, 12, 16, 20]
            knuckle_ids =   [6, 10, 14, 18]
            fingers_up = all(
                hand_landmarks[tip].y < hand_landmarks[knuckle].y
                for tip, knuckle in zip(fingertip_ids, knuckle_ids)
            )
            is_swiping = fingers_up and not is_pinching

            cv2.putText(frame, f'distance: {int(distance)}', (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f'pinching: {is_pinching}', (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f'swiping: {is_swiping}', (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    else:
        prev_finger_pos = None
        
    if is_swiping:
            clear_fog_swipe(fog)
            cv2.putText(frame, 'SWIPE!', (250, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)


    if is_pinching and index_x is not None:
        current_pos = (int(index_x), int(index_y))
        if prev_index_pos is not None:
            cv2.line(fog, prev_index_pos, current_pos, 0, thickness=40)
        prev_index_pos = current_pos
    else:
        prev_index_pos = None

# create blue-white fog color layer
    fog_color = np.full_like(frame, (220, 210, 200))  # blue-white tint in BGR
    
    # add noise texture
    noise = np.random.randint(0, 25, (480, 640), dtype=np.uint8)
    noise_3ch = cv2.merge([noise, noise, noise])
    fog_color = cv2.add(fog_color, noise_3ch)
    
    # use density map as alpha mask
    alpha = fog[..., np.newaxis] / 255.0  # convert 0-255 to 0.0-1.0 for blending
    output = (frame * (1 - alpha) + fog_color * alpha).astype(np.uint8)


    cv2.imshow("Fog Canvas", output)

cap.release()
cv2.destroyAllWindows()