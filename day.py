import cv2
import mediapipe as mp
import time

# Mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)

# Blink counter setup
blink_count = 0
blink_start = None
EAR_THRESHOLD = 0.2
CONSEC_FRAMES = 3
frame_counter = 0

# Eye landmark indices from Mediapipe (for both eyes)
LEFT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]

def get_ear(eye_landmarks):
    from math import dist
    A = dist(eye_landmarks[1], eye_landmarks[5])
    B = dist(eye_landmarks[2], eye_landmarks[4])
    C = dist(eye_landmarks[0], eye_landmarks[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Camera feed
cap = cv2.VideoCapture(0)
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mesh_points = [(int(point.x * w), int(point.y * h)) for point in face_landmarks.landmark]

            left_eye = [mesh_points[p] for p in LEFT_EYE_LANDMARKS]
            right_eye = [mesh_points[p] for p in RIGHT_EYE_LANDMARKS]

            left_ear = get_ear(left_eye)
            right_ear = get_ear(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0

            if avg_ear < EAR_THRESHOLD:
                frame_counter += 1
            else:
                if frame_counter >= CONSEC_FRAMES:
                    blink_count += 1
                frame_counter = 0

            cv2.putText(frame, f'Blinks: {blink_count}', (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow("Blink Detector (MediaPipe)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

duration = time.time() - start_time
print("\n--- Useless Blink Commentary ---")
if blink_count == 0:
    print("You didn't blink. Robot vibes.")
elif blink_count < 5:
    print("You blinked less than 5 times. Too focused? Or tired?")
elif blink_count < 15:
    print("Normal blinking detected. Carry on.")
else:
    print("Bro. That's a blink spam. Chill.")

cap.release()
cv2.destroyAllWindows()
