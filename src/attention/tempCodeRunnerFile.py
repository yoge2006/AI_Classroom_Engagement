import cv20
import mediapipe as mp
import numpy as np
from collections import deque
from attention.eye_utils import calculate_ear


def run_attention_engine():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,   # single person for accuracy v2.0
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)

    # ---- Parameters ----
    YAW_THRESHOLD = 25
    PITCH_THRESHOLD = 15
    SMOOTHING_FRAMES = 30

    attention_buffer = deque(maxlen=SMOOTHING_FRAMES)

    def compute_head_pose(landmarks, w, h):
        # Key landmarks
        nose = landmarks[1]
        chin = landmarks[152]
        left_cheek = landmarks[234]
        right_cheek = landmarks[454]

        nose_x, nose_y = nose.x * w, nose.y * h
        chin_x, chin_y = chin.x * w, chin.y * h
        left_x = left_cheek.x * w
        right_x = right_cheek.x * w

        # Yaw (left-right)
        yaw = (left_x - right_x) / w * 100

        # Pitch (up-down)
        pitch = (chin_y - nose_y) / h * 100

        return yaw, pitch

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(rgb)

        frame_score = 0
        status = "No Face"

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            yaw, pitch = compute_head_pose(landmarks, w, h)

            # ---- Head Pose Scoring ----
            head_score = 100

            if abs(yaw) > YAW_THRESHOLD:
                head_score -= 40

            if pitch > PITCH_THRESHOLD:   # looking down
                head_score -= 40

            head_score = max(head_score, 0)
            frame_score = head_score

            status = "Attentive" if frame_score >= 60 else "Distracted"

            # Debug info
            cv2.putText(frame, f"Yaw: {int(yaw)}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.putText(frame, f"Pitch: {int(pitch)}", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        attention_buffer.append(frame_score)

        smoothed_attention = int(np.mean(attention_buffer)) if attention_buffer else 0

        color = (0, 255, 0) if smoothed_attention >= 60 else (0, 0, 255)

        cv2.putText(frame, f"Attention Score: {smoothed_attention}%",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        cv2.putText(frame, status, (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow("Attention Engine v2.0", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
