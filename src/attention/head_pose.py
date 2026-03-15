import cv2
import mediapipe as mp
import numpy as np

def run_attention_tracker():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=10,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)

    ATTENTION_YAW_THRESHOLD = 20

    def estimate_yaw(landmarks, frame_width):
        nose = landmarks[1]
        left_cheek = landmarks[234]
        right_cheek = landmarks[454]

        left_x = left_cheek.x * frame_width
        right_x = right_cheek.x * frame_width

        yaw = (left_x - right_x) / frame_width * 100
        return yaw

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(rgb)

        attentive = 0
        total = 0

        if results.multi_face_landmarks:
            for face in results.multi_face_landmarks:
                total += 1
                yaw = estimate_yaw(face.landmark, w)

                if abs(yaw) < ATTENTION_YAW_THRESHOLD:
                    attentive += 1
                    label = "Attentive"
                    color = (0, 255, 0)
                else:
                    label = "Distracted"
                    color = (0, 0, 255)

                cv2.putText(frame, label, (20, 40 + total * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        score = int((attentive / total) * 100) if total > 0 else 0

        cv2.putText(frame, f"Attention Score: {score}%",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 0), 3)

        cv2.imshow("AI Classroom Attention Monitor", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
