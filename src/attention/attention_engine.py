import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from .eye_utils import calculate_ear
from tensorflow.keras.models import load_model
import os


def run_attention_engine(callback=None):

    # -------- Load CNN Model --------
    model_path = os.path.join(
        os.path.dirname(__file__),
        "../../CNN_classification/attention_cnn.h5"
    )
    cnn_model = load_model(model_path)
    IMG_SIZE = 224

    # -------- MediaPipe Setup --------
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)

    # ---------- Parameters ----------
    YAW_THRESHOLD = 20
    PITCH_THRESHOLD = 15
    EAR_THRESHOLD = 0.22
    SMOOTHING_FRAMES = 30

    attention_buffer = deque(maxlen=SMOOTHING_FRAMES)

    LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

    def compute_head_pose(landmarks):
        nose = landmarks[1]
        chin = landmarks[152]
        left_cheek = landmarks[234]
        right_cheek = landmarks[454]

        yaw = (left_cheek.x - right_cheek.x) * 100
        pitch = (chin.y - nose.y) * 100

        return yaw, pitch

    def get_eye_points(landmarks, indices, w, h):
        return [(landmarks[i].x * w, landmarks[i].y * h) for i in indices]

    def predict_cnn(face_img):
        face_img = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
        face_img = face_img / 255.0
        face_img = np.expand_dims(face_img, axis=0)

        prediction = cnn_model.predict(face_img, verbose=0)[0][0]
        return prediction

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(rgb)

        frame_score = 0
        cnn_score = 0
        avg_ear = 0
        yaw = 0
        pitch = 0

        if results.multi_face_landmarks:

            landmarks = results.multi_face_landmarks[0].landmark

            # -------- Face Crop for CNN --------
            xs = [lm.x for lm in landmarks]
            ys = [lm.y for lm in landmarks]

            x_min = int(min(xs) * w)
            x_max = int(max(xs) * w)
            y_min = int(min(ys) * h)
            y_max = int(max(ys) * h)

            face_crop = frame[y_min:y_max, x_min:x_max]

            # ---------- Head Pose ----------
            yaw, pitch = compute_head_pose(landmarks)
            head_score = 100

            if abs(yaw) > YAW_THRESHOLD:
                head_score -= 50

            if pitch > PITCH_THRESHOLD:
                head_score -= 40

            head_score = max(head_score, 0)

            # ---------- Eye EAR ----------
            left_eye = get_eye_points(landmarks, LEFT_EYE_IDX, w, h)
            right_eye = get_eye_points(landmarks, RIGHT_EYE_IDX, w, h)

            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)

            avg_ear = (left_ear + right_ear) / 2.0

            eye_score = 100

            if avg_ear < EAR_THRESHOLD:
                eye_score = 20

            # ---------- CNN Prediction ----------
            if face_crop.size > 0:
                cnn_confidence = predict_cnn(face_crop)
                cnn_score = cnn_confidence * 100

            # ---------- Hybrid Fusion ----------
            frame_score = int(
                0.54 * head_score +
                0.4 * eye_score +
                0.06 * cnn_score
            )

            # ---------- Hard Override ----------
            if avg_ear < EAR_THRESHOLD or abs(yaw) > 35:
                frame_score = min(frame_score, 30)

        # ---------- Temporal Smoothing ----------
        attention_buffer.append(frame_score)

        smoothed_attention = int(np.mean(attention_buffer)) if attention_buffer else 0

        # ---------- Send Score to Flask ----------
        if callback:
            callback(smoothed_attention)

        # ---------- Status ----------
        status = "Attentive" if smoothed_attention >= 60 else "Distracted"

        color = (0, 255, 0) if smoothed_attention >= 60 else (0, 0, 255)

        # ---------- Display ----------
        cv2.putText(frame, f"Attention Score: {smoothed_attention}%",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        cv2.putText(frame, f"Yaw: {int(yaw)}",
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        cv2.putText(frame, f"Pitch: {int(pitch)}",
                    (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        cv2.putText(frame, f"EAR: {avg_ear:.2f}",
                    (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        cv2.putText(frame, f"CNN: {int(cnn_score)}%",
                    (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

        cv2.putText(frame, status,
                    (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        small_frame = cv2.resize(frame, (480,270))
        cv2.imshow("Attention Engine v3.2 (Final Hybrid)", small_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()