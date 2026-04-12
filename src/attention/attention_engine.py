import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from .eye_utils import calculate_ear
from tensorflow.keras.models import load_model
import os
import face_recognition


def run_attention_engine(callback=None):

    # -------- Load CNN Model --------
    model_path = os.path.join(
        os.path.dirname(__file__),
        "../../CNN_classification/attention_cnn.h5"
    )
    cnn_model = load_model(model_path)
    IMG_SIZE = 224

    # -------- MediaPipe Setup (Multi-Face) --------
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=10,
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

    # Per-face attention buffers (up to 10 faces)
    attention_buffers = [deque(maxlen=SMOOTHING_FRAMES) for _ in range(10)]

    # Global smoothing buffer for overall class score
    global_buffer = deque(maxlen=SMOOTHING_FRAMES)

    LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

    def compute_head_pose(landmarks):
        nose = landmarks[1]
        chin = landmarks[152]
        left_cheek = landmarks[234]
        right_cheek = landmarks[454]
        top_head = landmarks[10]

        # Calculate Yaw: horizontal shift of the nose from the face center
        face_width = max(abs(right_cheek.x - left_cheek.x), 0.001)
        center_x = (left_cheek.x + right_cheek.x) / 2.0
        yaw = ((nose.x - center_x) / face_width) * 100

        # Calculate Pitch: vertical shift of the nose from the face center
        face_height = max(abs(chin.y - top_head.y), 0.001)
        center_y = (top_head.y + chin.y) / 2.0
        # Subtract ~0.1 (10%) to rough-center pitch at 0 when looking straight ahead
        pitch = (((nose.y - center_y) / face_height) - 0.1) * 100

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

        if not hasattr(run_attention_engine, "frame_counter"):
            run_attention_engine.frame_counter = 0
        run_attention_engine.frame_counter += 1

        face_count = 0
        face_scores = []
        per_face_data = []

        if results.multi_face_landmarks:

            face_count = len(results.multi_face_landmarks)

            for face_idx, face_landmarks in enumerate(results.multi_face_landmarks):

                landmarks = face_landmarks.landmark

                frame_score = 0
                cnn_score = 0
                avg_ear = 0
                yaw = 0
                pitch = 0

                # -------- Face Crop for CNN --------
                xs = [lm.x for lm in landmarks]
                ys = [lm.y for lm in landmarks]

                x_min = max(int(min(xs) * w), 0)
                x_max = min(int(max(xs) * w), w)
                y_min = max(int(min(ys) * h), 0)
                y_max = min(int(max(ys) * h), h)

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
                    
                    # Generate 128D Embedding for tracking (LAG OPTIMIZATION)
                    # Running this every frame drops FPS to 2. We run it every 15 frames (~0.5 sec)
                    if run_attention_engine.frame_counter % 15 == 0 or run_attention_engine.frame_counter == 1:
                        try:
                            # Use FULL frame with face location for consistent embeddings.
                            # Dlib needs full-frame context for proper landmark alignment.
                            # Format: (top, right, bottom, left) in CSS order
                            box = [(y_min, x_max, y_max, x_min)]
                            face_encodings = face_recognition.face_encodings(rgb, known_face_locations=box)
                            embedding = face_encodings[0] if len(face_encodings) > 0 else None
                        except Exception as e:
                            print("FACE ENCODING ERROR:", e)
                            embedding = None
                    else:
                        embedding = None
                        face_crop = None  # Save bandwidth too
                else:
                    embedding = None
                    face_crop = None

                # ---------- Hybrid Fusion ----------
                frame_score = int(
                    0.54 * head_score +
                    0.4 * eye_score +
                    0.06 * cnn_score
                )

                # ---------- Hard Override ----------
                if avg_ear < EAR_THRESHOLD or abs(yaw) > 28:
                    frame_score = min(frame_score, 30)

                # Per-face smoothing
                if face_idx < len(attention_buffers):
                    attention_buffers[face_idx].append(frame_score)
                    smoothed = int(np.mean(attention_buffers[face_idx]))
                else:
                    smoothed = frame_score

                face_scores.append(smoothed)

                # Determine per-face status
                status = "Attentive" if smoothed >= 60 else "Distracted"
                color = (0, 255, 0) if smoothed >= 60 else (0, 0, 255)

                per_face_data.append({
                    "face_idx": face_idx,
                    "score": smoothed,
                    "yaw": int(yaw),
                    "pitch": int(pitch),
                    "ear": avg_ear,
                    "cnn": int(cnn_score),
                    "status": status,
                    "embedding": embedding,
                    "face_crop": face_crop
                })

                # ---------- Draw per-face label ----------
                label_y = y_min - 10 if y_min > 30 else y_max + 20
                cv2.putText(frame, f"#{face_idx+1} {status} {smoothed}%",
                            (x_min, label_y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 2)

                # Draw bounding box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

        # ---------- Overall Class Score ----------
        if face_scores:
            overall_score = int(np.mean(face_scores))
        else:
            overall_score = 0

        global_buffer.append(overall_score)
        smoothed_overall = int(np.mean(global_buffer)) if global_buffer else 0

        # ---------- Send Data to Flask ----------
        if callback:
            callback({
                "score": smoothed_overall,
                "face_count": face_count,
                "per_face": per_face_data
            })

        # ---------- Display ----------
        attentive_count = sum(1 for s in face_scores if s >= 60)
        distracted_count = face_count - attentive_count

        cv2.putText(frame, f"Class Score: {smoothed_overall}%",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 3)

        cv2.putText(frame, f"Faces: {face_count}  Attentive: {attentive_count}  Distracted: {distracted_count}",
                    (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)

        small_frame = cv2.resize(frame, (640, 360))
        
        # Save globally for Flask to serve via MJPEG stream instead of a popup
        ret_jpg, buffer_img = cv2.imencode('.jpg', small_frame)
        if ret_jpg:
            run_attention_engine.latest_frame = buffer_img.tobytes()

        # Check if stop was requested from the web UI
        if hasattr(run_attention_engine, '_stop_flag') and run_attention_engine._stop_flag:
            run_attention_engine._stop_flag = False
            run_attention_engine.latest_frame = None
            break

    cap.release()