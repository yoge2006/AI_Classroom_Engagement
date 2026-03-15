import cv2
import os
import mediapipe as mp

# -------- SETTINGS --------
SAVE_TRAIN = False
  # True = save to train, False = save to val
IMG_SIZE = 224

# Folder paths
base_path = "dataset/train" if SAVE_TRAIN else "dataset/val"
attentive_path = os.path.join(base_path, "attentive")
inattentive_path = os.path.join(base_path, "inattentive")

os.makedirs(attentive_path, exist_ok=True)
os.makedirs(inattentive_path, exist_ok=True)

# MediaPipe Face Detection
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

attentive_count = 0
inattentive_count = 0

print("Press A for attentive")
print("Press D for inattentive")
print("Press ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)

    face_crop = None

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            x2 = int((bbox.xmin + bbox.width) * w)
            y2 = int((bbox.ymin + bbox.height) * h)

            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size > 0:
                face_crop = cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    cv2.putText(frame, f"Attentive: {attentive_count}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.putText(frame, f"Inattentive: {inattentive_count}", (10,60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    cv2.imshow("Dataset Collector", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('a') and face_crop is not None:
        filename = os.path.join(attentive_path, f"a_{attentive_count}.jpg")
        cv2.imwrite(filename, face_crop)
        attentive_count += 1
        print("Saved attentive")

    elif key == ord('d') and face_crop is not None:
        filename = os.path.join(inattentive_path, f"d_{inattentive_count}.jpg")
        cv2.imwrite(filename, face_crop)
        inattentive_count += 1
        print("Saved inattentive")

    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()
