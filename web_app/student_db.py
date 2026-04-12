import sqlite3
import numpy as np
import os
import time

DB_PATH = os.path.join(os.path.dirname(__file__), "classroom_data.db")
THUMBNAILS_DIR = os.path.join(os.path.dirname(__file__), "static", "thumbnails")

def init_db():
    if not os.path.exists(THUMBNAILS_DIR):
        os.makedirs(THUMBNAILS_DIR)

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id TEXT PRIMARY KEY,
            label TEXT,
            embedding_blob BLOB,
            thumbnail_path TEXT,
            created_at REAL
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS attention_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT,
            score INTEGER,
            status TEXT,
            timestamp REAL,
            FOREIGN KEY(student_id) REFERENCES students(id)
        )
    ''')

    conn.commit()
    conn.close()

def _embedding_to_blob(embedding):
    return embedding.astype(np.float32).tobytes()

def _blob_to_embedding(blob):
    return np.frombuffer(blob, dtype=np.float32)

def register_student(embedding, face_image):
    import cv2
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Generate next ID using MAX to avoid collisions after deletions
    c.execute("SELECT id FROM students ORDER BY created_at DESC LIMIT 1")
    row = c.fetchone()
    if row:
        # Extract number from "Student_XXX"
        try:
            last_num = int(row[0].split("_")[1])
        except (IndexError, ValueError):
            last_num = 0
        next_num = last_num + 1
    else:
        next_num = 1
    student_id = f"Student_{next_num:03d}"
    
    # Save thumbnail (face_crop is already BGR from OpenCV)
    thumb_filename = f"{student_id}_{int(time.time())}.jpg"
    thumb_path = os.path.join(THUMBNAILS_DIR, thumb_filename)
    cv2.imwrite(thumb_path, face_image)

    # Insert to DB
    c.execute('''
        INSERT INTO students (id, label, embedding_blob, thumbnail_path, created_at)
        VALUES (?, ?, ?, ?, ?)
    ''', (student_id, student_id, _embedding_to_blob(embedding), f"thumbnails/{thumb_filename}", time.time()))

    conn.commit()
    conn.close()
    return student_id

def find_student(query_embedding, tolerance=0.6):
    import face_recognition
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("SELECT id, embedding_blob FROM students")
    rows = c.fetchall()
    conn.close()

    if not rows:
        return None

    known_ids = [row[0] for row in rows]
    known_embeddings = [_blob_to_embedding(row[1]) for row in rows]

    # Compare embeddings
    distances = face_recognition.face_distance(known_embeddings, query_embedding)
    
    if len(distances) == 0:
        return None

    best_match_index = np.argmin(distances)
    if distances[best_match_index] <= tolerance:
        return known_ids[best_match_index]

    return None

def log_attention(student_id, score, status):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO attention_logs (student_id, score, status, timestamp)
        VALUES (?, ?, ?, ?)
    ''', (student_id, score, status, time.time()))
    conn.commit()
    conn.close()

def get_student_report():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute('''
        SELECT 
            s.id, 
            s.label, 
            s.thumbnail_path,
            COUNT(l.id) as total_frames,
            AVG(l.score) as avg_score,
            SUM(CASE WHEN l.status = 'Attentive' THEN 1 ELSE 0 END) as attentive_frames
        FROM students s
        LEFT JOIN attention_logs l ON s.id = l.student_id
        GROUP BY s.id
        ORDER BY s.created_at ASC
    ''')
    
    rows = c.fetchall()
    conn.close()

    report = []
    for row in rows:
        total = row['total_frames'] or 1 # prevent div by zero
        attentive = row['attentive_frames'] or 0
        avg = int(row['avg_score']) if row['avg_score'] is not None else 0
        
        report.append({
            "id": row['id'],
            "label": row['label'],
            "thumbnail": row['thumbnail_path'],
            "avg_score": avg,
            "attentive_percent": int((attentive / total) * 100) if total > 1 else 0
        })

    return report

def rename_student(student_id, new_label):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE students SET label = ? WHERE id = ?", (new_label, student_id))
    conn.commit()
    conn.close()

def delete_student(student_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Get thumbnail path to delete physical file
    c.execute("SELECT thumbnail_path FROM students WHERE id = ?", (student_id,))
    row = c.fetchone()
    if row and row[0]:
        static_dir = os.path.join(os.path.dirname(__file__), "static")
        file_path = os.path.join(static_dir, row[0].replace("/", os.sep))
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error removing thumbnail: {e}")

    # Delete db records
    c.execute("DELETE FROM attention_logs WHERE student_id = ?", (student_id,))
    c.execute("DELETE FROM students WHERE id = ?", (student_id,))
    
    conn.commit()
    conn.close()

