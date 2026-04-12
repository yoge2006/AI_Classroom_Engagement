import sys
import os
import csv
import time

from flask import Flask, render_template, request, redirect, session, send_file, jsonify, Response
from flask_socketio import SocketIO
import threading

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.attention.attention_engine import run_attention_engine
from session_data import (
    session_scores, face_counts,
    reset_session, get_session_duration
)
import student_db


app = Flask(__name__)
app.secret_key = "secret123"

# Initialize SQLite Student Database
student_db.init_db()

socketio = SocketIO(app)

teachers = {
    "teacher@edu.com": "1234"
}

attention_score = 0
current_face_count = 0
peak_face_count = 0


@app.route("/", methods=["GET", "POST"])
def login():

    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        if email in teachers and teachers[email] == password:
            session["user"] = email
            return redirect("/dashboard")

    return render_template("login.html")


@app.route("/dashboard")
def dashboard():

    if "user" not in session:
        return redirect("/")

    return render_template("dashboard.html")


@app.route("/session")
def session_page():
    return render_template("session.html")


def start_ai():

    global attention_score, current_face_count, peak_face_count

    reset_session()

    # Cache for tracking faces between heavy embedding frames
    session_face_cache = {}

    def callback(data):
        global attention_score, current_face_count, peak_face_count

        try:
            attention_score = data["score"]
            current_face_count = data["face_count"]

            if data["face_count"] > peak_face_count:
                 peak_face_count = data["face_count"]

            session_scores.append(data["score"])
            face_counts.append(data["face_count"])

            # Process per-face auto-registration and tracking
            processed_faces = []
            for face_data in data.get("per_face", []):
                embedding = face_data.get("embedding")
                face_crop = face_data.get("face_crop")
                face_idx = face_data.get("face_idx", 0)
                
                student_id = session_face_cache.get(face_idx, "Unknown")
                
                # Heavy calculation frame (~every 15 frames)
                if embedding is not None and face_crop is not None:
                    # Try to find existing student
                    matched_id = student_db.find_student(embedding)
                    
                    # If not found, auto-register them
                    if not matched_id:
                         student_id = student_db.register_student(embedding, face_crop)
                    else:
                         student_id = matched_id
                         
                    # Update persistent tracking cache
                    session_face_cache[face_idx] = student_id
                    
                # Log their individual attention score (EVERY frame)
                if student_id != "Unknown":
                    student_db.log_attention(student_id, face_data["score"], face_data["status"])

                # Remove bulky embedding and crop data before sending over socket
                face_data.pop("embedding", None)
                face_data.pop("face_crop", None)
                face_data["student_id"] = student_id
                processed_faces.append(face_data)
                
            # Send clean data to frontend
            data["per_face"] = processed_faces

        except Exception as e:
            print(f"[CALLBACK ERROR] {e}")
            import traceback
            traceback.print_exc()

    run_attention_engine(callback)


monitoring_thread = None

@socketio.on("start_monitoring")
def start_monitoring():
    global monitoring_thread
    # Prevent starting multiple threads
    if monitoring_thread and monitoring_thread.is_alive():
        return
    monitoring_thread = threading.Thread(target=start_ai, daemon=True)
    monitoring_thread.start()


@socketio.on("stop_monitoring")
def stop_monitoring():
    from src.attention.attention_engine import run_attention_engine
    run_attention_engine._stop_flag = True


@socketio.on("request_data")
def send_data():

    socketio.emit("attention_update", {
        "score": attention_score,
        "face_count": current_face_count,
        "peak_faces": peak_face_count
    })


def generate_video_stream():
    from src.attention.attention_engine import run_attention_engine
    while True:
        if hasattr(run_attention_engine, 'latest_frame') and run_attention_engine.latest_frame is not None:
             frame = run_attention_engine.latest_frame
             yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
             socketio.sleep(0.04) # Yields to SocketIO to process live dashboard data
        else:
             socketio.sleep(0.1)

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/api/session_stats")
def session_stats():

    avg_score = int(sum(session_scores) / len(session_scores)) if session_scores else 0
    max_faces = peak_face_count
    duration = get_session_duration()

    return jsonify({
        "avg_score": avg_score,
        "max_faces": max_faces,
        "duration": duration,
        "total_frames": len(session_scores)
    })

# ---------- NEW STUDENT ROUTES ----------

@app.route("/students")
def students_page():
    if "user" not in session:
        return redirect("/")
    return render_template("students.html")


@app.route("/api/students")
def api_students():
    report = student_db.get_student_report()
    return jsonify(report)


@app.route("/api/student/<student_id>/rename", methods=["POST"])
def api_rename_student(student_id):
    data = request.json
    new_label = data.get("name")
    if new_label:
        student_db.rename_student(student_id, new_label)
        return jsonify({"success": True})
    return jsonify({"success": False, "error": "No name provided"}), 400

@app.route("/api/student/<student_id>", methods=["DELETE"])
def api_delete_student(student_id):
    try:
        student_db.delete_student(student_id)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# ----------------------------------------

@app.route("/download_report")
def download_report():

    filename = "session_report.csv"

    with open(filename, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow(["Frame", "Timestamp (s)", "Attention Score", "Faces Detected"])

        for i, score in enumerate(session_scores):
            fc = face_counts[i] if i < len(face_counts) else 0
            writer.writerow([i, i * 2, score, fc])

    return send_file(filename, as_attachment=True)


if __name__ == "__main__":
    socketio.run(app, debug=True, port=5001)