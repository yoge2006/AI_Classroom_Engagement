import sys
import os
import csv

from flask import Flask, render_template, request, redirect, session, send_file
from flask_socketio import SocketIO
import threading

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.attention.attention_engine import run_attention_engine
from session_data import session_scores


app = Flask(__name__)
app.secret_key = "secret123"

socketio = SocketIO(app)

teachers = {
    "teacher@edu.com": "1234"
}

attention_score = 0


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

    def callback(score):
        global attention_score
        attention_score = score
        session_scores.append(score)

    run_attention_engine(callback)


@socketio.on("start_monitoring")
def start_monitoring():

    thread = threading.Thread(target=start_ai)
    thread.start()


@socketio.on("request_data")
def send_data():

    socketio.emit("attention_update", {
        "score": attention_score
    })


@app.route("/download_report")
def download_report():

    filename = "session_report.csv"

    with open(filename, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow(["Frame", "Attention Score"])

        for i, score in enumerate(session_scores):
            writer.writerow([i, score])

    return send_file(filename, as_attachment=True)


if __name__ == "__main__":
    socketio.run(app, debug=True, port=5001)