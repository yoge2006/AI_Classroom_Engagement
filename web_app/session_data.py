import time

session_scores = []
face_counts = []
session_start_time = None


def reset_session():
    global session_start_time
    session_scores.clear()
    face_counts.clear()
    session_start_time = time.time()


def get_session_duration():
    if session_start_time is None:
        return 0
    return int(time.time() - session_start_time)