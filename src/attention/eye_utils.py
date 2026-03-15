import numpy as np

def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def calculate_ear(eye_landmarks):
    """
    eye_landmarks: list of 6 (x, y) tuples for one eye
    EAR formula:
    (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """

    p1, p2, p3, p4, p5, p6 = eye_landmarks

    vertical_1 = euclidean_distance(p2, p6)
    vertical_2 = euclidean_distance(p3, p5)
    horizontal = euclidean_distance(p1, p4)

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear
