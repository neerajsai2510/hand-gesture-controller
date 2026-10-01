import math


def normalize_landmarks(landmarks):
    wrist = landmarks[0]

    normalized_landmarks = []

    for landmark in landmarks:
        x = landmark.x - wrist.x
        y = landmark.y - wrist.y
        z = landmark.z - wrist.z

        normalized_landmarks.append([x, y, z])

    scale = 0.0

    for landmark in normalized_landmarks:
        x, y, z = landmark

        distance = math.sqrt(
            x * x +
            y * y +
            z * z
        )

        scale = max(scale, distance)

    if scale == 0:
        return []

    for landmark in normalized_landmarks:
        landmark[0] /= scale
        landmark[1] /= scale
        landmark[2] /= scale

    feature_vector = []

    for landmark in normalized_landmarks:
        feature_vector.extend(landmark)

    return feature_vector