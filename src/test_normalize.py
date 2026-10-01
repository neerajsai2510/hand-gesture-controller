from normalize import normalize_landmarks


class Landmark:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


landmarks = [
    Landmark(0.50, 0.80, 0.00),
    Landmark(0.60, 0.70, 0.00),
    Landmark(0.70, 0.60, 0.00),
]


normalized = normalize_landmarks(landmarks)


for index, landmark in enumerate(normalized):
    print(index, landmark)