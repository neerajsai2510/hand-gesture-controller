import cv2
import mediapipe as mp
import joblib

from normalize import normalize_landmarks


MODEL_PATH = "models/hand_landmarker.task"
CLASSIFIER_PATH = "models/gesture_classifier.pkl"


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


classifier = joblib.load(CLASSIFIER_PATH)

print("Gesture classifier loaded successfully.")


options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)


hand_landmarker = HandLandmarker.create_from_options(
    options
)


camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open webcam.")
    hand_landmarker.close()
    exit()


frame_timestamp = 0


while camera.isOpened():

    success, frame = camera.read()

    if not success:
        print("ERROR: Could not read webcam frame.")
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    results = hand_landmarker.detect_for_video(
        mp_image,
        frame_timestamp
    )

    frame_timestamp += 1

    gesture = "NO HAND"
    confidence = 0.0

    if results.hand_landmarks:

        for hand_landmarks in results.hand_landmarks:

            feature_vector = normalize_landmarks(
                hand_landmarks
            )

            if len(feature_vector) == 63:

                prediction = classifier.predict(
                    [feature_vector]
                )[0]

                probabilities = classifier.predict_proba(
                    [feature_vector]
                )[0]

                confidence = max(probabilities)

                gesture = prediction

            height, width, _ = frame.shape

            for landmark in hand_landmarks:

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )

            break

    cv2.putText(
        frame,
        f"Gesture: {gesture}",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Confidence: {confidence * 100:.2f}%",
        (20, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "Real-Time Gesture Recognition",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()
hand_landmarker.close()