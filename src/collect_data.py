import csv
import os

import cv2
import mediapipe as mp

from normalize import normalize_landmarks


MODEL_PATH = "models/hand_landmarker.task"
DATASET_PATH = "data/raw/gesture_data.csv"


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


GESTURE_KEYS = {
    ord("t"): "thumbs_up",
    ord("T"): "thumbs_up",

    ord("d"): "thumbs_down",
    ord("D"): "thumbs_down",

    ord("p"): "open_palm",
    ord("P"): "open_palm",

    ord("f"): "fist",
    ord("F"): "fist",

    ord("2"): "two_fingers",

    ord("n"): "no_gesture",
    ord("N"): "no_gesture",
}


os.makedirs("data/raw", exist_ok=True)


options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)


hand_landmarker = HandLandmarker.create_from_options(options)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open webcam.")
    hand_landmarker.close()
    exit()


file_exists = os.path.exists(DATASET_PATH)

with open(DATASET_PATH, "a", newline="") as file:

    writer = csv.writer(file)

    if not file_exists:

        header = []

        for landmark_index in range(21):
            header.append(f"x{landmark_index}")
            header.append(f"y{landmark_index}")
            header.append(f"z{landmark_index}")

        header.append("label")

        writer.writerow(header)

    frame_timestamp = 0
    current_gesture = None
    sample_count = 0

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

        # Draw landmarks and collect data
        if results.hand_landmarks:

            for hand_landmarks in results.hand_landmarks:

                height, width, _ = frame.shape

                # Draw all 21 landmarks
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

                # Collect only when a gesture is selected
                if current_gesture:

                    feature_vector = normalize_landmarks(
                        hand_landmarks
                    )

                    if len(feature_vector) == 63:

                        writer.writerow(
                            feature_vector + [current_gesture]
                        )

                        file.flush()

                        sample_count += 1

                break

        # Display current gesture
        gesture_text = current_gesture if current_gesture else "None"

        cv2.putText(
            frame,
            f"Gesture: {gesture_text}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Samples: {sample_count}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "T:ThumbsUp D:ThumbsDown P:Palm F:Fist 2:Two N:None Q:Quit",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        cv2.imshow("Dataset Collection", frame)

        key = cv2.waitKey(1) & 0xFF

        if key != 255:

            print("Key pressed:", key)

        if key in GESTURE_KEYS:

            current_gesture = GESTURE_KEYS[key]

            print(
                f"Collecting gesture: {current_gesture}"
            )

        elif key == ord("q"):

            break


camera.release()
cv2.destroyAllWindows()
hand_landmarker.close()

print(f"Dataset saved to: {DATASET_PATH}")
print(f"Total samples collected: {sample_count}")