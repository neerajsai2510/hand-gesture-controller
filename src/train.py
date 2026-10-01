import csv

import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


DATASET_PATH = "data/raw/gesture_data.csv"
MODEL_PATH = "models/gesture_classifier.pkl"


features = []
labels = []


with open(DATASET_PATH, "r", newline="") as file:

    reader = csv.DictReader(file)

    for row in reader:

        feature_vector = []

        for index in range(63):
            feature_vector.append(
                float(row[
                    f"{'xyz'[index % 3]}{index // 3}"
                ])
            )

        features.append(feature_vector)
        labels.append(row["label"])


print("Total samples:", len(features))
print("Number of features:", len(features[0]))


X_train, X_test, y_train, y_test = train_test_split(
    features,
    labels,
    test_size=0.20,
    random_state=42,
    stratify=labels
)


print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


classifier = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)


classifier.fit(X_train, y_train)


predictions = classifier.predict(X_test)


accuracy = accuracy_score(
    y_test,
    predictions
)


print()
print("Model Accuracy:", accuracy)
print()


print("Classification Report:")
print(
    classification_report(
        y_test,
        predictions
    )
)


joblib.dump(
    classifier,
    MODEL_PATH
)


print()
print("Model saved to:", MODEL_PATH)