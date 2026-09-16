"""
Simple Classifier for Teachable Machine .h5 models
Works well with TensorFlow 2.15 / 2.16 on Apple Silicon
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from typing import List, Tuple
import os


class Classifier:
    def __init__(self, model_path: str, labels_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels not found: {labels_path}")

        print(f"Loading model: {model_path}")
        self.model = load_model(model_path, compile=False)

        with open(labels_path, "r", encoding="utf-8") as f:
            self.labels = [line.strip() for line in f.readlines() if line.strip()]

        print(f"Model loaded | TF {tf.__version__} | {len(self.labels)} labels")

        self.data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        resized = cv2.resize(img, (224, 224))
        array = np.asarray(resized, dtype=np.float32)
        return (array / 127.0) - 1.0

    def predict(self, img: np.ndarray) -> Tuple[List[float], int, str]:
        processed = self.preprocess(img)
        self.data[0] = processed

        predictions = self.model.predict(self.data, verbose=0)
        probs = predictions[0].tolist()

        index = int(np.argmax(predictions))
        label = self.labels[index] if index < len(self.labels) else f"Class {index}"

        return probs, index, label

    def getPrediction(self, img: np.ndarray, draw: bool = True,
                      pos: Tuple[int, int] = (30, 50),
                      scale: float = 1.5,
                      color: Tuple[int, int, int] = (0, 255, 0),
                      thickness: int = 2) -> Tuple[List[float], int]:
        probs, index, label = self.predict(img)

        if draw:
            cv2.putText(img, label, pos, cv2.FONT_HERSHEY_COMPLEX,
                        scale, color, thickness)

        return probs, index

    def get_label(self, index: int) -> str:
        return self.labels[index] if 0 <= index < len(self.labels) else f"Unknown ({index})"


# ====================== Quick Test ======================

if __name__ == "__main__":
    MODEL_PATH = "hand-gesture/hand-sign-detection/model/keras_model.h5"
    LABELS_PATH = "hand-gesture/hand-sign-detection/model/labels.txt"

    classifier = Classifier(MODEL_PATH, LABELS_PATH)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not opened. Try changing to cv2.VideoCapture(1)")
        exit()

    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        probs, idx = classifier.getPrediction(frame, draw=True, scale=1.7, color=(0, 255, 100))
        conf = probs[idx] * 100
        print(f"→ {classifier.get_label(idx)} | Confidence: {conf:.1f}%")

        cv2.imshow("Classifier", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()