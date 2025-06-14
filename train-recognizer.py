import cv2
import os
import numpy as np
from PIL import Image
import pickle

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "dataset")

current_id = 0
label_ids = {}
x_train = []
y_labels = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(root).lower()
            if label not in label_ids:
                label_ids[label] = current_id
                current_id += 1

            id_ = label_ids[label]
            pil_image = Image.open(path).convert("L")
            image_array = np.array(pil_image, "uint8")
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.2, minNeighbors=5)

            for (x, y, w, h) in faces:
                roi = image_array[y:y + h, x:x + w]
                x_train.append(roi)
                y_labels.append(id_)

# Save labels
with open("labels.pkl", "wb") as f:
    pickle.dump(label_ids, f)

# Train model
recognizer.train(x_train, np.array(y_labels))
recognizer.save("face_trained.yaml")

print("✅ Training complete. Files saved: face_trained.yaml, labels.pkl")
