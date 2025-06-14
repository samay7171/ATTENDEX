import cv2
import numpy as np

# Load trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_trained.yml")

# Load labels
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Load face detection model
prototxt_path = "deploy.prototxt"
model_path = "res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Start webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print("✅ Recognition started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0), swapRB=False, crop=False)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (startX, startY, endX, endY) = box.astype("int")

            face = frame[startY:endY, startX:endX]

            if face.size == 0:
                continue

            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            try:
                face = cv2.resize(face, (200, 200))
            except:
                continue

            label, confidence_rec = recognizer.predict(face)
            name = labels[label]

            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} ({confidence_rec:.0f})", (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
