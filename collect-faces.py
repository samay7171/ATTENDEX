import cv2
import os

name = input("Enter your name: ").strip()
folder = os.path.join("dataset", name)
os.makedirs(folder, exist_ok=True)

cam = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

count = 0
while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        count += 1
        cv2.imwrite(f"{folder}/{count}.jpg", gray[y:y+h, x:x+w])
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Collecting Faces", frame)
    if cv2.waitKey(1) == 27 or count >= 30:
        break

cam.release()
cv2.destroyAllWindows()
