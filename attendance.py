import cv2
import os
import csv
from datetime import datetime

# Load trained model
model = cv2.face.LBPHFaceRecognizer_create()
model.read("face_model.yml")

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Map labels to names
image_dir = "Images"
label_map = {}
label_id = 0

for person_name in os.listdir(image_dir):
    label_map[label_id] = person_name
    label_id += 1

# Attendance file
attendance_file = "attendance.csv"

if not os.path.exists(attendance_file):
    with open(attendance_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Time", "Date"])

marked = set()

# Open camera
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Camera not opening")
    exit()

while True:
    ret, frame = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))

        label, confidence = model.predict(face)

        if confidence < 110:
            name = label_map[label]

            if name not in marked:
                now = datetime.now()
                with open(attendance_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        name,
                        now.strftime("%H:%M:%S"),
                        now.strftime("%d-%m-%Y")
                    ])
                marked.add(name)

            cv2.putText(frame, name, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        else:
            cv2.putText(frame, "Unknown", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

    cv2.imshow("Smart Attendance", frame)

    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()
