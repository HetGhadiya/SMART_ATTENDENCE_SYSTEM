import cv2
import os
import numpy as np

data_path = "Images"

faces = []
labels = []
label_map = {}
label_id = 0

for person in os.listdir(data_path):
    person_path = os.path.join(data_path, person)

    if not os.path.isdir(person_path):
        continue

    label_map[label_id] = person

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        faces.append(img)
        labels.append(label_id)

    label_id += 1

faces = np.array(faces)
labels = np.array(labels)

model = cv2.face.LBPHFaceRecognizer_create()
model.train(faces, labels)

model.save("face_model.yml")

print("✅ Training completed successfully")
