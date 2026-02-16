import cv2
import os

name = input("Enter Student Name: ")
path = f"Images/{name}"

if not os.path.exists(path):
    os.makedirs(path)

cam = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cam.read()
    cv2.imshow("Capture Face", frame)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite(f"{path}/{count}.jpg", frame)
        count += 1
        print("Image saved")

    if count == 20:
        break

cam.release()
cv2.destroyAllWindows()
