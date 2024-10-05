import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime

video_capture = cv2.VideoCapture(0)

reshma_image =face_recognition.load_image_file("reshma.jpg")
reshma_encoding = face_recognition.face_encodings(reshma_image)[0]

prabhas_image = face_recognition.load_image_file("prabhas.jpg")
prabhas_encoding = face_recognition.face_encodings(prabhas_image)[0]

ramcharan_image = face_recognition.load_image_file("ramcharan.jpg")
ramcharan_encoding = face_recognition.face_encodings(ramcharan_image)[0]

sruthi_image = face_recognition.load_image_file("sruthi.jpg")
sruthi_encoding = face_recognition.face_encodings(sruthi_image)[0]

himaja_image = face_recognition.load_image_file("himaja.jpg")
himaja_encoding = face_recognition.face_encodings(himaja_image)[0]

sailaja_image = face_recognition.load_image_file("sailaja.jpg")
sailaja_encoding = face_recognition.face_encodings(sailaja_image)[0]

divya_image = face_recognition.load_image_file("divya.jpg")
divya_encoding = face_recognition.face_encodings(divya_image)[0]
 

known_face_encoding = [
    reshma_encoding,
    prabhas_encoding,
    ramcharan_encoding,
    sruthi_encoding,
   himaja_encoding,
   sailaja_encoding,
    divya_encoding
]

known_faces_names = [
    "reshma",
     "prabhas",
    "ramcharan" 
     "sruthi" ,
    "himaja", 
    "sailaja",
    "divya"
]

students = known_faces_names.copy()

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(current_date+'.csv', 'w+', newline='')
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    face_names = []

    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encoding, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_faces_names[best_match_index]

        face_names.append(name)

        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.7, (255, 255, 255), 1)

        if name in students:
            students.remove(name)
            print(students)
            current_time = now.strftime("%H-%M-%S")
            lnwriter.writerow([name, current_time])

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()

