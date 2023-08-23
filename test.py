import cv2
import face_recognition
import numpy as np

cam = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
names = ["Neil Peter"]
name_face = {"Neil Peter" : "faces/neil.jpg"}
encodings = []

for name in names:
    face = cv2.imread(name_face[name])
    encodings.append(np.array(face_recognition.face_encodings(face)[0]))

while(cam.isOpened()):
    ret, frame = cam.read()
    if ret:
        faces = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, faces)
        i = 0
        for (t, r, b, l) in faces:
            cv2.rectangle(frame, (l, t), (r, b), (255, 0, 0), 2)
            face = frame[t:b, l:r]
            cv2.imshow("Face", face)
            enc = face_encodings[i]
            nm = "?"
            matches = face_recognition.compare_faces(encodings, enc)
            if True in matches:
                ind = matches.index(True)
                nm = names[ind]
            frame = cv2.putText(frame, nm, (l, t), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            i += 1

        cv2.imshow("Video", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cam.release()
cv2.destroyAllWindows()