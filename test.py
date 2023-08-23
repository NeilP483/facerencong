import cv2
import face_recognition
import numpy as np

cam = cv2.VideoCapture(0)

#Add file reading
#names = ["Neil Peter", "Sunil Peter"]
#faces_fs = ["faces/neil.jpg", "faces/sunil.jpg"]
names = []
faces_fs = []
with open("name_face.txt") as config:
    for line in config:
        both = line.split('|')
        name = both[0].strip()
        names.append(name)
        fs = both[1].strip()
        faces_fs.append(fs)

encodings = []

for name, face_fs in zip(names, faces_fs):
    face = cv2.imread(face_fs)
    encodings.append(np.array(face_recognition.face_encodings(face)[0]))
faces = []
face_encodings = []
names_for_faces = []
j = 0
while(cam.isOpened()):
    ret, frame = cam.read()
    if ret:
        if j % 5 == 0:
            faces = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(frame, faces)
            names_for_faces = []
            for enc in face_encodings:
                nm = "?"
                matches = face_recognition.compare_faces(encodings, enc)
                if True in matches:
                    ind = matches.index(True)
                    nm = names[ind]
                names_for_faces.append(nm)
        j += 1
        i = 0
        for (t, r, b, l) in faces:
            cv2.rectangle(frame, (l, t), (r, b), (255, 0, 0), 2)
            nm = names_for_faces[i]
            frame = cv2.putText(frame, nm, (l, t), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            i += 1

        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cam.release()
cv2.destroyAllWindows()