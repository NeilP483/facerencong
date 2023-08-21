import cv2

cam = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')

while(cam.isOpened()):
    ret, frame = cam.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces  = face_cascade.detectMultiScale(gray, 1.3, 5)
        final = frame
        for (x, y, w, h) in faces:
            final = frame[y:y + h, x:x + w]
        cv2.imshow("Video", final)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
cam.release()
cv2.destroyAllWindows()