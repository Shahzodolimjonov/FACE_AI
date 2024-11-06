import cv2

def blur_face(img):
    (h,w) = img.shape[:2]
    dW = int(w/3.0)
    dH = int(h/3.0)
    if dW % 2 == 0:
        dW -= 1
    if dH % 2 == 0:
        dH -= 1
    return cv2.GaussianBlur(img, (dW,dH), 0)


face_cascade_db = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
eye_cascade_db = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_eye.xml")
capture = cv2.VideoCapture(0)

while True:
    ret, img = capture.read()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade_db.detectMultiScale(img_gray, 1.1, 19) 
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        img[y:y+h, x:x+w] = blur_face(img[y:y+h, x:x+w])

# eyes
        # img_gray_face = img_gray[y:y+h,x:x+w]
        # eyes = eye_cascade_db.detectMultiScale(img_gray_face, 1.1, 19)
        # for (ex, ey, ew, eh) in eyes:
        #     cv2.rectangle(img, (x+ex, y+ey), (x+ex + ew, y+ey + eh), (0, 0, 255), 2)

# 
    cv2.imshow("Camera", img)

    k = cv2.waitKey(10)
    if cv2.waitKey(10) & 0xFF == ord('x'):
        break

capture.release()
cv2.destroyAllWindows()