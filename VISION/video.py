
import cv2

def blur_face(img):
    (h,w)=img.shape[:2]
    dW=int(w/3.0)
    dH=int(h/3.0)
    if dW % 2 == 0:
        dW-= 1
    if dH % 2 == 0:
        dH-= 1
    return cv2.GaussianBlur(img,(dW,dH),0)

capture = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while True:
    Return, img = capture.read(0)

    faces = face_cascade.detectMultiScale(img, scaleFactor=2 ,minNeighbors=5,minSize=(20,20))

    for(x,y,w,h) in faces:
        #cv2.rectangle(img, (x,y),(x+w, y+w),(255,0,0),2)
        img[y:y+h,x:x+h] = blur_face(img[y:y+h,x:x+h])

    cv2.imshow ('from camera',img)
    k = cv2.waitKey(30)
    if k == 27:
        break

capture.release()
cv2.destroyAllWindows()
