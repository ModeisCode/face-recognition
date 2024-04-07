import cv2
import dlib
import face_recognition

detector = dlib.get_frontal_face_detector()

imgs_encoded = []

def load_new_img(path: str,index=0):
    global imgs_encoded
    new_img = face_recognition.load_image_file(path)
    new_img_encoded = face_recognition.face_encodings(new_img)[index]
    imgs_encoded.append(new_img_encoded)

load_new_img("1.jpg")
load_new_img("2.jpg")
load_new_img("3.jpg")
load_new_img("4.jpg")
load_new_img("5.jpg")
load_new_img("6.jpg")
load_new_img("7.jpg")
load_new_img("8.jpg")
load_new_img("9.jpg")


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    face_loc = []
    faces = detector(frame)
    for face in faces:
        x = face.left()
        y = face.top()
        w = face.right()
        h = face.bottom()
        face_loc.append((y,w,h,x))

    #face_loc = face_recognition.face_locations(frame)
    face_encoded = face_recognition.face_encodings(frame,face_loc)
    
    i = 0
    for face in face_encoded:
        y,w,h,x = face_loc[i]
        res = face_recognition.compare_faces(imgs_encoded,face)
        if res[0] == True:
            cv2.rectangle(frame,(x,y),(w,h),(255,0,0),2)
            cv2.putText(frame,"ADMIN",(x,h+35),cv2.FONT_HERSHEY_PLAIN,0.9,(200,0,0),2)
        else:
            cv2.rectangle(frame,(x,y),(w,h),(255,0,0),2)
            cv2.putText(frame,"ANY",(x,h+35),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)
        
    cv2.imshow("1",frame)
    if cv2.waitKey(10) & 0xFF==ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
