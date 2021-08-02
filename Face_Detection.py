import cv2
import numpy as np
import sys

def detectAndDisplay(frame):
    
    # convert to gray scale img
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.equalizeHist(gray_frame)
    
    # detect face
    faces = face_cascade.detectMultiScale(gray_frame)

    for (x, y, w, h) in faces:

        # draw rectangle at area of face
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 3)
        
        # extract area of ROI
        faceROI = gray_frame[y:y+h, x:x+w]

        # detect eyes
        eyes = eyes_cascade.detectMultiScale(faceROI)

        for (eyes_x, eyes_y, eyes_w, eyes_h) in eyes:
            # get center point and radius
            eyes_center = (x++eyes_x+eyes_w//2, y+eyes_y+eyes_h//2)
            radius = int(round(eyes_w + eyes_h)*0.25)
            # draw rectangle at area of face
            frame = cv2.circle(frame, eyes_center, radius, (255,255,0), 3)
            
    cv2.imshow('recognition', frame)
            
cap = cv2.VideoCapture(0)

face_cascade_name = 'C:/opencv/sources/data/haarcascades/haarcascade_frontalface_alt.xml'
eyes_cascade_name = 'C:/opencv/sources/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml'

face_cascade = cv2.CascadeClassifier()
eyes_cascade = cv2.CascadeClassifier()

if not face_cascade.load(face_cascade_name):
    print("haarcascade_frontalface_alt load failed")
    sys.exit()

if not eyes_cascade.load(eyes_cascade_name):
    print("haarcascade_eye_tree_eyeglasses load failed")
    sys.exit()

while True:
    _, frame = cap.read()

    if _ is None:
        sys.exit()

    detectAndDisplay(frame)

    key = cv2.waitKey(3)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()