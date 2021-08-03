import cv2
import numpy as np
import sys

def DNN_detectAndDisplay(frame):
    width = frame.shape[1]
    height = frame.shape[0]
    min_confidence = cv2.getTrackbarPos("minConfidence", "dnn face")/100
    
    model = 'res10_300x300_ssd_iter_140000.caffemodel'
    prototxt = 'deploy.prototxt.txt'

    model = cv2.dnn.readNetFromCaffe(prototxt, model)

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300,300), (104, 177, 123))

    model.setInput(blob)
    detections = model.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]

        if confidence > min_confidence:
            box = detections[0,0,i,3:7]*np.array([width, height, width, height])
            (startX, startY, endX, endY) = box.astype("int")

            txt = "{:.2f}%".format(confidence*100)
            y = startY - 10 if startY - 10 > 10 else startY + 10

            cv2.rectangle(frame, (startX, startY), (endX, endY), (0,255,0), 2)
            cv2.putText(frame, txt, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    cv2.imshow("dnn face", frame)

cap = cv2.VideoCapture(0)

cv2.namedWindow("dnn face")

cv2.createTrackbar("minConfidence", "dnn face", 0, 100, lambda x:x)
cv2.setTrackbarPos("minConfidence", "dnn face", 50)

while True:
    _, frame = cap.read()

    if _ is None:
        sys.exit()

    cv2.imshow('frame', frame)
    DNN_detectAndDisplay(frame)

    key = cv2.waitKey(3)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()