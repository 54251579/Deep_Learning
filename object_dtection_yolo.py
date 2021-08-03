import cv2
import numpy as np
import sys

# load files
net = cv2.dnn.readNet("yolo/yolov3.weights", "yolo/yolov3.cfg")
classes = []

# yolo network reconstruction
with open("yolo/coco.names", "r") as f:
    classes = [i.strip() for i in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# webcam
cap = cv2.VideoCapture(0)

# create window
cv2.namedWindow("yolo")
# create and set track bar
cv2.createTrackbar("minConfidence", "yolo", 0, 100, lambda x:x)
cv2.setTrackbarPos("minConfidence", "yolo", 50)

while True:
    _, frame = cap.read()

    if _ is None:
        sys.exit()

    height, width, color = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    min_confidence = cv2.getTrackbarPos("minConfidence", "yolo")/100

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > min_confidence:
                # get center point
                centerX = int(detection[0]*width)
                centerY = int(detection[1]*height)

                # get rectengle's widht, height
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                # get rectengle's coordinate
                x = int(centerX - w / 2)
                y = int(centerY - h / 2)

                # append elements
                boxes.append((x, y, w, h))
                confidences.append(float(confidence))
                class_ids.append(class_id)

                # noise filtering
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            # get coordinate
            x, y, w, h = boxes[i]
            # get name of object
            label = str(classes[class_ids[i]])
            # get score
            score = confidences[i]

            # draw text and rextengle
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,255,0), 5)
            cv2.putText(frame, label, (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 1)

    cv2.imshow("yolo", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()