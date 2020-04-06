import cv2
import numpy as np
import time
import cython

# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image
video_src='video3.mp4'
cap = cv2.VideoCapture(video_src)
class_ids = []
confidences = []
boxes = []
cx=[]
cy=[]
a=[]
center_x=[]
center_y=[]
m=0
g=0
while 1:
    ret,img=cap.read()
    scale_percent=200
    width=int(img.shape[1]*scale_percent/100)
    height=int(img.shape[0]*scale_percent/100)
    dim=(width,height)
    img=cv2.resize(img,dim)
    height,width,channels=img.shape
    
    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00192, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
        
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = (int(detection[0] * width))
                center_y = (int(detection[1] * height))
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                cx.append(center_x)
                cy.append(center_y)
                print(m,center_x,center_y)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                m+=1
    #print(cx,"  ",cy)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    #counting individuals
    count_car=0
    count_bus=0
    count_truck=0
    count_person=0
    count=0
    for i in range(len(boxes)):
        count+=1
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[50]
            if label=="car":
                count_car+=1
            if label=="bus":
                count_bus+=1
            if label=="truck":
                count_truck+=1
            if label=="person":
                count_person+=1
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
            cv2.putText(img, label, (x, y + 30), font, 1, color, 1)
    print(count)
    cx.clear()
    tot_car="car : "+str(count_car)
    tot_truck="truck : "+str(count_truck)
    tot_person="person : "+str(count_person)
    cv2.putText(img,"Counting System:",(width-180,10),font, 1, (0,0,255),1)
    j=10;
    k=10;
    if count_car>0:
        j=j+k
        cv2.putText(img,tot_car,(width-180,j),font, 1, (0,0,255),1)
    if count_truck>0:
        j=j+k
        cv2.putText(img,tot_truck,(width-180,j),font, 1, (0,0,255),1)
    if count_person>0:
        j=j+k
        cv2.putText(img,tot_person,(width-180,j),font, 1, (0,0,255),1)
    cv2.line(img,(0,int(height*(1/5))),(int(width),int(height*(1/5))),(0,0,255),1)
    cv2.line(img,(0,int(height*(2/5))),(int(width),int(height*(2/5))),(0,0,255),1)
    cv2.line(img,(0,int(height*(3/5))),(int(width),int(height*(3/5))),(0,0,255),1)
    cv2.line(img,(0,int(height*(4/5))),(int(width),int(height*(4/5))),(0,0,255),1)
    cv2.line(img,(0,int(height*(5/5))),(int(width),int(height*(5/5))),(0,0,255),1)
    cv2.line(img,(int(width*(1/5)),0),(int(width*(1/5)),int(height)),(0,0,255),1)
    cv2.line(img,(int(width*(2/5)),0),(int(width*(2/5)),int(height)),(0,0,255),1)
    cv2.line(img,(int(width*(3/5)),0),(int(width*(3/5)),int(height)),(0,0,255),1)
    cv2.line(img,(int(width*(4/5)),0),(int(width*(4/5)),int(height)),(0,0,255),1)
    #cv2.line(img,(int(width*(0.9/5)),int(height*(3.1/5))),(int(width*(2.2/5)),int(height*(3.2/5))),(0,255,255),1)
    #cv2.line(img,(int(width*(2.7/5)),int(height*(3.2/5))),(int(width*(4.4/5)),int(height*(3/5))),(0,255,255),1)
    cv2.imshow("Image", img)
    k=cv2.waitKey(50) & 0xff
    if k==27:
        break
cv2.release()
cv2.destroyAllWindows()
