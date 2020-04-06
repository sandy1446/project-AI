import cv2
import numpy as np
import time
import cython
def eqn(x1,y1,x,y):
    global res
    print("\n",x[0],y[0],x[1],y[1])
    a=y[1]-y[0]
    b=x[0]-x[1]
    c=(x[0]*(y[0]-y[1])-y[0]*(x[0]-x[1]))
    print(x[0],y[0])
    res=a*x1+b*y1+c
    print("\n",x1,y1)
    print("\n",a,b,c)
    print("\n",res)
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
cm=[]
m=0
g=0
l=1  
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
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                m+=1
    cm.append(m)
    #print(cm)
    #print(cx)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    #counting individuals
    count_person=0
    count_car=0
    count_bus=0
    count_truck=0
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[30]
            print(g,cx[g],cy[g])
     
    k_1=cm[g]
    k_2=cm[g-l]
    for i in range(cm[g]):
        if g>6:
            l=5
        if i>k_2 and i<k_1:
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[60]
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
                cv2.line(img,(cx[i],cy[i]),(cx[i],cy[i]), (0,255,255),1)
    k_1=k_1+1
    k_2=k_2+1
    g=g+1
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
    print(height)
    print(width)
cv2.release()
cv2.destroyAllWindows()
