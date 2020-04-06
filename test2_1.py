import cv2
import numpy as np
import time
count=0
def eqn(x1,y1,x,y):
    global res
    j=0
    print("\n",x[0],y[0],x[1],y[1])
    a=y[1]-y[0]
    b=x[0]-x[1]
    c=(x[0]*(y[0]-y[1])-y[0]*(x[0]-x[1]))
    print(x[0],y[0])
    res=a*x1+b*y1+c
    print("\n",x1,y1)
    print("\n",a,b,c)     
img=cv2.imread('room_ser.jpg')
width=int(img.shape[1]*0.2)
height=int(img.shape[0]*0.2)
img=cv2.resize(img,(width,height))
cv2.imshow("Image",img)
cv2.line(img,(120,0),(120,690),(0,0,255),2)
m=0
n=0
x1=90
y1=45
x=(240,240)
y=(120,90)
j=0
font=cv2.FONT_HERSHEY_PLAIN
while 1:
    i=eqn(x1,y1,x,y)
    cv2.line(img,(120,0),(120,690),(0,0,255),2)
    cv2.line(img,(x[0],y[0]),(x[1],y[1]),(0,255,0),2)
    time.sleep(0.4)
    if res<1:
        s="in"
        m=m+1
    else:
        s="out"
        n=n+1
    print("\n",s)
    x1=x1+10
    y1=y1+10
    j=j+1
    time.sleep(0.4)
    print("hlo")
    if m>2 and n>2:
        count=1
    k=cv2.waitKey(50) & 0xff
    if k==27:
        break
    tot_count="count ="+str(count)
    cv2.putText(img,tot_count,(width-60,10),font,1,(0,0,255),1)
cv2.release()
cv2.destroyAllWindows()

