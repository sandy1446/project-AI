import cv2
import numpy as np
import time
global center
center=[]
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
img=cv2.imread("room_ser.jpg")
cv2.imshow("Image",img)
j=0
x1=90
y1=45
x=(240,240)
y=(120,90)
while 1:
    i=eqn(x1,y1,x,y)
    cv2.line(img,(120,0),(120,690),(0,0,255),2)
    cv2.line(img,(x[0],y[0]),(x[1],y[1]),(0,255,0),2)
    if res<0:
        s="in"
    else:
        s="out"
    print("\n",s)
    
    k=cv2.waitKey(50) & 0xff
    if k==27:
        break
    x1=x1+10
    y1=y1+10
cv2.release()
cv2.destroyAllWindows()

