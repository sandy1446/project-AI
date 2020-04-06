import cv2
import numpy as np
import time
x=[]
y=[]
x=(240,240)
y=(120,90)
x1=90
y1=45
while 1:
    print("\n",x[0],y[0],x[1],y[1])
    a=y[1]-y[0]
    b=x[0]-x[1]
    c=(x[0]*(y[0]-y[1])-y[0]*(x[0]-x[1]))
    print(x[0],y[0])
    res=a*x1+b*y1+c
    print("\n",x1,y1)
    print("\n",a,b,c)
    print("\n",res)
    time.sleep(0.3)
    if res<0:
        s='in'
    else:
        s='out'
    print(s)
    x1=x1+10
    y1=y1+10
    k=cv2.waitKey(50) & 0xff
    if k==27:
        break
cv2.release()
cv2.destroyAllWindows()

