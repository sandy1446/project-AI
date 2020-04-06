import cv2
import numpy as np

img=cv2.imread('E:\python37\img.jpg',cv2.IMREAD_COLOR)

cv2.line(img,(0,0),(150,150),(0,0,255),1)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
 
