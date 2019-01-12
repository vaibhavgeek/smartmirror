import cv2
import numpy as np
# sys
#simple upperbody detection
imgPath = input("image_path")
cascPath = "haarcascade_upperbody.xml"
# Create the haar cascade
upperbodyCascade = cv2.CascadeClassifier(cascPath)
img = cv2.imread(imgPath,0)
upperBody_cascade = cv2.CascadeClassifier('haarcascade_upperbody.xml')    
arrUpperBody = upperBody_cascade.detectMultiScale(img)
if arrUpperBody != ():
        for (x,y,w,h) in arrUpperBody:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        print('body found')

cv2.namedWindow("img",cv2.WINDOW_NORMAL)
cv2.resizeWindow("img", (800, 800))
cv2.imshow("img",img)
cv2.waitKey(0)