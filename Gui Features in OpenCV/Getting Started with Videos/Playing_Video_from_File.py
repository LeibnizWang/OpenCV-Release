import numpy as np
import cv2 as cv
cap = cv.VideoCapture('output.avi')
while(cap.isOpened()):
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow('frame',gray)
    if cv.waitKey(25) & 0xFF == ord('q'):#waitKey用于改变播放速度
        break
cap.release()
cv.destroyAllWindows()