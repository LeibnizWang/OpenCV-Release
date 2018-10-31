import numpy as np
import cv2 as cv

cap = cv.VideoCapture(1)
# Initiate ORB detector
orb = cv.ORB_create()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # find the keypoints with ORB
    kp = orb.detect(frame, None)
    # compute the descriptors with ORB
    kp, des = orb.compute(frame, kp)
    # draw only keypoints location,not size and orientation
    frame2 = cv.drawKeypoints(frame, kp, None, color=(0, 255, 0), flags=0)
    cv.imshow('dst1', frame)
    cv.imshow('dst2', frame2)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()



