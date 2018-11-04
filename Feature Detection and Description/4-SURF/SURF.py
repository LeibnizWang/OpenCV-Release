import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)
# Create SURF object. You can specify params here or later.
# Here I set Hessian Threshold to 400
# We set it to some 5000. Remember, it is just for representing in picture.
# In actual cases, it is better to have a value 300-500
surf = cv.xfeatures2d.SURF_create(5000)
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Find keypoints and descriptors directly
    kp, des = surf.detectAndCompute(gray, None)
    frame = cv.drawKeypoints(frame, kp, None,(255,0,0),4)
    cv.imshow('dst', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()



