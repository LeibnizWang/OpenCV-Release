import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)
sift = cv.xfeatures2d.SIFT_create()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    kp = sift.detect(gray, None)
    #frame = cv.drawKeypoints(gray, kp, frame)
    frame = cv.drawKeypoints(gray, kp, frame,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv.imshow('dst', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()



