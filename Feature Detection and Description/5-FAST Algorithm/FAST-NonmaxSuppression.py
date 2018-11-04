import numpy as np
import cv2 as cv

cap = cv.VideoCapture(1)
# Initiate FAST object with default values
fast = cv.FastFeatureDetector_create()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Disable nonmaxSuppression
    fast.setNonmaxSuppression(0)
    kp = fast.detect(frame, None)
    print("Total Keypoints without nonmaxSuppression: {}".format(len(kp)))
    frame3 = cv.drawKeypoints(frame, kp, None, color=(0, 255, 0))
    cv.imshow('dst3', frame3)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()



