import numpy as np
import cv2 as cv

cap = cv.VideoCapture(1)
# Initiate FAST object with default values
fast = cv.FastFeatureDetector_create()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # find and draw the keypoints
    kp = fast.detect(frame, None)
    frame2 = cv.drawKeypoints(frame, kp, None, color=(255, 0, 0))

    # Print all default params
    print("Threshold: {}".format(fast.getThreshold()))
    print("nonmaxSuppression:{}".format(fast.getNonmaxSuppression()))
    print("neighborhood: {}".format(fast.getType()))
    print("Total Keypoints with nonmaxSuppression: {}".format(len(kp)))
    cv.imshow('dst2', frame2)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()



