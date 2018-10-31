import numpy as np
import cv2 as cv

cap = cv.VideoCapture(1)
# Initiate FAST detector
star = cv.xfeatures2d.StarDetector_create()
# Initiate BRIEF extractor
brief = cv.xfeatures2d.BriefDescriptorExtractor_create()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # find the keypoints with STAR
    kp = star.detect(frame, None)
    # compute the descriptors with BRIEF
    kp, des = brief.compute(frame, kp)
    print(brief.descriptorSize())
    #print(des.shape)
    frame = cv.drawKeypoints(frame, kp, None, color=(0, 255, 0))
    cv.imshow('dst2', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()



