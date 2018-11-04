import numpy as np
import cv2 as cv

cap1 = cv.VideoCapture(1)
cap2 = cv.VideoCapture(0)

# Initiate SIFT detector
sift = cv.xfeatures2d.SIFT_create()
# BFMatcher with default params
bf = cv.BFMatcher()

while True:
    # Capture frame-by-frame
    ret1, frame1 = cap1.read()
    gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)# trainImage
    ret2, frame2 = cap2.read()
    gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)# queryImage
    # find the keypoints and descriptors with ORB
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    # Match descriptors.
    matches = bf.knnMatch(des1, des2, k=2)
    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
    # Draw first 10 matches.
    img = cv.drawMatchesKnn(gray1, kp1, gray2, kp2, good, None, flags=2)
    cv.imshow('dst', img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap1.release()
cap2.release()
cv.destroyAllWindows()



