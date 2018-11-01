import numpy as np
import cv2 as cv

cap1 = cv.VideoCapture(1)
cap2 = cv.VideoCapture(0)

# Initiate ORB detector
orb = cv.ORB_create()
# create BFMatcher object
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

while True:
    # Capture frame-by-frame
    ret1, frame1 = cap1.read()
    gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)# trainImage
    ret2, frame2 = cap2.read()
    gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)# queryImage
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    # Match descriptors.
    matches = bf.match(des1, des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)
    # Draw first 10 matches.
    img = cv.drawMatches(gray1, kp1, gray2, kp2, matches[:10], None, flags=2)
    cv.imshow('dst', img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap1.release()
cap2.release()
cv.destroyAllWindows()



