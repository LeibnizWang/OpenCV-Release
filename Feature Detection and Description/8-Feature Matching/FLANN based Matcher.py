import numpy as np
import cv2 as cv

cap1 = cv.VideoCapture(1)
cap2 = cv.VideoCapture(0)

# Initiate SIFT detector
sift = cv.xfeatures2d.SIFT_create()
# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv.FlannBasedMatcher(index_params,search_params)

while True:
    # Capture frame-by-frame
    ret1, frame1 = cap1.read()
    gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)# trainImage
    ret2, frame2 = cap2.read()
    gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)# queryImage
    # find the keypoints and descriptors with ORB
    kp1, des1 = sift.detectAndCompute(frame1, None)
    kp2, des2 = sift.detectAndCompute(frame2, None)
    # Match descriptors.
    matches = flann.knnMatch(des1, des2, k=2)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in xrange(len(matches))]
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=0)
    img = cv.drawMatchesKnn(frame1, kp1, frame2, kp2, matches, None, **draw_params)
    cv.imshow('dst', img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap1.release()
cap2.release()
cv.destroyAllWindows()



