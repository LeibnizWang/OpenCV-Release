import numpy as np
import cv2 as cv

cap1 = cv.VideoCapture(0)
cap2 = cv.VideoCapture(1)
MIN_MATCH_COUNT = 10
# Initiate SIFT detector
sift = cv.xfeatures2d.SIFT_create()

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv.FlannBasedMatcher(index_params, search_params)

while True:
    # Capture frame-by-frame
    ret1, frame1 = cap1.read()
    gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)# trainImage
    ret2, frame2 = cap2.read()
    gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)# queryImage

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(frame1, None)
    kp2, des2 = sift.detectAndCompute(frame2, None)
    matches = flann.knnMatch(des1, des2, k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        h, w, d = frame1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv.perspectiveTransform(pts, M)
        frame2 = cv.polylines(frame2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None
    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)
    img = cv.drawMatches(frame1, kp1, frame2, kp2, good, None, **draw_params)
    cv.imshow('dst', img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap1.release()
cap2.release()
cv.destroyAllWindows()



