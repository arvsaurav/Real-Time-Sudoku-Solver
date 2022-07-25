import cv2
import numpy as np

# 1. Preprocessing image

def preProcess(frame):
    # converting image to grayscale
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # applying blurs to remove noise
    blur = cv2.GaussianBlur(img, (55, 55), 3)
    # applying adaptive thresholding to obtain binary image
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    return thresh

# 2. Finding the biggest contour assuming that is the sudoku puzzle

def biggestContour(contours):
    # maxContour -> will store all the points of biggest contour
    maxContour = np.array([])
    # maxArea -> will store maximum area of the above biggest contour
    maxArea = 0
    # our sudoku grid will be a polygon having 4 sides and having maximum area among all the polygons.
    # loop to find out sudoku grid
    for currentContour in contours:
        area = cv2.contourArea(currentContour)
        if area > 50:
            perimeter = cv2.arcLength(currentContour, True)
            # cv2.approxPolyDP() function with a precision factor is applied to approximate/determine the shape of the polygon.
            # cv2.approxPolyDP() returns set of (x, y) points.
            # i.e., if the polygon is square or rectangle, cv2.approxPolyDp() will return 4 sets of (x,y) coordinates in numpy array
            approximation = cv2.approxPolyDP(currentContour, 0.02 * perimeter, True)

            if area > maxArea and len(approximation) == 4:
                maxArea = area
                maxContour = approximation
    return maxContour, maxArea

# 3. Reordering points for warp perspective

# arranging our four points in top-left, top-right, bottom-left, and bottom-right order
def reorder(biggest):
    biggest = biggest.reshape((4, 2))
    newBiggest = np.zeros((4, 1, 2), dtype=np.int32)
    add = biggest.sum(1)
    newBiggest[0] = biggest[np.argmin(add)]
    newBiggest[3] = biggest[np.argmax(add)]
    diff = np.diff(biggest, axis=1)
    newBiggest[1] = biggest[np.argmin(diff)]
    newBiggest[2] = biggest[np.argmax(diff)]
    return newBiggest

