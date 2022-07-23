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
    maxArea = 0
    # maxContour -> contour having maximum area
    maxContour = np.array([])
    # our sudoku grid will be a polygon having 4 sides and maximum area among all the polygons.
    # loop to find out sudoku grid
    for currentContour in contours:
        area = cv2.contourArea(currentContour)
        if area > 50:
            perimeter = cv2.arcLength(currentContour, True)
            # cv2.approxPolyDP() function with a precision factor is applied to approximate/determine the shape of the polygon.
            approximation = cv2.approxPolyDP(currentContour, 0.02 * perimeter, True)

            if area > maxArea and len(approximation) == 4:
                maxArea = area
                maxContour = approximation
    return maxContour, maxArea