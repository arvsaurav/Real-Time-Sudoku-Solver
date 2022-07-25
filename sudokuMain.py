import cv2
import sudokuFunctions

# use of webcam to capture image of sudoku

# opening webcam
cap = cv2.VideoCapture(0)
# result is a boolean variable, returns true if frame is available
# frame is an image array vector captured based on the default frames per second
result, frame = cap.read()
if result:
    # displaying image
    cv2.imshow('capturedFrame', frame)
    # saving the image
    cv2.imwrite("Resources/frame.jpg", frame)
    # waitKey(0) will display the window infinitely until any keypress
    cv2.waitKey(0)
    # If keyboard interrupt occurs, destroy image window
    cv2.destroyWindow('capturedFrame')
else:
    print('No image detected. Please! Try again')

cv2.destroyAllWindows()
cap.release()

capturedImage = "Resources/frame.jpg"
heightImg = 450
widthImg = 450

# 1. Preparing the image

img = cv2.imread(capturedImage)
# resize image to make it a square image
img = cv2.resize(img, (widthImg, heightImg))
imgThreshold = sudokuFunctions.preProcess(img)

# 2. Finding all contours

# contour - a curve joining all the continuous points (along the boundary), having same color or intensity

# copying image for displaying purpose
imgContour = img.copy()
imgBigContour = img.copy()
# contours variable will store Python list of all contours in the image.
# Each individual contour is a Numpy array of (x,y) coordinates of boundary points of the object.
contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
# draw all detected contours
cv2.drawContours(imgContour, contours, -1, (0, 225, 0), 3)

# 3. Detecting sudoku grid using biggest contour

# finding the biggest contour
# biggest -> stores biggest contour i.e., it stores 4 coordinates that forms the biggest contour
# maxArea -> stores area of the biggest contour
biggest, maxArea = sudokuFunctions.biggestContour(contours)
print("Coordinates of biggest contours are : " + biggest)
if biggest.size != 0:
    # reordering points
    biggest = sudokuFunctions.reorder(biggest)
    print("Coordinates of biggest contours after reordering are : " + biggest)
    
else:
    print("Sudoku Not Found!")