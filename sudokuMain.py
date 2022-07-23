import cv2

# use of webcam to capture image of sudoku

# opening webcam
cap = cv2.VideoCapture(0)
# result is a boolean variable, returns true if frame is available
# frame is an image array vector captured based on the default frames per second
result, frame = cap.read()
if result:
    # displaying image
    cv2.imshow('all', frame)
    # saving the image
    cv2.imwrite("Resources/frame.jpg", frame)
    # If keyboard interrupt occurs, destroy image window
    cv2.waitKey(0)
    cv2.destroyWindow('all')
else:
    print('No image detected. Please! Try again')

cv2.destroyAllWindows()
cap.release()

capturedImage = "Resources/frame.jpg"
heightImg = 450
widthImg = 450

# preparing the image

img = cv2.imread(capturedImage)
# resize image to make it a square image
img = cv2.resize(img, (widthImg, heightImg))
