import cv2

# use of webcam to capture image of sudoku
# opening webcam
cap = cv2.VideoCapture(0)
while(1):
    # reading frame from webcam
    # ret is a boolean variable, returns true if frame is available
    # frame is an image array vector captured based on the default frames per second
    ret, frame = cap.read()
    # converting image to grayscale
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # applying blurs to remove noise
    blur = cv2.GaussianBlur(img, (55, 55), 3)
    # applying adaptive thresholding to obtain binary image
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    cv2.imshow('thresh', thresh)

    # contour - a curve joining all the continuous points (along the boundary), having same color or intensity

    # finding contours in image
    # contours variable will store Python list of all contours in the image.
    # Each individual contour is a Numpy array of (x,y) coordinates of boundary points of the object.
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    # max_contour -> contour having maximum area
    max_contour = [0]
    # our sudoku grid will be a polygon having 4 sides and maximum area among all the polygons.
    # loop to find out sudoku grid
    for current_contour in contours:
        area = cv2.contourArea(current_contour)
        if area > 50:
            perimeter = cv2.arcLength(current_contour, True)
            # cv2.approxPolyDP() function with a precision factor is applied to approximate/determine the shape of the polygon.
            approximation = cv2.approxPolyDP(current_contour, 0.02 * perimeter, True)

            if area > max_area and len(approximation) == 4:
                max_area = area
                max_contour = approximation

    # if contour have 4 corners, sudoku grid is found.
    if len(max_contour) == 4:
        # determined shape of contour is drawn on the image.
        cv2.drawContours(frame, [max_contour], -1, (255, 0, 2), 5)
        cv2.drawContours(frame, max_contour, -1, (0, 255,0), 8)

    # displaying contour edges and corners
    cv2.imshow('all', frame)
    cv2.waitKey(1)

    # saving the image i.e., our desired sudoku puzzle.
    cv2.imwrite("Resources/frame.jpg", frame)
    break
    # for closing the camera enter "q", if somehow camera is not get closed automatically.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()