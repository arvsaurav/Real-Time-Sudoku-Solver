import cv2

# preprocessing image

def preProcess(frame):
    # converting image to grayscale
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # applying blurs to remove noise
    blur = cv2.GaussianBlur(img, (55, 55), 3)
    # applying adaptive thresholding to obtain binary image
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    return thresh
