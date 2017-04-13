import numpy as np
import cv2

infile = "/Users/Belal/PycharmProjects/sign2text/train_data/a_pouliot/Alphabet_front.mov"

# define the upper and lower boundaries of the HSV pixel
# intensities to be considered 'skin'
lower = np.array([0, 48, 80], dtype = "uint8")
upper = np.array([20, 255, 255], dtype = "uint8")

""""
    # take first frame of the video
    ret,frame = cap.read()
    # setup initial location of window
    r,h,c,w = 250,90,400,125  # simply hardcoded the values
    track_window = (c,r,w,h)

    # set up the ROI for tracking
    #roi = frame[r:r+h, c:c+w]
    ##hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    #mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
    #roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])


    width = int(cap.get(3) + 0.5)
    height = int(cap.get(4) + 0.5)
    frame = cv2.resize(frame, (width//2, height//2))
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(converted, lower, upper)

    # apply a series of erosions and dilations to the mask
    # using an elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    #skinMask = cv2.erode(skinMask, kernel, iterations=2)
    #skinMask = cv2.dilate(skinMask, kernel, iterations=2)
    skinMask = cv2.morphologyEx(skinMask, cv2.MORPH_OPEN, kernel)

    # blur the mask to help remove noise, then apply the
    # mask to the frame
    #skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    skin = cv2.bitwise_and(frame, frame, mask=skinMask)

    # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

    # apply meanshift to get the new location
    ret, track_window = cv2.meanShift(dst, track_window, term_crit)

    # Draw it on image
    x,y,w,h = track_window
    img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)

    # Display the resulting frame
    cv2.imshow('img2', img2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    """""

cap = cv2.VideoCapture(0)
while True:

    ret, frame = cap.read()
    # grab the current frame
    (grabbed, frame) = cap.read()

    # resize the frame, convert it to the HSV color space,
    # and determine the HSV pixel intensities that fall into
    # the specified upper and lower boundaries
    # Get the width and height of frame
    width = int(cap.get(3) + 0.5)
    height = int(cap.get(4) + 0.5)
    frame = cv2.resize(frame, (width//2, height//2))
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(converted, lower, upper)

    # apply a series of erosions and dilations to the mask
    # using an elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinMask = cv2.erode(skinMask, kernel, iterations=2)
    skinMask = cv2.dilate(skinMask, kernel, iterations=2)
    #skinMask = cv2.morphologyEx(skinMask, cv2.MORPH_OPEN, kernel)

    # blur the mask to help remove noise, then apply the
    # mask to the frame
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    skin = cv2.bitwise_and(frame, frame, mask=skinMask)

    imgray = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cnt = contours[0]
    #cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
    # show the skin in the image along with the mask
    if len(contours) > 0:
        # find largest contour in mask, use to compute minEnCircle
        c = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        x_ = int(x-radius)
        y_ = int(y-radius)
        h_ = int(radius*2)
        w_ = int(radius*2)
        cv2.rectangle(frame, (x_, y_), (x_ + w_, y_ + h_), (0, 255, 0), 2)

    cv2.imshow("images", np.hstack([frame, skin]))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()