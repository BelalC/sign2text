# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 16:57:28 2016
@author: Karthick Perumal
"""

import cv2
import numpy as np

## Our sketch generating functuion
def sketch(image):
    #convert image to grayscale
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Clean up image using Gaussian Blur
    img_gray_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    
    # Extract edges
    canny_edges = cv2.Canny(img_gray_blur, 25, 50) ## Feel free to change the thresholds depending on lighting conditions
    
    ## Do an invert binarize on the image
    ret, mask = cv2.threshold(canny_edges, 70, 255, cv2.THRESH_BINARY_INV) ## can play with the threshold parameter. In principle we could use other threshold functions
    return mask
    
### Initialize webcam, cap if the object provided by VideoCapture
### It contains a boolean indicating if it was successful (ret)
### It also contains the images collected from the webcam (frame)
infile = "/Users/Belal/Desktop/scene2-camera1.mov"
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.imshow('Our Live Sketcher', sketch(frame))
    if cv2.waitKey(1) == 13: ### 13 is the Enter Key
        break

## Release camera and close windows
cap.release()   # When using a webcam, it is necessary to do cap.release() to close the videocapture
cv2.destroyAllWindows()