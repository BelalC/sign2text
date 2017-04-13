
"""
Script used to generate data from video frames.

This script displays a box on the screen,
after pressing the 'r' key on the keyboard,
a single frame the size of the box is saved and written to the output directory
"""

import os
import cv2

outdir = "/Users/Belal/PycharmProjects/sign2text/test_output/"
prefix = "frame_"
suffix = ".jpg"

session = 0
frame_number = 0

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if ret:

        # Get the width and height of frame
        width = int(cap.get(3)+0.5)
        height = int(cap.get(4)+0.5)
        x_ = width//4
        y_ = height//7
        w_ = width//3
        h_ = width//3
        cv2.rectangle(frame, (x_-2, y_-2), (x_ + w_+2, y_ + h_+2), (0, 255, 0), 1)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('r'):
            for i in range(1):
                # crop ROI
                crop_y = y_ + w_
                crop_x = x_ + w_
                cropped_frame = frame[y_+2:crop_y-2, x_+2: crop_x-2, :]
                # save the processed frame
                outname = prefix + "%07d%s" % (frame_number, suffix)
                cv2.imwrite(os.path.join(outdir, outname), cropped_frame)
                # print frame number
                print("frame %07d" % frame_number)
                frame_number += 1

        elif cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break


# Release everything if job is finished
cap.release()
cv2.destroyAllWindows()
