#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LIVE DEMO
This script loads a pre-trained model (for best results use pre-trained weights for classification block)
and classifies American Sign Language finger spelling frame-by-frame in real-time
"""

import string
import cv2
import time
from processing import square_pad, preprocess_for_vgg
from model import create_model
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", default=None,
                help="path to the model weights")
required_ap = ap.add_argument_group('required arguments')
required_ap.add_argument("-m", "--model",
                         type=str, default="resnet", required=True,
                         help="name of pre-trained network to use")
args = vars(ap.parse_args())


# ====== Create model for real-time classification ======
# =======================================================

# Map model names to classes
MODELS = ["resnet", "vgg16", "inception", "xception", "mobilenet"]

if args["model"] not in MODELS:
    raise AssertionError("The --model command line argument should be a key in the `MODELS` dictionary")

# Create pre-trained model + classification block, with or without pre-trained weights
my_model = create_model(model=args["model"],
                        model_weights_path=args["weights"])

# Dictionary to convert numerical classes to alphabet
label_dict = {pos: letter
              for pos, letter in enumerate(string.ascii_uppercase)}


# ====================== Live loop ======================
# =======================================================

video_capture = cv2.VideoCapture(0)

fps = 0
start = time.time()

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    fps += 1

    # Draw rectangle around face
    x = 313
    y = 82
    w = 451
    h = 568
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 3)

    # Crop + process captured frame
    hand = frame[83:650, 314:764]
    hand = square_pad(hand)
    hand = preprocess_for_vgg(hand)

    # Make prediction
    my_predict = my_model.predict(hand,
                                  batch_size=1,
                                  verbose=0)

    # Predict letter
    top_prd = np.argmax(my_predict)

    # Only display predictions with probabilities greater than 0.5
    if np.max(my_predict) >= 0.50:

        prediction_result = label_dict[top_prd]
        preds_list = np.argsort(my_predict)[0]
        pred_2 = label_dict[preds_list[-2]]
        pred_3 = label_dict[preds_list[-3]]

        width = int(video_capture.get(3) + 0.5)
        height = int(video_capture.get(4) + 0.5)

        # Annotate image with most probable prediction
        cv2.putText(frame, text=prediction_result,
                    org=(width // 2 + 230, height // 2 + 75),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=17, color=(255, 255, 0),
                    thickness=15, lineType=cv2.LINE_AA)
        # Annotate image with second most probable prediction (displayed on bottom left)
        cv2.putText(frame, text=pred_2,
                    org=(width // 2 + width // 5 + 40, (360 + 240)),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=6, color=(0, 0, 255),
                    thickness=6, lineType=cv2.LINE_AA)
        # Annotate image with third probable prediction (displayed on bottom right)
        cv2.putText(frame, text=pred_3,
                    org=(width // 2 + width // 3 + 5, (360 + 240)),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=6, color=(0, 0, 255),
                    thickness=6, lineType=cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Press 'q' to exit live loop
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Calculate frames per second
end = time.time()
FPS = fps/(end-start)
print("[INFO] approx. FPS: {:.2f}".format(FPS))

# Release the capture
video_capture.release()
cv2.destroyAllWindows()

