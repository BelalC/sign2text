#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EVALUATION
This script loads a pre-trained model and scores it against the defined test set. The evaluation metrics utilised are
...
"""

import string
import cv2
from processing import square_pad, preprocess_for_vgg
from tensorflow.keras.applications import ResNet50, VGG16, InceptionV3, Xception
from tensorflow.keras.layers import Input
from model import create_model
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", default=None,
                help="path to custom model weights; NOTE: this assumes a pre-trained model with 2 fully"
                     "connected classification layers")
ap.add_argument("-c", "--color", default="RGB", help="choose whether images are grayscale or RGB")
required_ap = ap.add_argument_group('required arguments')
required_ap.add_argument("-m", "--model",
                         type=str, default="resnet", required=True,
                         help="name of pre-trained network to use")
args = vars(ap.parse_args())


# ====== Create model for evaluation ======
# =========================================

# Map model names to classes
MODELS = {
    "vgg16": VGG16,
    "inception": InceptionV3,
    "xception": Xception,
    "resnet": ResNet50
}

# grayscale or color
if args["color"] == "grayscale":
    num_channels = 1
else:
    num_channels = 3

MODELS_LIST = ["resnet", "vgg16", "inception", "xception"]
if args["model"] not in MODELS_LIST:
    raise AssertionError("The --model command line argument should be one of: %s" % ", ".join(MODELS_LIST))

# Create pre-trained model + classification block, with or without pre-trained weights
if args["weights"]:
    my_model = create_model(model_name=args["model"],
                            model_weights_path=args["weights"])
else:
    my_model = MODELS[args["model"]](include_top=False, input_tensor=Input(shape=(224, 224, num_channels)))

# Dictionary to convert numerical classes to alphabet
label_dict = {pos: letter
              for pos, letter in enumerate(string.ascii_uppercase)}

# ======================
# ====== EVALUATE ======
# ======================

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