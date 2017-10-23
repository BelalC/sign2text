#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FEATURE EXTRACTION
This script extracts features from the final (non-classification) layers of
the pre-trained deep neural network models included in Keras.
"""

from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.mobilenet import MobileNet
from keras.preprocessing.image import ImageDataGenerator
from model import create_model

import numpy as np
import argparse
import string
import os
import joblib


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--train", required=True,
                help="path to the train image directory")
ap.add_argument("-test", "--test", required=True,
                help="path to the test image directory")
ap.add_argument("-o", "--out_dir", default=os.getcwd(),
                help="directory to output features")
ap.add_argument("-model", "--model",
                type=str, default="vgg16",
                help="name of pre-trained network to use")
ap.add_argument("-c", "--colour",
                type=str, default="rgb",
                help="choose whether to load gray scale or color images")
ap.add_argument("-b", "--batch",
                type=int, default=3,
                help="set batch size for feature extraction")
#ap.add_argument("-i", "--input_shape",
#                type=tuple, default=None,
#                help="set input shape for models")
args = vars(ap.parse_args())

# define a dictionary that maps model names to their classes
MODELS = {
    "vgg16": VGG16,
    "inception": InceptionV3,
    "xception": Xception,
    "resnet": ResNet50,
    "mobilenet": MobileNet
}

# ensure a valid model name was supplied via command line argument
if args["model"] not in MODELS.keys():
    raise AssertionError("The --model command line argument should "
                         "be a key in the `MODELS` dictionary")

# Create label dictionary
label_dict = {pos: letter
              for pos, letter in enumerate(string.ascii_uppercase)}

# define batch size
batch_size = args["batch"]

# define input tensor
input_shape = (224, 224, 3)

# load pre-trained transfer learning model
print("[INFO] loading {}...".format(args["model"]))
transfer_model = create_model(model=args["model"],
                              top_model=False,
                              color_mode=args["colour"],
                              input_shape=input_shape)
print("[INFO] model loaded.")

# ==== TO BE INCLUDED: Extract features from an arbitrary intermediate layer
# ====================

# define train data generator
train_datagen = ImageDataGenerator(rescale=1.,
                                   featurewise_center=True)

train_datagen.mean = np.array([103.939, 116.779, 123.68], dtype=np.float32).reshape(1, 1, 3)

train_generator = train_datagen.flow_from_directory(
            args['train'],
            target_size=(244, 244),
            batch_size=batch_size,
            class_mode="categorical",
            color_mode=args['colour'],
            shuffle=False
            )

# define test data generator
test_datagen = ImageDataGenerator(rescale=1.,
                                  featurewise_center=True)

test_datagen.mean = np.array([103.939, 116.779, 123.68], dtype=np.float32).reshape(1, 1, 3)

test_generator = test_datagen.flow_from_directory(
            args['test'],
            target_size=(244, 244),
            batch_size=batch_size,
            class_mode="categorical",
            color_mode=args['colour'],
            shuffle=False
            )

##############################
train_steps_per_epoch = int(train_generator.samples//batch_size)
test_steps_per_epoch = int(test_generator.samples//batch_size)
##############################

# extract features
print("[INFO] extracting training features...")
train_bottleneck_features = transfer_model.predict_generator(train_generator, steps=train_steps_per_epoch)

print("[INFO] extracting test features...")
test_bottleneck_features = transfer_model.predict_generator(test_generator, steps=test_steps_per_epoch)

# save bottleneck features
print("[INFO] saving features...")
train_features_dir = os.path.join(args['out_dir'], args['model'] + "_train_bottleneck_features.pkl")
joblib.dump(train_bottleneck_features,  train_features_dir)

test_features_dir = os.path.join(args['out_dir'], args['model'] + "_test_bottleneck_features.pkl")
joblib.dump(test_bottleneck_features,  test_features_dir)

# save bottleneck labels
print("[INFO] saving labels...")
train_labels = list(train_generator.classes)
test_labels = list(test_generator.classes)

train_labels_dir = os.path.join(args['out_dir'], args['model'] + "_train_bottleneck_labels.pkl")
joblib.dump(train_labels, train_labels_dir)

test_labels_dir = os.path.join(args['out_dir'], args['model'] + "_test_bottleneck_labels.pkl")
joblib.dump(test_labels, test_labels_dir)
