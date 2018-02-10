#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.mobilenet import MobileNet

from keras.layers import Flatten, Dense, Dropout, Input
from keras.models import Sequential, Model

import argparse

# Map model names to classes
MODELS = {
    "vgg16": VGG16,
    "inception": InceptionV3,
    "xception": Xception,
    "resnet": ResNet50,
    "mobilenet": MobileNet
}

# Define path to pre-trained classification block weights - this is
vgg_weights_path = "weights/snapshot_vgg_weights.hdf5"
res_weights_path = "weights/snapshot_res_weights.hdf5"
mob_weights_path = "weights/snapshot_mob_weights.hdf5"

def create_model(model, model_weights_path=None, top_model=True, color_mode="rgb", input_shape=None):
    """Create custom model for transfer learning

    Steps:
    (i) load pre-trained NN architecture
    (ii) (optional) add custom classification block of two fully connected layers
    (iii) load pre-trained model weights, if available

    Parameters
    ----------
    model: str
        choose which pre-trained Keras deep learning model to use for the 'bottom' layers of the custom model
    model_weights_path: str
        optional path to weights for classification block; otherwise, pre-trained weights will be loaded
    top_model: bool
        whether to include custom classification block, or to load model 'without top' to extract features
    color_mode: str
        whether the image is gray scale or RGB; this will determine number of channels of model input layer

    Returns
    -------
    my_model: keras.model
        Model utilised for prediction or training
    """

    # ensure a valid model name was supplied
    if model not in MODELS.keys():
        raise AssertionError("The model parameter must be a key in the `MODELS` dictionary")

    # gray scale or color
    if color_mode == "grayscale":
        num_channels = 1
    else:
        num_channels = 3

    # Create pre-trained model for feature extraction, without classification block
    print("[INFO] loading %s..." % (model,))
    model = MODELS[model](include_top=False,
                          input_shape=(224, 224, 3))

    # For transfer learning
    if top_model:
        # Create classification block
        top_model = Sequential()
        top_model.add(Flatten(input_shape=model.output_shape[1:]))
        top_model.add(Dense(256, activation='relu'))
        top_model.add(Dense(26, activation='softmax'))

        # Load weights for classification block
        print("[INFO] loading model weights.")
        if model_weights_path is not None:
            # user-supplied weights
            top_model.load_weights(model_weights_path)
        elif model == "vgg16":
            # pre-trained weights for transfer learning with VGG16
            top_model.load_weights(vgg_weights_path)
        elif model == "resnet":
            # pre-trained weights for transfer learning with ResNet50
            print("ResNet50 pre-trained weights are not available yet, please use VGG16 for now!")
            top_model.load_weights(res_weights_path)
        elif model == "mobnet":
            # pre-trained weights for transfer learning with ResNet50
            print("ResNet50 pre-trained weights are not available yet, please use VGG16 for now!")
            top_model.load_weights(mob_weights_path)

        # Join pre-loaded model + classification block
        print("[INFO] creating model.")
        my_model = Model(inputs=model.input,
                         outputs=top_model(model.output))
        return my_model
    else:
        return model
