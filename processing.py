#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
from glob import glob
import numpy as np


def preprocess_frame(directory, img_format="jpg", size=244,
                     drop_green=False, gray=False):
    """Pre-processing for frames captured from video stored in single directory

    Parameters
    ----------
    directory: str
        Path to directory containing images to be processed
    img_format: str
        Format of image to be loaded; must be one of either 'jpg' or 'png'
        default = 'jpg'
    size: int
        Size to which image is re-sized (square of shape: size x size)
    drop_green: bool
        Whether to drop the green channel, for images captured on green screen
    gray: bool
        Whether to convert image to gray scale

    Returns
    -------
    images: np.ndarray
        4D array of processed images
    """
    assert img_format in ["jpg", "png"], "img_format parameter must be one of 'jpg' or 'png'"
    img_format = "*." + img_format
    nb_images = len(glob(directory + img_format))
    assert nb_images > 0, "No images found in directory"
    num_channels = 3
    images = np.empty((nb_images, size, size, num_channels))
    for i, infile in enumerate(glob(directory + img_format)):
        img = cv2.imread(infile)
        if drop_green:
            img[:, :, 1] = 0
        if gray:
            num_channels = 1
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = square_pad(img)
        img = cv2.resize(img, (size, size))
        img = np.reshape(img, (1, size, size, num_channels))
        images[i, :, :, :] = img

    return images


def square_pad(img, padding_color=[0, 0, 0]):
    """Add margins to image to make it square keeping largest original dimension

    Parameters
    ----------
    img: numpy.ndarray
        Image to be processed
    padding_color: list
        Define background colour to pad image; preserves RGB/BGR colour channel order of img

    Returns
    -------
    padded_img: np.ndarray
        Image padded to a square shape

    """
    height = img.shape[0]
    width = img.shape[1]
    # find difference between longest side
    diff = np.abs(width - height)
    # amount of padding = half the diff between width and height
    pad_diff = diff // 2

    if height > width:
        # letter is longer than it is wide
        pad_top = 0
        pad_bottom = 0
        pad_left = pad_diff
        pad_right = pad_diff
        padded_img = cv2.copyMakeBorder(img,
                                        top=pad_top,
                                        bottom=pad_bottom,
                                        left=pad_left,
                                        right=pad_right,
                                        borderType=cv2.BORDER_CONSTANT,
                                        value=padding_color)
    elif width > height:
        # image is wide
        pad_top = pad_diff
        pad_bottom = pad_diff
        pad_left = 0
        pad_right = 0
        padded_img = cv2.copyMakeBorder(img,
                                        top=pad_top,
                                        bottom=pad_bottom,
                                        left=pad_left,
                                        right=pad_right,
                                        borderType=cv2.BORDER_CONSTANT,
                                        value=padding_color)
    elif width == height:
        padded_img = img.copy()

    return padded_img


def preprocess_for_vgg(img, size=224, color=True):
    """Image pre-processing for VGG16 network

    Parameters
    ----------
    img: numpy.ndarray
        Image to be processed
    size: int
        Size to which image is re-sized (square of shape: size x size)
    color: bool
        If the image is colour (BGR colour channels), then it is zero-centred by mean pixel

    Returns
    -------
    x: np.ndarray
        Pre-processed image ready to feed into VGG16 network; re-shaped to (1, size, size, 3)
    """
    img = cv2.resize(img, (size, size))
    x = np.array(img, dtype=float)
    x_fake_batch = x.reshape(1, *x.shape)
    x = x_fake_batch
    if color:
        # Zero-center by mean pixel
        x[:, :, :, 2] -= 123.68
        x[:, :, :, 1] -= 116.779
        x[:, :, :, 0] -= 103.939 
    return x


def edit_bg(img, bg_img_path):
    """Change black background to another image pixel-by-pixel

    Parameters
    ----------
    img: np.ndarray
        Image to be processed; must have BLACK BACKGROUND
    bg_img_path: str
        Path to background image on which to superimpose original image

    Returns
    -------
    img_front: np.ndarray
        Original image superimposed on to new background image; black pixels are replaced by background image
    """

    img_front = img.copy()
    img_back = cv2.imread(bg_img_path)
    height, width = img_front.shape[:2]
    resize_back = cv2.resize(img_back, (width, height), interpolation=cv2.INTER_CUBIC)
    for i in range(width):
        for j in range(height):
            pixel = img_front[j, i]
            if np.all(pixel == [0, 0, 0]):
                img_front[j, i] = resize_back[j, i]
    return img_front

