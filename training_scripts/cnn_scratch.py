#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras.layers import Flatten, Dense, Dropout, Convolution2D, Activation, MaxPooling2D
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, Callback
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adadelta, Adagrad, Adam, Adamax, Nadam

import numpy as np
import argparse
import os


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--train", required=True,
                help="path to the train image directory")
ap.add_argument("-test", "--test", required=True,
                help="path to the test image directory")
ap.add_argument("-o", "--out_dir", default=os.getcwd(),
                help="directory to output features")
ap.add_argument("-b", "--batch_size",
                type=int, default=32,
                help="batch size")
ap.add_argument("-e", "--epochs",
                type=int, default=100,
                help="number of epochs")
ap.add_argument("-c", "--channels",
                type=int, default=3,
                help="choose number of channels (1 or 3); gray scale or color images")
args = vars(ap.parse_args())


#################
batch_size = args["batch_size"]
nb_classes = 26
nb_epoch = args["epochs"]
image_size = 244
num_channels = args["channels"]
if num_channels ==3:
    colour = "rgb"
elif num_channels == 1:
    colour = "grayscale"
#################

# input image dimensions
img_rows, img_cols = image_size, image_size

# number of convolutional filters to use
nb_filters = 32     
nb_filters2 = 64
nb_filters3 = 128

# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

input_shape = (img_rows, img_cols, num_channels)

model = Sequential()
                                                                               
model.add(Convolution2D(nb_filters,
                        (kernel_size[0], kernel_size[1]),
                        input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters2, (kernel_size[0], kernel_size[1])))
model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=pool_size))
#model.add(Convolution2D(nb_filters3, (kernel_size[0], kernel_size[1])))
#model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
                                                                                                                                             
model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.25))
                                                                               
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=Adadelta(),
              metrics=['accuracy'])

print("model loaded.")

# TRAINING

# Model Checkpoint
filepath = "back_up_" + "_weights.hdf5"

save_snapshots = ModelCheckpoint(filepath,
                                 monitor='val_acc',
                                 save_best_only=True,
                                 save_weights_only=True,
                                 mode='max',
                                 verbose=1)


# Save loss history
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracy = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracy.append(logs.get('acc'))

loss_history = LossHistory()
#callbacks_list = [loss_history]

# define train data generator
train_datagen = ImageDataGenerator(rescale=1.,
                                   featurewise_center=True,
                                   rotation_range=15.0,
                                   width_shift_range=0.15,
                                   height_shift_range=0.15)

train_datagen.mean = np.array([103.939, 116.779, 123.68], dtype=np.float32).reshape(1, 1, 3)

train_generator = train_datagen.flow_from_directory(
            args['train'],
            target_size=(244, 244),
            batch_size=batch_size,
            class_mode="categorical",
            color_mode=colour,
            shuffle=False
            )

# define validation data generator
test_datagen = ImageDataGenerator(rescale=1.,
                                  featurewise_center=True)

test_datagen.mean = np.array([103.939, 116.779, 123.68], dtype=np.float32).reshape(1, 1, 3)

test_generator = test_datagen.flow_from_directory(
            args['test'],
            target_size=(244, 244),
            batch_size=batch_size,
            class_mode="categorical",
            color_mode=colour,
            shuffle=False
            )

##############################
steps_per_epoch = int(train_generator.samples//batch_size)
validation_steps = int(test_generator.samples//batch_size)
##############################

# train model
my_history = model.fit_generator(
    generator=train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=nb_epoch,
    verbose=1,
    #callbacks=callbacks_list,
    validation_data=test_generator,
    validation_steps=validation_steps,
    class_weight=None,
    pickle_safe=False)

#save_loss_history = loss_history.losses
#save_accuracy_history = loss_history.accuracy
#np.savetxt("loss_history.txt", save_loss_history, delimiter=",")
#np.savetxt("accuracy_history.txt", save_accuracy_history, delimiter=",")
#my_model.save_weights('my_model_weights.h5')

evaluation_cost = my_history.history['val_loss']
evaluation_accuracy = my_history.history['val_acc']
training_cost = my_history.history['loss']
training_accuracy = my_history.history['acc']

np.save("evaluation_cost.npy", evaluation_cost)
np.save("evaluation_accuracy.npy", evaluation_accuracy)
np.save("training_cost.npy", training_cost)
np.save("training_accuracy.npy", training_accuracy)

