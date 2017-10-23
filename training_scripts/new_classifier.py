#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
TRANSFER LEARNING
This script loads pre-trained deep neural network models included in Keras with a custom classification block,
and trains the new model.
"""


from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.mobilenet import MobileNet

from keras.layers import Flatten, Dense, Input, Dropout
from keras.models import Sequential, Model
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
ap.add_argument("-model", "--model",
                type=str, default="vgg16",
                help="name of pre-trained network to use")
ap.add_argument("-e", "--epochs",
                type=int, default=30,
                help="number of epochs")
ap.add_argument("-b", "--batch_size",
                type=int, default=32,
                help="batch size")
ap.add_argument("-c", "--colour",
                type=str, default="rgb",
                help="choose whether to load gray scale or color images")
args = vars(ap.parse_args())


####################
batch_size = args['batch_size']
####################

# define a dictionary that maps model names to their classes
# Map model names to classes
MODELS = {
    "vgg16": VGG16,
    "inception": InceptionV3,
    "xception": Xception,
    "resnet": ResNet50,
    "mobilenet": MobileNet
}

if args["model"] == "mobilenet":
    shape = (224, 224)
    target_size = (224, 224)
else:
    shape = (244, 244, 3)
    target_size = (244, 244)

# load model
print("[INFO] loading {}...".format(args["model"]))
Network = MODELS[args["model"]]
model = Network(include_top=False,
                weights="imagenet",
                input_tensor=Input(shape=shape))
print("[INFO] model loaded.")


# freeze base model layers
for layer in model.layers:
    layer.trainable = False


# Classification block
print("[INFO] creating classification block")
top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.2))
top_model.add(Dense(26, activation='softmax'))

# load top_model weights
# top_model_weights_path = "/Users/Belal/PycharmProjects/sign2text/bottleneck_features/bottleneck_top_model.h5"
# top_model.load_weights(top_model_weights_path)

# Join model + classification block
my_model = Model(inputs=model.input,
                 outputs=top_model(model.output))

# compile model
print("[INFO] compiling model")
my_model.compile(optimizer=Adadelta(),
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])

# TRAINING

# Model Checkpoint
filepath = "snapshot" + args["model"] + "_weights.hdf5"

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

callbacks_list = [save_snapshots, loss_history]

# define train data generator
train_datagen = ImageDataGenerator(rescale=1.,
                                   featurewise_center=True,
                                   rotation_range=15.0,
                                   width_shift_range=0.15,
                                   height_shift_range=0.15,)

train_datagen.mean = np.array([103.939, 116.779, 123.68], dtype=np.float32).reshape(1, 1, 3)

train_generator = train_datagen.flow_from_directory(
            args['train'],
            target_size=(244, 244),
            batch_size=batch_size,
            class_mode="categorical",
            color_mode=args['colour'],
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
            color_mode=args['colour'],
            shuffle=False
            )

##############################
epochs = args["epochs"]
steps_per_epoch = int(train_generator.samples//batch_size)
validation_steps = int(test_generator.samples//batch_size)
##############################

# train model
my_history = my_model.fit_generator(
    generator=train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    verbose=1,
    callbacks=callbacks_list,
    validation_data=test_generator,
    validation_steps=validation_steps,
    class_weight=None, max_q_size=10,
    pickle_safe=False)

# save_loss_history = loss_history.losses
# save_accuracy_history = loss_history.accuracy
# np.savetxt("loss_history.txt", save_loss_history, delimiter=",")
# np.savetxt("accuracy_history.txt", save_accuracy_history, delimiter=",")
my_model.save_weights('my_model_weights.h5')

evaluation_cost = my_history.history['val_loss']
evaluation_accuracy = my_history.history['val_acc']
training_cost = my_history.history['loss']
training_accuracy = my_history.history['acc']

np.save("evaluation_cost.npy", evaluation_cost)
np.save("evaluation_accuracy.npy", evaluation_accuracy)
np.save("training_cost.npy", training_cost)
np.save("training_accuracy.npy", training_accuracy)

"""
f, (ax1, ax2) = plt.subplots(1, 2)
f.set_figwidth(10)
ax1.plot(evaluation_cost, label='test')
ax1.plot(training_cost, label='train')
ax1.set_title('Cost')
ax1.legend()
ax2.plot(evaluation_accuracy, label='test')
ax2.plot(training_accuracy, label='train')
ax2.set_title('Accuracy')
ax2.legend(loc='lower right')

plt.savefig("loss_curves.jpg")
plt.close()
"""
