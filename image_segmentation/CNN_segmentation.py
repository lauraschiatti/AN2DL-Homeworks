# !/usr/bin/env python3.6
#  -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

# CNN for semantic segmentation
# -----------------------------

# Fix the seed for random operations
# to make experiments reproducible.
SEED = 1234
tf.random.set_seed(SEED)

# Get current working directory
import os

cwd = os.getcwd()

# Set GPU memory growth
# Allows to only as much GPU memory as needed
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     # Currently, memory growth needs to be the same across GPUs
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Memory growth must be set before GPUs have been initialized
#     print(e)

# The dataset consists of images, their corresponding labels, and pixel-wise masks.
# The masks are basically labels for each pixel. Each pixel is given one of three categories :

# Class 0 : Pixel belonging to the background.
# Class 1 : Pixel of the building. (corresponding to the value 255 in the stored masks)


# ImageDataGenerator
# ------------------

from keras.preprocessing.image import ImageDataGenerator

apply_data_augmentation = False

# Create training ImageDataGenerator object
# We need two different generators for images and corresponding masks

# fraction of images reserved for validation
valid_split = 0.2

if apply_data_augmentation:
    train_img_data_gen = ImageDataGenerator(rotation_range=10,
                                            width_shift_range=10,
                                            height_shift_range=10,
                                            zoom_range=0.3,
                                            horizontal_flip=True,
                                            vertical_flip=True,
                                            fill_mode='constant',
                                            cval=0,
                                            rescale=1. / 255,
                                            validation_split=valid_split)

    train_mask_data_gen = ImageDataGenerator(rotation_range=10,
                                             width_shift_range=10,
                                             height_shift_range=10,
                                             zoom_range=0.3,
                                             horizontal_flip=True,
                                             vertical_flip=True,
                                             fill_mode='constant',
                                             cval=0,
                                             validation_split=valid_split)
else:
    train_img_data_gen = ImageDataGenerator(rescale=1. / 255,
                                            validation_split=valid_split)
    train_mask_data_gen = ImageDataGenerator(validation_split=valid_split)



# Create generators to read images from dataset directory
# -------------------------------------------------------

dataset_dir = os.path.join(cwd, 'image_segmentation/dataset')  # path to dataset
train_dir = os.path.join(dataset_dir, 'training')
test_dir = os.path.join(dataset_dir, 'test')

# Batch size
batch_size = 4

# input image dimensions
img_h = 256
img_w = 256

# number of input channels (color space)
output_channels = 1  # greyscale

num_classes = 2

# Training
print('\ntrain_gen ... ')
train_img_gen = train_img_data_gen.flow_from_directory(os.path.join(train_dir, 'images'),
                                                       subset='training',  # subset of data
                                                       target_size=(img_w, img_h),
                                                       batch_size=batch_size,
                                                       color_mode='grayscale',
                                                       class_mode=None,
                                                       shuffle=True,
                                                       interpolation='bilinear',
                                                       seed=SEED)

train_mask_gen = train_mask_data_gen.flow_from_directory(os.path.join(train_dir, 'masks'),
                                                         subset='training',
                                                         target_size=(img_h, img_w),
                                                         batch_size=batch_size,
                                                         color_mode='grayscale',
                                                         class_mode=None,
                                                         shuffle=True,
                                                         interpolation='bilinear',
                                                         seed=SEED)

train_gen = zip(train_img_gen, train_mask_gen)

# Validation
print('\nvalid_gen ... ')
valid_img_gen = train_img_data_gen.flow_from_directory(os.path.join(train_dir, 'images'),
                                                       subset='validation',
                                                       target_size=(img_h, img_w),
                                                       batch_size=batch_size,
                                                       color_mode='grayscale',
                                                       class_mode=None,
                                                       shuffle=False,
                                                       interpolation='bilinear',
                                                       seed=SEED)

valid_mask_gen = train_mask_data_gen.flow_from_directory(os.path.join(train_dir, 'masks'),
                                                         subset='validation',
                                                         target_size=(img_h, img_w),
                                                         batch_size=batch_size,
                                                         color_mode='grayscale',
                                                         class_mode=None,
                                                         shuffle=False,
                                                         interpolation='bilinear',
                                                         seed=SEED)
valid_gen = zip(valid_img_gen, valid_mask_gen)


# # Create Dataset objects
# # ----------------------

# Validation
# ----------
# When using data augmentation on masks we recommend to cast mask tensor to tf.int32
def prepare_target(x_, y_):
    y_ = tf.cast(y_, tf.int32)
    return x_, y_


valid_dataset = tf.data.Dataset.from_generator(lambda: valid_gen,
                                               output_types=(tf.float32, tf.float32),
                                               output_shapes=([None, img_h, img_w, output_channels],
                                                              [None, img_h, img_w, output_channels]))
valid_dataset = valid_dataset.map(prepare_target)

# Repeat
valid_dataset = valid_dataset.repeat()


# Let's test data generator
# -------------------------


# -------------------------------------- #
#   Convolutional Neural Network (CNN)
# -------------------------------------- #
# Encoder-Decoder

# Create Model
# ------------

def create_model(depth, start_f, num_classes, dynamic_input_shape):
    model = tf.keras.Sequential()

    # Encoder
    # -------
    for i in range(depth):

        if i == 0:
            if dynamic_input_shape:
                input_shape = [None, None, output_channels]
            else:
                input_shape = [img_h, img_w, output_channels]
        else:
            input_shape = [None]

        model.add(tf.keras.layers.Conv2D(filters=start_f,
                                         kernel_size=(3, 3),
                                         strides=(1, 1),
                                         padding='same',
                                         input_shape=input_shape))
        model.add(tf.keras.layers.ReLU())
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

        start_f *= 2

    # Decoder
    # -------
    for i in range(depth):
        model.add(tf.keras.layers.UpSampling2D(2, interpolation='bilinear'))
        model.add(tf.keras.layers.Conv2D(filters=start_f // 2,
                                         kernel_size=(3, 3),
                                         strides=(1, 1),
                                         padding='same'))

        model.add(tf.keras.layers.ReLU())

        start_f = start_f // 2

    # Prediction Layer
    # ----------------
    model.add(tf.keras.layers.Conv2D(filters=num_classes,
                                     kernel_size=(1, 1),
                                     strides=(1, 1),
                                     padding='same',
                                     activation='softmax'))

    return model


model = create_model(depth=4,
                     start_f=4,
                     num_classes=num_classes,
                     dynamic_input_shape=False)

# Visualize created model as a table
model.summary()

# Visualize initialized weights
print(model.weights)


# Optimization params
# -------------------

# Loss
# Sparse Categorical Crossentropy to use integers (mask) instead of one-hot encoded labels
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
# learning rate
lr = 1e-3
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
# -------------------

# Validation metrics
# ------------------

metrics = ['accuracy']
# ------------------

# Compile Model
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


# Just for exercise try to restore a model after training it
# !! Use this just when restoring model..
# ---------------------------------------

eval_out = model.evaluate(x=valid_dataset,
                          steps=len(valid_img_gen),
                          verbose=0)

# print("eval_out", eval_out)

