# !/usr/bin/env python3.6
#  -*- coding: utf-8 -*-

import os
import tensorflow as tf

from image_classification.utils import data_loader as data
from CNNClassifier import CNNClassifier

# -------------------------------------- #
### CNN Classification ###
# -------------------------------------- #

# Fix the seed for random operations
# ----------------------------------
seed = 123
tf.random.set_seed(seed)


# img shape
img_h = 256
img_w = 256
channels = 3
num_classes = 21

# Get current working directory
cwd = os.getcwd()

# todo: solve GPU

# Data loader
# -----------
(train_generator, valid_generator, test_generator) = data.setup_data_generator()
# data.show_batch(train_generator)

(train_dataset, valid_dataset, test_dataset) = data.setup_dataset()


# Create CNN model
# ------------

depth = 5
num_filters = 8

# Create Model instance
model = CNNClassifier(depth=depth, num_filters=num_filters, num_classes=num_classes)

# Build Model (Required)
model.build(input_shape=(None, img_h, img_w, channels))

# Visualize created model as a table
model.feature_extractor.summary()

# Visualize initialized weights
print("initial model weights", model.weights)


# Prepare the model for training
# ------------------------------

loss = tf.keras.losses.CategoricalCrossentropy()

lr = 1e-3 # learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

metrics = ['accuracy'] # validation metrics to monitor

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


# Training with callbacks
# -----------------------

model.fit(x=train_dataset,
		  epochs=10,  #100 ### set repeat in training dataset
		  steps_per_epoch=len(train_generator),
		  validation_data=valid_dataset,
		  validation_steps=len(valid_generator))# callbacks=cb.callbacks)


# Model evaluation
# ----------------
# model.load_weights('/path/to/checkpoint')  # use this if you want to restore saved model

eval_out = model.evaluate(x=test_dataset,
                          steps=len(test_generator),
                          verbose=0)

print("eval_out", eval_out)


