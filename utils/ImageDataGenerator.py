# !/usr/bin/env python3.6
#  -*- coding: utf-8 -*-

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Fix the seed for random operations
# to make experiments reproducible.
# ----------------------------------
SEED = 123
tf.random.set_seed(SEED)

# Get current working directory
cwd = os.getcwd()

# Set GPU memory growth
# ---------------------


# Data Loading
# ------------

# Data splitting
# TRAIN_SAMPLES = 50000 # e.g., Reserve 10,000 sampels for validation, then 50000 samples for training
#
# x_valid = x_train[TRAIN_SAMPLES:, ...]
# y_valid = y_train[TRAIN_SAMPLES:, ...]
#
# x_train = x_train[:TRAIN_SAMPLES, ...]
# y_train = y_train[:TRAIN_SAMPLES, ...]

# Create Datasets
# bs = 32
# train_dataset = dp.images_dataset(x_train, y_train, bs=bs, shuffle=True)
# valid_dataset = dp.images_dataset(x_valid, y_valid, bs=1)
# test_dataset = dp.images_dataset(x_test, y_test, bs=1)

print('---------- ---------- ---------- ---------- ---------- ')


# Check that is everything is ok..


# ImageDataGenerator

# train_dataset, valid_dataset, test_dataset = image_data_generator()