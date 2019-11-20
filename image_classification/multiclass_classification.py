# !/usr/bin/env python3.6
#  -*- coding: utf-8 -*-

import tensorflow as tf

import utils.data_loader as data


# -------------------------------------- #
	### Multi-class Classification ###
# -------------------------------------- #


# Fix the seed for random operations
# ----------------------------------
seed = 123
tf.random.set_seed(seed)


# todo: gpu management
# Set GPU memory growth
# ---------------------


# Data loader
# -----------
train_dataset, valid_dataset, test_dataset, train_gen, valid_gen, test_gen = data.setup_data_generator()

data.show_batch(train_dataset)


# Create model
# ------------
model = data.create_multiclass_model()

# Visualize created model as a table
model.summary()

# Visualize initialized weights
print('model initial weights', model.weights)
