# !/usr/bin/env python3.6
#  -*- coding: utf-8 -*-

import os
import tensorflow as tf

# Fix the seed for random operations
# to make experiments reproducible.
# ----------------------------------
# SEED = 123
# tf.random.set_seed(SEED)

# Get current working directory
# cwd = os.getcwd()


# Set GPU memory growth
# ---------------------
# gpus = tf.config.experimental.list_physical_devices('GPU')
#
# if gpus:
#   try:
#     # Currently, memory growth needs to be the same across GPUs
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Memory growth must be set before GPUs have been initialized
#     print(e, 'No GPUs found.')
