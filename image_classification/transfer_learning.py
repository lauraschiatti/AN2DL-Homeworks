# !/usr/bin/env python3.6
#  -*- coding utf-8 -*-


import tensorflow as tf
from utils import data_manager as data

# Fix the seed for random operations
# ----------------------------------
seed = 123
tf.random.set_seed(seed)


# User a pre-trained network for transfer learning
# Load VGG16 model
# ----------------
vgg = tf.keras.applications.VGG16(weights='imagenet',
                                  include_top=False,
                                  input_shape=(data.img_h, data.img_w, data.channels))
vgg.summary()
print(vgg.layers)

model_name = 'CNN+TF'