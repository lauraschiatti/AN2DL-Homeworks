# !/usr/bin/env python3.6
#  -*- coding utf-8 -*-

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Fix the seed for random operations
# to make experiments reproducible.
# ----------------------------------
SEED = 123
tf.random.set_seed(SEED)

cwd = os.getcwd() # current working directory

DATASET_DIR = os.path.join(cwd, 'dataset')
TRAIN_DIR = os.path.join(DATASET_DIR, 'training')
TEST_DIR = os.path.join(DATASET_DIR, 'test')


# Set GPU memory growth
# ---------------------
# import tensorflow as tf
# print('Num GPUs Available ', len(tf.config.experimental.list_physical_devices('GPU')))



# Data Loading
# ------------

bs = 8 # 32 # batch size

decide_class_indices = False

if decide_class_indices:
	class_list = ['owl',  				# 0
				  'galaxy',  			# 1
				  'lightning',  		# 2
				  'wine-bottle',  		# 3
				  't-shirt',  			# 4
				  'waterfall', 			# 5
				  'sword',  			# 6
				  'school-bus',  		# 7
				  'calculator',  		# 8
				  'sheet-music',  		# 9
				  'airplanes',  		# 10
				  'lightbulb',  		# 11
				  'skyscraper',  		# 12
				  'mountain-bike',  	# 13
				  'fireworks',  		# 14
				  'computer-monitor',  	# 15
				  'bear',  				# 16
				  'grand-piano', 		# 17
				  'kangaroo',  			# 18
				  'laptop']  			# 19
else:
	class_list = None


# ImageDataGenerator objects (no data augmentation applied)

VALID_SPLIT = 0.2 # fraction of images reserved for validation

train_datagen = ImageDataGenerator(rescale=1./255, # transforms every pixel value from range [0,255] -> [0,1]
                                   # shear_range=0.2,
                                   # zoom_range=0.2,
                                   # rotation_range=45,
                                   # horizontal_flip=True,
                                   # vertical_flip=True,
                                   validation_split=VALID_SPLIT)


test_datagen = ImageDataGenerator(rescale=1./255,
                                  validation_split=VALID_SPLIT)


# Create generators to read images from dataset directory

print('\ntrain_generator ... ')
train_gen = train_datagen.flow_from_directory(TRAIN_DIR,
											 subset='training', # subset of data
											 batch_size=bs,
											 classes=class_list,
											 class_mode='categorical',
											 shuffle=True,
											 seed=SEED)

print('\nvalid_generator ... ')
valid_gen = train_datagen.flow_from_directory(TRAIN_DIR,
											 subset='validation',
											 batch_size=bs,
											 classes=class_list,
											 class_mode='categorical',
											 shuffle=False,
											 seed=SEED)

