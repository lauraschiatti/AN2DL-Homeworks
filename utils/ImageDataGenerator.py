# !/usr/bin/env python3.6
#  -*- coding utf-8 -*-

import os
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

# Fix the seed for random operations
# to make experiments reproducible.
# ----------------------------------
seed = 123
tf.random.set_seed(seed)

cwd = os.getcwd() # current working directory

dataset_dir = os.path.join(cwd, 'dataset')
train_dir = os.path.join(dataset_dir, 'training')


# Set GPU memory growth
# ---------------------
# import tensorflow as tf
# print('Num GPUs Available ', len(tf.config.experimental.list_physical_devices('GPU')))



# Data Loading
# ------------

batch_size = 32  # (default)

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


# Image generators from directory

valid_split = 0.2 # fraction of images reserved for validation

#@todo: which is the correct size for the images?
img_w = 256
img_h = 256


train_datagen = ImageDataGenerator(rescale=1./255, # transforms every pixel value from range [0,255] -> [0,1]
                                   # shear_range=0.2,
                                   # zoom_range=0.2,
                                   # rotation_range=45,
                                   # horizontal_flip=True,
                                   # vertical_flip=True,
                                   validation_split=valid_split)

print('\ntrain_gen ... ')
train_gen = train_datagen.flow_from_directory(train_dir,
											 subset='training', # subset of data
											 batch_size=batch_size,
											 target_size=(img_w, img_h), # images are automatically resized
											 color_mode='rgb',
											 classes=class_list,
											 class_mode='categorical',
											 shuffle=True,
											 seed=seed)

print('\nvalid_gen ... ')
valid_gen = train_datagen.flow_from_directory(train_dir,
											 subset='validation',
											 batch_size=batch_size,
											 target_size=(img_w, img_h),
											 color_mode='rgb',
											 classes=class_list,
											 class_mode='categorical',
											 shuffle=False,
											 seed=seed)


test_datagen = ImageDataGenerator(rescale=1./255) #, validation_split=valid_split)

# test directory doesn’t have subdirectories the classes of those images are unknown
print('\ntest_gen ... ')
test_gen= test_datagen.flow_from_directory(dataset_dir, 	# specify the parent dir of the test dir
										  batch_size=batch_size,
										  target_size=(img_w, img_h),
										  color_mode='rgb',
										  classes=['test'], # load the test “class”
										  shuffle=False,
										  seed=seed)
# img shape
images, labels = next(train_gen)

num_classes = labels.shape[1]
channels = images.shape[3]

print("num_classes", num_classes)
print("channels", channels)


# Create dataset objects
train_dataset = tf.data.Dataset.from_generator(lambda: train_gen,  # generator
											   output_types=(tf.float32, tf.float32),
											   output_shapes=([None, img_h, img_w, channels], [None, num_classes]))
train_dataset = train_dataset.repeat() # repeat


valid_dataset = tf.data.Dataset.from_generator(lambda: valid_gen,
											   output_types=(tf.float32, tf.float32),
											   output_shapes=([None, img_h, img_w, channels], [None, num_classes]))
valid_dataset = valid_dataset.repeat()  # repeat


test_dataset = tf.data.Dataset.from_generator(lambda: test_gen,
											  output_types=(tf.float32, tf.float32),
											  output_shapes=([None, img_h, img_w, channels], [None, num_classes]))
test_dataset = test_dataset.repeat()  # repeat

# check the class labels
print("class labels ...", train_gen.class_indices, end='\n')






# Check that is everything is ok..
# --------------------------------

# model.fit_generator(
#     train_generator,
#     steps_per_epoch = train_generator.samples // batch_size,
#     validation_data = validation_generator,
#     validation_steps = validation_generator.samples // batch_size,
#     epochs = nb_epochs)