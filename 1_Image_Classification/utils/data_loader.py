# !/usr/bin/env python3.6
#  -*- coding utf-8 -*-

import os

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

# Global config params  @todo: create config file with params
# -------------------

# fix the seed for random operations to make experiments reproducible
seed = 123
tf.random.set_seed(seed)

# path to dataset
cwd = os.getcwd()
dataset_dir = os.path.join(cwd, '1_Image_Classification/dataset')
train_dir = os.path.join(dataset_dir, 'training')

# fraction of images reserved for validation
valid_split = 0.2

# image size @todo: which is the correct size for the images?
img_w = 256
img_h = 256

# color space
channels = 3  # rgb

# batch size
bs = 32  # (default)

# number of classes
num_classes = 20

class_list = ['owl',  # 0
			  'galaxy',  # 1
			  'lightning',  # 2
			  'wine-bottle',  # 3
			  't-shirt',  # 4
			  'waterfall',  # 5
			  'sword',  # 6
			  'school-bus',  # 7
			  'calculator',  # 8
			  'sheet-music',  # 9
			  'airplanes',  # 10
			  'lightbulb',  # 11
			  'skyscraper',  # 12
			  'mountain-bike',  # 13
			  'fireworks',  # 14
			  'computer-monitor',  # 15
			  'bear',  # 16
			  'grand-piano',  # 17
			  'kangaroo',  # 18
			  'laptop']  # 19


# Create image generators from directory
# --------------------------------
def image_data_generator():
	apply_data_augmentation = False

	# Create training ImageDataGenerator object
	if apply_data_augmentation:
		# Train and Validation
		train_data_gen = ImageDataGenerator(rescale=1. / 255, # every pixel value from range [0,255] -> [0,1]
											shear_range=0.2,
											zoom_range=0.2,
											rotation_range=45,
											horizontal_flip=True,
											vertical_flip=True,
											validation_split=valid_split)
	else:
		train_data_gen = ImageDataGenerator(rescale=1. / 255,
											validation_split=valid_split)


	print('\ntrain_gen ... ')
	train_gen = train_data_gen.flow_from_directory(train_dir,
												   subset='training',  # subset of data
												   batch_size=bs,
												   target_size=(img_w, img_h),  # images are automatically resized
												   color_mode='rgb',
												   classes=class_list,
												   class_mode='categorical',
												   shuffle=True,
												   seed=seed)

	print('\nvalid_gen ... ')
	valid_gen = train_data_gen.flow_from_directory(train_dir,
													  subset='validation',
													  batch_size=bs,
													  target_size=(img_w, img_h),
													  color_mode='rgb',
													  classes=class_list,
													  class_mode='categorical',
													  shuffle=False,
													  seed=seed)

	# Test
	test_data_gen = ImageDataGenerator(rescale=1. / 255)  # , validation_split=valid_split)

	print('\ntest_gen ... ')
	# test directory doesn’t have subdirectories the classes of those images are unknown
	test_gen = test_data_gen.flow_from_directory(dataset_dir,  # specify the parent dir of the test dir
													batch_size=bs,
													target_size=(img_w, img_h),
													color_mode='rgb',
													classes=['test'],  # load the test “class”
													shuffle=False,
													seed=seed)

	# set config params from generator
	images, labels = next(train_gen)

	global num_classes, channels
	num_classes = labels.shape[1]
	channels = images.shape[3]

	print("num_classes", num_classes)
	print("channels", channels)

	# Create dataset objects to retrieve
	train_dataset = dataset_from_generator(train_gen)
	train_dataset = train_dataset.repeat()

	valid_dataset = dataset_from_generator(valid_gen)
	valid_dataset = valid_dataset.repeat()

	test_dataset = dataset_from_generator(test_gen)
	test_dataset = test_dataset.repeat()

	print("class labels ...", train_gen.class_indices, end='\n')  # check the class labels

	return train_dataset, valid_dataset, test_dataset


# Create dataset objects
# -----------------------

def dataset_from_generator(generator):
	dataset = tf.data.Dataset.from_generator(lambda: generator,
											 output_types=(tf.float32, tf.float32),
											 output_shapes=([None, img_h, img_w, channels], [None, num_classes]))
	return dataset


# Iterate Dataset object to access samples inside it
# --------------------------------------------------

def show_batch(train_dataset):

	import matplotlib.pyplot as plt
	import numpy as np

	iterator = iter(train_dataset)
	image_batch, label_batch = next(iterator)

	plt.figure() #figsize=(10, 10))

	# create grid of subplots
	for i in range(1, 25):
		plt.subplot(5, 5, i)  # create an axes object in the figure (n_rows, n_cols, plot_id)

		# plot raw pixel data
		image = image_batch[i]  # i-th image
		image = image * 255  # denormalize
		plt.imshow(np.uint8(image))

		# label = tf.where(label_batch[i] == 1)
		# # plt.title(class_list[label])
		plt.axis('off')

	plt.show()  # show the figure


train_dataset, valid_dataset, test_dataset = image_data_generator()

show_batch(train_dataset)


