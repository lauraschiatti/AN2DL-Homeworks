# !/usr/bin/env python3.6
#  -*- coding utf-8 -*-

import os
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Global config params  @todo: create config dic with params
# -------------------
						# Parameters
						# params = {'dim': (32,32,32),
						#           'batch_size': 64,
						#           'num_classes': 6,
						#           'n_channels': 1,
						#           'shuffle': True}

# fix the seed for random operations to make experiments reproducible
seed = 123
tf.random.set_seed(seed)

# path to dataset
cwd = os.getcwd()
dataset_dir = os.path.join(cwd, '1_Image_Classification/dataset')
train_dir = os.path.join(dataset_dir, 'training')

# fraction of images reserved for validation
valid_split = 0.2 # 20%

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
def setup_data_generator():
	apply_data_augmentation = False

	# define data augmentation configuration for training data
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

	# setup train and valid generators
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



	# define data augmentation configuration for test data
	test_data_gen = ImageDataGenerator(rescale=1. / 255)  # , validation_split=validation_split)


	print('\ntest_gen ... ')
	# test directory doesn’t have subdirectories the classes of those images are unknown
	test_gen = test_data_gen.flow_from_directory(dataset_dir,  # specify the parent dir of the test dir
													batch_size=bs,
													target_size=(img_w, img_h),
													color_mode='rgb',
													classes=['test'],  # load the test “class”
													# to yield the images in “order”, to predict the outputs
												    # and match them with their unique ids or filenames
													shuffle=False,
													seed=seed)

	# get config params from train generator
	images, labels = next(train_gen)

	global num_classes, channels

	# inputs (x_train)
	channels = images.shape[3]
	print("channels", channels)

	# labels (y_labels)
	num_classes = labels.shape[1]
	print("num_classes", num_classes)


	# Create dataset objects to retrieve
	train_dataset = dataset_from_generator(train_gen)
	train_dataset = train_dataset.repeat()

	valid_dataset = dataset_from_generator(valid_gen)
	valid_dataset = valid_dataset.repeat()

	test_dataset = dataset_from_generator(test_gen)
	test_dataset = test_dataset.repeat()


	return train_dataset, valid_dataset, test_dataset, train_gen, valid_gen, test_gen


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
	for i in range(1, 9):
		plt.subplot(3, 3, i)  # create an axes object in the figure (n_rows, n_cols, plot_id)

		# plot raw pixel data
		image = image_batch[i]  # i-th image
		image = image * 255  # denormalize
		plt.imshow(np.uint8(image))

		# label = tf.where(label_batch[i] == 1)
		# # plt.title(class_list[label])
		plt.axis('off')

	plt.show()  # show the figure


def create_multiclass_model():
	which_model = 'base_weight_decay'
	# set_which_model(which_model)  # set model for training_callbacks

	# Create base model using functional API Model (e.g., Input -> Hidden -> Out)
	if which_model == 'model':
		# x = tf.keras.Input(shape=[28, 28])  # input tensor
		# flatten = tf.keras.layers.Flatten()(x)
		# h = tf.keras.layers.Dense(units=10, activation=tf.keras.activations.sigmoid)(flatten)  # hidden layers
		# output layer:probabccc of belonging to each class
		# out = tf.keras.layers.Dense(units=10, activation=tf.keras.activations.softmax)(h)
		# model = tf.keras.Model(inputs=x, outputs=out)

		# equivalent formulation:
		model = tf.keras.Sequential()
		model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))  # or as a list
		model.add(tf.keras.layers.Dense(units=10, activation=tf.keras.activations.sigmoid))
		model.add(tf.keras.layers.Dense(units=10, activation=tf.keras.activations.softmax))

	# Create base model using sequential model (e.g., Input -> Hidden -> Out)
	elif which_model == 'base':
		model = tf.keras.Sequential()
		model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))  # or as a list
		model.add(tf.keras.layers.Dense(units=1000, activation=tf.keras.activations.sigmoid))
		model.add(tf.keras.layers.Dense(units=10, activation=tf.keras.activations.softmax))

	# Create model with Dropout layer
	elif which_model == 'base_dropout':

		model = tf.keras.Sequential()
		model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))  # or as a list
		model.add(tf.keras.layers.Dense(units=1000, activation=tf.keras.activations.sigmoid))
		model.add(tf.keras.layers.Dropout(0.3))  # rate (probab): 0.3
		model.add(tf.keras.layers.Dense(units=10, activation=tf.keras.activations.softmax))

	# Create model with weights penalty (L2 regularization)
	elif which_model == 'base_weight_decay':

		model = tf.keras.Sequential()
		model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))  # or as a list
		model.add(tf.keras.layers.Dense(units=1000,
										activation=tf.keras.activations.sigmoid,
										kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
		model.add(tf.keras.layers.Dense(units=10,
										activation=tf.keras.activations.softmax,
										kernel_regularizer=tf.keras.regularizers.l2(0.0001)))

	return model




# --------------------------
# Multi-class Classification
# --------------------------

# Data loader
# -----------

train_dataset, valid_dataset, test_dataset, train_gen, valid_gen, test_gen = setup_data_generator()

show_batch(train_dataset)



# Create model
# ------------

model = create_multiclass_model()

# Visualize created model as a table
# model.summary()

# Visualize initialized weights
# print('model initial weights', model.weights)



# Specify the training configuration (optimizer, loss, metrics)
# -------------------------------------------------------------
loss = tf.keras.losses.CategoricalCrossentropy() # loss function to minimize
#
lr = 1e-4 # learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=lr) # stochastic gradient descent optimizer

metrics = ['accuracy'] # validation metrics to monitor

# # Compile Model
model.compile(optimizer=optimizer,
				loss=loss,
				metrics=metrics)



# Train the model
# ---------------
#@todo: check fit_generator
epochs = 10

step_size_train = train_gen.n // train_gen.batch_size  # num_train_samples/bs
step_size_valid = valid_gen.n // valid_gen.batch_size

history = model.fit_generator(generator= train_gen,
								steps_per_epoch=step_size_train,
								epochs=epochs,
								validation_data=valid_gen, # validation generator
								validation_steps=step_size_valid)
								# callbacks=[checkpointer, stopper]
							  	# shuffle=True)

#The returned 'history' object holds a record of the loss values and metric values during training
print('\nhistory dict:', history.history)


# Evaluate the model
# ----------

# print('Evaluate model on test data ... ')
# eval_out = model.evaluate_generator(generator=valid_gen,
# 									 	steps=step_size_valid)
#										verbose=0)

# print('test loss, test acc:', eval_out)


# Check Performance
# print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# Predict output
# --------------

# step_size_test = test_gen.n // test_gen.batch_size

# reset the test_generator before whenever you call the predict_generator.
# This is important, if you forget to reset the test_generator you will get outputs in a weird order.
# test_gen.reset()
#
# pred = model.predict_generator(generator=test_gen,
# 									steps=step_size_test,
# 									verbose=1)
#
#
# predicted_class_indices=np.argmax(pred,axis=1) # predicted abels


# and most importantly you need to map the predicted labels with their unique ids
# such as filenames to find out what you predicted for which image.

# labels = (train_gen.class_indices)
# labels = dict((v,k) for k,v in labels.items())
# predictions = [labels[k] for k in predicted_class_indices]


# save results in a csv file
# import pandas as pd
#
# image_filenames = test_gen.filenames
#
# results = pd.DataFrame({"Filename": image_filenames,
# 						   "Predictions": predictions})
# results.to_csv("results.csv", index=False)
