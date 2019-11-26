# !/usr/bin/env python3.6

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from utils import data_loader as data
from CNNClassifier import CNNClassifier

# -------------------------------------- #
### CNN Classification ###
# -------------------------------------- #

# Fix the seed for random operations
# ----------------------------------
seed = 123
tf.random.set_seed(seed)

# Get current working directory
cwd = os.getcwd()

# todo: solve GPU

# Data loader
# -----------
(train_generator, valid_generator,
 test_generator) = data.setup_data_generator()
# data.show_batch(train_generator)

# (train_dataset, valid_dataset, test_dataset) = data.setup_dataset()

# Create CNN model
# ------------

depth = 8
num_filters = 10

# Create Model instance
model = CNNClassifier(depth=depth,
                      num_filters=num_filters,
                      num_classes=data.num_classes)

# Build Model (Required)
model.build(input_shape=(None, data.img_h, data.img_w, data.channels))

# Visualize created model as a table
# model.feature_extractor.summary()

# Visualize initialized weights
# print('initial model weights', model.weights)

# Prepare the model for training
# ------------------------------

loss = tf.keras.losses.CategoricalCrossentropy()

lr = 1e-4  # learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

metrics = ['accuracy']  # validation metrics to monitor

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# Training with callbacks
# -----------------------

with_early_stop = True
epochs = 20

callbacks = []
if with_early_stop:
    callbacks.append(
        tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                         patience=epochs * 0.3))

trained_model = model.fit_generator(generator=train_generator,
                                    epochs=epochs,
                                    steps_per_epoch=len(train_generator),
                                    validation_data=valid_generator,
                                    validation_steps=len(valid_generator))

# Model evaluation
# ----------------
# model.load_weights('/path/to/checkpoint')  # use this if you want to restore saved model

eval_out = model.evaluate_generator(valid_generator,
                                    steps=len(valid_generator),
                                    verbose=0)

print('eval_out', eval_out)

# Check Performance
# test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

accuracy = trained_model.history['accuracy']
validation_accuracy = trained_model.history['val_accuracy']
loss = trained_model.history['loss']
validation_loss = trained_model.history['val_loss']

epochs = range(len(accuracy))

# Visualize History for Loss.
plt.title('Model loss')
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, validation_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend(['training', 'validation'], loc='upper right')
# plt.show()

# # Visualize History for Accuracy.
plt.title('Model accuracy')
plt.plot(epochs, accuracy, 'b', label='Training acc')
plt.plot(epochs, validation_accuracy, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend(['training', 'validation'], loc='lower right')
# plt.show()

# Compute predictions (probabilities -- the output of the last layer)
# -------------------

predictions = input('\nCompute and save predictions?: ' 'y - Yes  n - No\n')

target_size = (data.img_h, data.img_w)
results = {}
results_str = {}

test_dir = data.test_dir
image_filenames = next(os.walk(test_dir))[2] #[:10]

if predictions == 'y':

    print('\n# Labeling test images ... ')

    for filename in image_filenames:
        # convert the image to RGB
        img = Image.open(os.path.join(test_dir, filename)).convert('RGB')
        # resize the image
        img = img.resize(target_size)

        # data_normalization - convert to array
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)

        print("prediction for {}...".format(filename))
        predictions = model.predict(img_array * 1 / 255.)

        # Get predicted class as the index corresponding to the maximum value in the vector probability
        predicted_class = np.argmax(predictions, axis=-1) # multiple categories
        predicted_class = predicted_class[0]

        results[filename] = predicted_class
        results_str[filename] = data.class_list[predicted_class]

data.create_csv(results)

# Prints the nicely formatted dictionary
from pprint import pprint
pprint(results_str)

print('Num. of labeled images', results.__len__())
