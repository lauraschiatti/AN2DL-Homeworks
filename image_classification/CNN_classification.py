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
epochs = 10

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

# predictions = input('\nCompute and save predictions?: ' 'y - Yes  n - No\n')

newsize = (256, 256)  # target_size
results = {}

test_dir = data.test_dir
image_filenames = next(os.walk(test_dir))

# test only one image
for filename in image_filenames[2][:1]:
    # convert the image to RGB
    img = Image.open(os.path.join(test_dir, filename)).convert('RGB')
    # resize the image
    img = img.resize(newsize)

    # data_normalization - convert to array
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    print("predict for %s...\n" % filename)
    predictions = model.predict(img_array * 1 / 255.)
    results[filename] = data.class_list[np.argmax(predictions, axis=-1)[0]]

#
# if predictions == 'y':
#
#     print('\n# Labeling test images ... ')
#     test_dir = data.test_dir
#     image_filenames = next(os.walk(test_dir))[2]
#
#     for filename in image_filenames[:10]:
#         print('labeling ' + filename)
#
#         # load image
#         target_size = (data.img_w, data.img_h)
#         # convert the image to RGB
#         img = Image.open(os.path.join(test_dir, filename)).convert('RGB')
#         # resize the image
#         img = img.resize(newsize)
#
#         # data_normalization - convert to array
#         img_array = np.array(img)
#         img_array = np.expand_dims(img_array, axis=0)
#
#         # use predict_generator() is inferring the labels
#         # from the directory structure of training data.
#
#         # softmax = model.predict(x=img_array / 255.)      # data normalization
#         # print('predictions probabs:', softmax.tolist())
#
#         # Get predicted class as the index corresponding to the maximum value in the vector probability
#         # prediction = tf.argmax(softmax, axis=-1) # multiple categories
#
#         # predicted_class = predicted_class[0]
#
#         # results[filename] = prediction
#
#         print("predict for %s...\n" % filename)
#         predictions = model.classifier.predict(img_array)
#         results[filename] = data.class_list[np.argmax(predictions, axis=-1)[0]]
#
#         # todo: with generatory
#         # predictions = model.predict_generator(test_generator)
#         # predictions = np.argmax(predictions, axis=-1)
#         # label_map = (train_generator.class_indices)
#         # label_map = dict((v, k) for k, v in label_map.items())  # flip k,v
#         # predictions = [label_map[k] for k in predictions]
#
#         results[filename] = predictions

# create_csv(results)

# Prints the nicely formatted dictionary
from pprint import pprint
pprint(results)

print('Num. of labeled images', results.__len__())
