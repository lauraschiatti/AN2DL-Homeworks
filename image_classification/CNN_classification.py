# !/usr/bin/env python3.6

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from pprint import pprint
from image_classification.utils import data_loader as data
from image_classification.CNNClassifier import CNNClassifier

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

(train_dataset, valid_dataset, test_dataset) = data.setup_dataset()

# Create CNN model
# ------------

depth = 5
num_filters = 8

# Create Model instance
model = CNNClassifier(depth=depth,
                      num_filters=num_filters,
                      num_classes=data.num_classes)

# Build Model (Required)
model.build(input_shape=(None, data.img_h, data.img_w, data.channels))

# Visualize created model as a table
model.feature_extractor.summary()

# Visualize initialized weights
print("initial model weights", model.weights)

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
epochs = 1

callbacks = []
if with_early_stop:
    callbacks.append(
        tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                         patience=epochs * 0.2))

trained_model = model.fit_generator(generator=train_generator,
                                    epochs=epochs,
                                    steps_per_epoch=len(train_generator),
                                    validation_data=valid_dataset,
                                    validation_steps=len(valid_generator))

# Model evaluation
# ----------------
# model.load_weights('/path/to/checkpoint')  # use this if you want to restore saved model

eval_out = model.evaluate(x=test_generator,
                          steps=len(test_generator),
                          verbose=0)

print("eval_out", eval_out)

# Check Performance
# print("Baseline: %.2f%% (%.2f%%)" %
#       (results.mean() * 100, results.std() * 100))

accuracy = trained_model.history['accuracy']
validation_accuracy = trained_model.history['val_accuracy']
loss = trained_model.history['loss']
validation_loss = trained_model.history['val_loss']

epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'b', label='Training acc')
plt.plot(epochs, validation_accuracy, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.show()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, validation_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

# Compute predictions (probabilities -- the output of the last layer)
# -------------------

# predictions = input('\nCompute and save predictions?: ' 'y - Yes  n - No\n')

results = {}

# if predictions == 'y':
print('\n# Generate predictions for pictures ... ')
test_dir = data.test_dir
image_filenames = next(os.walk(test_dir))

for filename in image_filenames[2]:
    # convert the image to RGB
    img = Image.open(os.path.join(test_dir, filename)).convert('RGB')
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, 0)

    # data_normalization
    out_softmax = model.predict(x=img_array / 255.)
    # predicted class
    prediction = tf.argmax(out_softmax)
    results[filename] = prediction

# create_csv(results)

# Prints the nicely formatted dictionary
pprint(results)
