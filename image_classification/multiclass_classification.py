# !/usr/bin/env python3.6

import numpy as np
import tensorflow as tf
# from keras.layers import Flatten, Dense, Activation
# from keras import Input, activations, Model, losses, optimizers, Sequential
# import tensorflow.keras as k
import matplotlib.pyplot as plt
from PIL import Image
from pprint import pprint
import os

from image_classification.utils import data_loader as data
from create_submission_file import create_csv

# -------------------------------------- #
### Multi-class Classification ###
# -------------------------------------- #

# Fix the seed for random operations
# ----------------------------------
seed = 123
tf.random.set_seed(seed)

# Data loader
# -----------
# (train_generator, valid_generator, test_generator) = data.setup_data_generator()
# data.show_batch(train_generator)

(train_dataset, valid_dataset, test_dataset) = data.setup_dataset()

# Create model
# ------------
# We decided to use se Sequential model because it is simpler

sequential_model = True
x_shape = (data.img_h, data.img_w, data.channels)
hidden_layer_units = 100
hidden_layers_activation = tf.keras.activations.tanh

if sequential_model:
    # Sequential API implementation
    model = tf.keras.Sequential()
    # we can use input_shape=None when we don't know the shape
    model.add(tf.keras.layers.Flatten(input_shape=x_shape))
    model.add(
        tf.keras.layers.Dense(units=hidden_layer_units,
                              activation=hidden_layers_activation))
    model.add(
        tf.keras.layers.Dense(units=data.num_classes,
                              activation=tf.keras.activations.softmax))
else:
    # Functional API implementation
    input_tensor = tf.keras.Input(shape=x_shape)  # input tensor
    # input layer
    input_layer = tf.keras.layers.Flatten()(input_tensor)
    hidden_layers = tf.keras.layers.Dense(
        units=hidden_layer_units,
        activation=hidden_layers_activation)(input_layer)
    # output layer => probability of belonging to each class
    output_layer = tf.keras.layers.Dense(
        units=data.num_classes,
        activation=tf.keras.activations.softmax)(hidden_layers)

    model = tf.keras.Model(inputs=input_tensor, outputs=output_layer)

# Visualize created model as a table
model.summary()
# print('model initial weights', model.weights)

# Specify the training configuration (optimizer, loss, metrics)
# -------------------------------------------------------------
# loss function to minimize
categorical_crossentropy_loss = tf.keras.losses.CategoricalCrossentropy()

gamma = 1e-4
# stochastic gradient descent optimizer
adam_optimizer = tf.keras.optimizers.Adam(learning_rate=gamma)

# validation metrics to monitor
metrics = ['accuracy']

# Compile Model
model.compile(optimizer=adam_optimizer,
              loss=categorical_crossentropy_loss,
              metrics=metrics)

# Train the model
# ---------------
with_early_stop = True
epochs = 10

callbacks = []
if with_early_stop:
    callbacks.append(
        tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                         patience=epochs * 0.2))

# early stop with patience of 20% the epochs
trained_model = model.fit(train_dataset,
                          epochs=epochs,
                          steps_per_epoch=800,
                          callbacks=callbacks,
                          validation_data=valid_dataset,
                          validation_steps=200)
#
# # history lds a record of the loss values and metric values during training
print('\nhistory dict:', trained_model.history)

# Evaluate the model
# ----------

print('Evaluate model on test data ... ')
eval_out = model.evaluate(valid_dataset, steps=200, verbose=1)
print('test loss:', eval_out)

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
"""
# Predict output
# --------------

# step_size_test = test_gen.n // test_gen.batch_size

# reset the test_generator before whenever you call the predict_generator.
# This is important, if you forget to reset the test_generator you will get outputs in a weird order.
# test_gen.reset()
#
# pred = model.predict_generator(generator=test_gen,
#                                     steps=step_size_test,
#                                     verbose=1)
#
#
# predicted_class_indices=np.argmax(pred,axis=1) # predicted labels
"""
