# !/usr/bin/env python3.6

import numpy as np
import tensorflow as tf
# from keras.layers import Flatten, Dense, Activation
# from keras import Input, activations, Model, losses, optimizers, Sequential
# import tensorflow.keras as k
import matplotlib.pyplot as plt

from image_classification.utils import data_loader as data

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
x_shape = (256, 256, 3)
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
fit_model = model.fit(train_dataset,
                      epochs=epochs,
                      steps_per_epoch=800,
                      callbacks=callbacks,
                      validation_data=valid_dataset,
                      validation_steps=200)
#
# # history lds a record of the loss values and metric values during training
print('\nhistory dict:', fit_model.history)

# Evaluate the model
# ----------

print('Evaluate model on test data ... ')
eval_out = model.evaluate(valid_dataset, steps=200, verbose=1)
print('test loss:', eval_out)

# Check Performance
# print("Baseline: %.2f%% (%.2f%%)" %
#       (results.mean() * 100, results.std() * 100))

accuracy = fit_model.history['accuracy']
validation_accuracy = fit_model.history['val_accuracy']
loss = fit_model.history['loss']
validation_loss = fit_model.history['val_loss']

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
#                            "Predictions": predictions})
# results.to_csv("results.csv", index=False)
"""
