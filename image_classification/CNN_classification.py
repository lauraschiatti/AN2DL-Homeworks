# !/usr/bin/env python3
#  -*- coding utf-8 -*-

import tensorflow as tf
from utils import data_manager as data
from CNNClassifier import CNNClassifier

# Fix the seed for random operations
# ----------------------------------
seed = 123
tf.random.set_seed(seed)

# todo: solve GPU

# Data loader
# -----------
train_generator, valid_generator = data.setup_data_generator()
# data.show_batch(train_generator) # check data loader


# Create CNN model
# ----------------
model_name = 'CNN'

# depth of the input volume i.e. different color channels of an image
depth = 5

# number of convolutional filter kernels to use
#  weights where each is used for a convolution: trainable variables defining the filter.
num_filters = 32

# size of pooling area for max pooling
pool_size = 2

# convolution kernel size
kernel_size = 3

# Create model instance
model = CNNClassifier(depth=depth,
                      num_filters=num_filters,
                      pool_size=pool_size,
                      kernel_size=kernel_size,
                      num_classes=data.num_classes)

# Build model
model.build(input_shape=data.input_shape)

# Visualize created model as a table
# model.feature_extractor.summary()
# Visualize initialized weights
# print('initial model weights', model.weights)


# Prepare the model for training
# ------------------------------
loss = tf.keras.losses.CategoricalCrossentropy()

# learning rate
lr = 1e-4
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

metrics = ['accuracy']  # validation metrics to monitor

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# Train the model
# ---------------
with_early_stopping = True
epochs = 100

callbacks = []
if with_early_stopping:
    callbacks.append(
        tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                         patience=epochs * 0.2))

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

# history contains a trace of the loss and any other metrics specified during the compilation of the model
print('\nhistory dict:', trained_model.history)

# Check Performance
data.visualize_performance(trained_model)

# Generate predictions
# -------------------
predictions = input('\nCompute and save predictions?: ' 'y - Yes  n - No\n')

if predictions == 'y':
    data.generate_predictions(model, model_name)
