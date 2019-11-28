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

# todo: create dataset_split.json file indicating how do you split the training set...

# Data loader
# -----------
train_generator, valid_generator = data.setup_data_generator()
# data.show_batch(train_generator)

# Create CNN model
# ------------
model_name = 'CNN'
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

# Train the model
# ---------------
with_early_stopping = True
epochs = 20

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
