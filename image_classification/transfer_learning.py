# !/usr/bin/env python3.6
#  -*- coding utf-8 -*-

import tensorflow as tf
from utils import data_manager as data

# Fix the seed for random operations
# ----------------------------------
seed = 123
tf.random.set_seed(seed)

# Data loader
# -----------
train_generator, valid_generator = data.setup_data_generator()


# Use a pre-trained network for transfer learning: train dense layers for new classification task

# # VGG16 architecture consists of twelve convolutional layers,
# some of which are followed by maximum pooling layers
# and then four fully-connected layers and finally a 1000-way softmax classifier


# build the VGG16 network
# ------------------------
vgg = tf.keras.applications.VGG16(weights='imagenet',
                                  include_top=False,
                                  input_shape=data.input_shape)

vgg.summary()
# print("vgg.layers", vgg.layers)

model_name = 'CNN+TF'

# Two types of transfer learning: feature extraction and fine-tuning
fine_tuning = True

if fine_tuning:
    freeze_until = 12  # layer from which we want to fine-tune

    # set the first freeze_until layers (up to the last conv block => depth = 5)
    # to non-trainable (weights will not be updated)
    for layer in vgg.layers[:freeze_until]:
        layer.trainable = False

else:
    vgg.trainable = False


# build a classifier model to put on top of the convolutional model
# we add dense layers so that the model can learn more complex functions
model = tf.keras.Sequential()
model.add(vgg)
model.add(tf.keras.layers.Flatten())

# dense layers
model.add(tf.keras.layers.Dense(units=512, activation='relu'))
model.add(tf.keras.layers.Dense(units=512, activation='relu'))

# final layer with softmax activation
model.add(tf.keras.layers.Dense(units=data.num_classes, activation='softmax'))

# Visualize created model as a table
model.summary()

# Visualize initialized weights
print("model.weights", model.weights)


# Prepare the model for training
# ------------------------------
loss = tf.keras.losses.CategoricalCrossentropy()

# learning rate
lr = 1e-4
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

metrics = ['accuracy']  # validation metrics to monitor

model.compile(optimizer=optimizer,
              loss=loss,
              metrics=metrics)


# Fine-tune the model
epochs = 100
step_size_train = train_generator.n // train_generator.batch_size
trained_model = model.fit_generator(generator=train_generator,
                                    steps_per_epoch=step_size_train,
                                    epochs=epochs)


# history contains a trace of the loss and any other metrics specified during the compilation of the model
print('\nhistory dict:', trained_model.history)


# Model evaluation
# ----------------

eval_out = model.evaluate_generator(valid_generator,
                                    steps=len(valid_generator),
                                    verbose=0)

print('eval_out', eval_out)

# Check Performance
data.visualize_performance(trained_model)


# Generate predictions
# -------------------
predictions = input('\nCompute and save predictions?: ' 'y - Yes  n - No\n')

if predictions == 'y':
    data.generate_predictions(model, model_name)