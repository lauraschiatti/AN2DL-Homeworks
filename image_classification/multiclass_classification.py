# !/usr/bin/env python3
#  -*- coding utf-8 -*-

import tensorflow as tf

from utils import data_manager as data

# Fix the seed for random operations
# ----------------------------------
seed = 123
tf.random.set_seed(seed)

# Parameters
hidden_layer_units = 512
epochs = 20

# Data loader
# -----------
train_generator, valid_generator = data.setup_data_generator()
# data.show_batch(train_generator)


# Create model
# ------------
model_name = 'Multilayer-perceptron'

# We decided to use se Sequential model because it is simpler
sequential_model = True
x_shape = (data.img_h, data.img_w, data.channels)
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

callbacks = []
if with_early_stop:
    callbacks.append(
        tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                         patience=epochs * 0.2))

# early stop with patience of 20% the epochs
trained_model = model.fit_generator(generator=train_generator,
                                    epochs=epochs,
                                    steps_per_epoch=800,
                                    callbacks=callbacks,
                                    validation_data=valid_generator,
                                    validation_steps=200)

# history contains a trace of the loss and any other metrics specified during the compilation of the model
print('\nhistory dict:', trained_model.history)


# Check Performance
data.visualize_performance(trained_model)

# Model evaluation
# ----------------
# model.load_weights('/path/to/checkpoint')  # use this if you want to restore saved model

eval_out = model.evaluate_generator(valid_generator,
                                    steps=len(valid_generator),
                                    verbose=0)
# test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('eval_out', eval_out)

# Check Performance
data.visualize_performance(trained_model)

# Generate predictions
# -------------------
predictions = input('\nCompute and save predictions?: ' 'y - Yes  n - No\n')

if predictions == 'y':
    data.generate_predictions(model, model_name)
