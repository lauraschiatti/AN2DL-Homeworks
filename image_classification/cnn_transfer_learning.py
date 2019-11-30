import os
import tensorflow as tf
from utils import data_manager as data
from CNNClassifier import CNNClassifier

# Fix the seed for random operations
# ----------------------------------
seed = 123
tf.random.set_seed(seed)

# Data loader
# -----------
train_generator, valid_generator = data.setup_data_generator()

# Use a pre-trained network for transfer learning: train dense layers for new classification task

# ------------------------------- #
#   Load previously trained CNN model
# ------------------------------- #
new_model = CNNClassifier(depth=data.depth,
                          num_filters=data.num_filters,
                          pool_size=data.pool_size,
                          kernel_size=data.kernel_size,
                          num_classes=data.num_classes)

# Build Model (Required)
new_model.build(input_shape=(None, data.img_h, data.img_w, data.channels))

loss = tf.keras.losses.CategoricalCrossentropy()
lr = 1e-4  # learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
metrics = ['accuracy']  # validation metrics to monitor

new_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# Train the model
# ---------------
with_early_stopping = True

callbacks = []
if with_early_stopping:
    callbacks.append(
        tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                         patience=data.epochs * 0.2))

# train on 0 epochs
new_model.fit_generator(generator=train_generator,
                        epochs=0,
                        steps_per_epoch=len(train_generator),
                        validation_data=valid_generator,
                        validation_steps=len(valid_generator))

# Load the state of the old model
model_filename = os.path.join(data.cwd, 'image_classification/models')
new_model.load_weights(model_filename + '/CNN-weights')

######################################

model_name = 'CNN+TF'

# Two types of transfer learning: feature extraction and fine-tuning
fine_tuning = True

if fine_tuning:
    freeze_until = 8  # layer from which we want to fine-tune

    # set the first freeze_until layers (up to the last conv block => depth = 5)
    # to non-trainable (weights will not be updated)
    for layer in new_model.layers[:freeze_until]:
        layer.trainable = False

else:
    new_model.trainable = False

# ------------------------------------------------ #
#   Fine tuning using previously trained model
# ------------------------------------------------ #

# ------------------------------------------------------------------------- #
#   Build a classifier model to put on top of the convolutional model
# ------------------------------------------------------------------------- #

# we add dense layers so that the model can learn more complex functions
model = tf.keras.Sequential()
model.add(vgg)
model.add(tf.keras.layers.Flatten())

# dense layers
model.add(tf.keras.layers.Dense(units=512, activation='relu'))
model.add(tf.keras.layers.Dense(units=512, activation='relu'))
model.add(tf.keras.layers.Dense(units=512, activation='relu'))
model.add(tf.keras.layers.Dense(units=512, activation='relu'))
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

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# Fine-tune the model
epochs = 150
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
