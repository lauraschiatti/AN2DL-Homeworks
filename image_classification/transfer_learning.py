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

# User a pre-trained network for transfer learning
# Load VGG16 model
# ----------------
vgg = tf.keras.applications.VGG16(weights='imagenet',
                                  include_top=False,
                                  input_shape=(data.img_h, data.img_w,
                                               data.channels))
vgg.summary()
print(vgg.layers)

model_name = 'CNN+TF'

x = vgg.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
# we add dense layers so that the model can learn more complex functions and classify for better results.
x = tf.keras.layers.Dense(1024, activation='relu')(x)
# dense layer 2
x = tf.keras.layers.Dense(1024, activation='relu')(x)
# dense layer 3
x = tf.keras.layers.Dense(512, activation='relu')(x)
# final layer with softmax activation
preds = tf.keras.layers.Dense(data.num_classes, activation='softmax')(x)

model = tf.keras.Model(inputs=vgg.input, outputs=preds)

for layer in model.layers[:20]:
    layer.trainable = False
for layer in model.layers[20:]:
    layer.trainable = True

model.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

epochs = 5
step_size_train = train_generator.n // train_generator.batch_size
trained_model = model.fit_generator(generator=train_generator,
                                    steps_per_epoch=step_size_train,
                                    epochs=epochs)

# history contains a trace of the loss and any other metrics specified during the compilation of the model
print('\nhistory dict:', trained_model.history)

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
