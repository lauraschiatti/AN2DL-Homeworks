# !/usr/bin/env python3.6
#  -*- coding: utf-8 -*-

import tensorflow as tf

import utils.data_loader as data


# -------------------------------------- #
	### Multi-class Classification ###
# -------------------------------------- #


# Fix the seed for random operations
# ----------------------------------
seed = 123
tf.random.set_seed(seed)


# todo: gpu management
# Set GPU memory growth
# ---------------------


# Data loader
# -----------
train_generator, valid_generator, test_generator = data.setup_data_generator()

data.show_batch(train_generator)


# Create model
# ------------

# config = model.get_config()  ## dict with model config
# model = Model.from_config(config)
# or, for Sequential:
# model = Sequential.from_config(config)


##----------##
# x = tf.keras.Input(shape=[256, 256])  # input tensor
#
# flatten = tf.keras.layers.Flatten()(x)
# h = tf.keras.layers.Dense(units=10, activation=tf.keras.activations.sigmoid)(flatten)  # hidden layers
# #output layer:probabccc of belonging to each class
# out = tf.keras.layers.Dense(units=10, activation=tf.keras.activations.softmax)(h)
# model = tf.keras.Model(inputs=x, outputs=out)
#


# Visualize created model as a table
# model.summary()
# print('model initial weights', model.weights)



# Specify the training configuration (optimizer, loss, metrics)
# -------------------------------------------------------------
loss = tf.keras.losses.CategoricalCrossentropy() # loss function to minimize

lr = 1e-4 # learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=lr) # stochastic gradient descent optimizer

metrics = ['accuracy'] # validation metrics to monitor

# Compile Model
model.compile(optimizer=optimizer,
				loss=loss,
				metrics=metrics)



# Train the model
# ---------------
# todo: check fit_generator
# epochs = 10
#
# step_size_train = train_gen.n // train_gen.batch_size  # num_train_samples/bs
# step_size_valid = valid_gen.n // valid_gen.batch_size
#
# fit_model = model.fit_generator(generator= train_gen,
# 								steps_per_epoch=step_size_train,
# 								epochs=epochs,
# 								validation_data=valid_gen, # validation generator
# 								validation_steps=step_size_valid)
# 								# callbacks=[checkpointer, stopper]
# 							  	# shuffle=True)


# example
# model.fit_generator(
#     train_generator,
#     steps_per_epoch = train_generator.samples // batch_size,
#     validation_data = validation_generator,
#     validation_steps = validation_generator.samples // batch_size,
#     epochs = nb_epochs)

fit_model = model.fit_generator(train_generator,
								steps_per_epoch=2000,
								epochs=50,
								validation_data=valid_generator,
								validation_steps=800)


# history lds a record of the loss values and metric values during training
print('\nhistory dict:', fit_model.history)


# Evaluate the model
# ----------

# print('Evaluate model on test data ... ')
# eval_out = model.evaluate_generator(generator=valid_gen,
# 									 	steps=step_size_valid)
#										verbose=0)

# print('test loss, test acc:', eval_out)


# Check Performance
# print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
#
# epochs = range(len(acc))
#
# plt.plot(epochs, acc, 'b', label='Training acc')
# plt.plot(epochs, val_acc, 'r', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.legend()
#
# plt.figure()
#
# plt.plot(epochs, loss, 'b', label='Training loss')
# plt.plot(epochs, val_loss, 'r', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
#
# plt.show()


# Predict output
# --------------

# step_size_test = test_gen.n // test_gen.batch_size

# reset the test_generator before whenever you call the predict_generator.
# This is important, if you forget to reset the test_generator you will get outputs in a weird order.
# test_gen.reset()
#
# pred = model.predict_generator(generator=test_gen,
# 									steps=step_size_test,
# 									verbose=1)
#
#
# predicted_class_indices=np.argmax(pred,axis=1) # predicted abels


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
# 						   "Predictions": predictions})
# results.to_csv("results.csv", index=False)


