# !/usr/bin/env python3
#  -*- coding utf-8 -*-

import os
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from datetime import datetime

# fix the seed for random operations to make experiments reproducible
seed = 123
tf.random.set_seed(seed)

cwd = os.getcwd()
dataset_dir = os.path.join(cwd,
                           'image_classification/dataset')  # path to dataset
train_dir = os.path.join(dataset_dir, 'training')
test_dir = os.path.join(dataset_dir, 'test')

# input image dimensions
img_w = 256
img_h = 256

# number of input channels (color space)
channels = 3  # rgb

# input shape
input_shape = (img_h, img_w, channels)

class_list = [
    'owl',  # 0
    'galaxy',  # 1
    'lightning',  # 2
    'wine-bottle',  # 3
    't-shirt',  # 4
    'waterfall',  # 5
    'sword',  # 6
    'school-bus',  # 7
    'calculator',  # 8
    'sheet-music',  # 9
    'airplanes',  # 10
    'lightbulb',  # 11
    'skyscraper',  # 12
    'mountain-bike',  # 13
    'fireworks',  # 14
    'computer-monitor',  # 15
    'bear',  # 16
    'grand-piano',  # 17
    'kangaroo',  # 18
    'laptop'  # 19
]

num_classes = len(class_list)  # 20

# number of training samples to feed the NN at each training step
batch_size = 16  # 8, 32, 64             # training size: 1247 samples

# batch size: 16 samples/iteration
# more or less 78 iterations/epochh


# Create image generators from directory
# --------------------------------------
def setup_data_generator(with_data_augmentation=True,
                         create_test_generator=False):
    # the data, split between train and test sets

    # NOTE: splitting is done with 'flow_from_directory(…, subset=training/validation)
    # The fixed random seed is enough to reproduce the splitting.

    # fraction of images reserved for validation
    valid_split = 0.2

    # define data augmentation configuration
    apply_data_augmentation = with_data_augmentation
    if apply_data_augmentation:

        train_data_gen = ImageDataGenerator(
            rescale=1. / 255,  # every pixel value from range [0,255] -> [0,1]
            shear_range=0.2,
            zoom_range=0.2,
            rotation_range=45,
            horizontal_flip=True,
            vertical_flip=True,
            validation_split=valid_split)

    else:
        train_data_gen = ImageDataGenerator(rescale=1. / 255,
                                            validation_split=valid_split)

    print('\ntrain_gen ... ')
    train_generator = train_data_gen.flow_from_directory(
        train_dir,
        subset='training',  # subset of data
        batch_size=batch_size,
        target_size=(img_w, img_h),  # images are automatically resized
        color_mode='rgb',
        classes=class_list,
        class_mode='categorical',
        shuffle=True,
        seed=seed)

    print('\nvalid_gen ... ')
    valid_generator = train_data_gen.flow_from_directory(
        train_dir,
        subset='validation',
        batch_size=batch_size,
        target_size=(img_w, img_h),
        color_mode='rgb',
        classes=class_list,
        class_mode='categorical',
        shuffle=False,
        seed=seed)

    get_input_params_from_generator(train_generator)

    if create_test_generator:
        test_generator = create_test_data_generator()
        return train_generator, valid_generator, test_generator

    return train_generator, valid_generator


def setup_data_generator_using_split_folders(with_data_augmentation=True,
                                             create_test_generator=False):
    apply_data_augmentation = with_data_augmentation

    import split_folders

    dataset_split_dir = os.path.join(dataset_dir, 'dataset_split')

    # define data augmentation configuration
    if apply_data_augmentation:

        train_data_gen = ImageDataGenerator(
            rescale=1. / 255,  # every pixel value from range [0,255] -> [0,1]
            shear_range=0.2,
            zoom_range=0.2,
            rotation_range=45,
            horizontal_flip=True,
            vertical_flip=True)
    else:
        train_data_gen = ImageDataGenerator(rescale=1. / 255)

    valid_data_gen = ImageDataGenerator(rescale=1. / 255)

    # setup generators
    print('\ntrain_gen ... ')
    train_generator = train_data_gen.flow_from_directory(
        os.path.join(dataset_split_dir, 'train'),
        batch_size=batch_size,
        target_size=(img_w, img_h),  # images are automatically resized
        color_mode='rgb',  # read all pictures as rgb
        classes=class_list,
        class_mode='categorical',
        shuffle=True,
        seed=seed)

    print('\nvalid_gen ... ')
    valid_generator = valid_data_gen.flow_from_directory(
        os.path.join(dataset_split_dir, 'val'),
        batch_size=batch_size,
        target_size=(img_w, img_h),
        color_mode='rgb',
        classes=class_list,
        class_mode='categorical',
        shuffle=False,
        seed=seed)

    get_input_params_from_generator(train_generator)

    if create_test_generator:
        test_generator = create_test_data_generator()
        return train_generator, valid_generator, test_generator

    return train_generator, valid_generator


# Extract num_classes and channels directly from generator
# --------------------------------------------------------
def get_input_params_from_generator(generator):
    images, labels = next(generator)

    global num_classes, channels

    # inputs (x_train)
    channels = images.shape[3]
    print("channels", channels)

    # labels (y_labels)
    num_classes = labels.shape[1]
    print("num_classes", num_classes)


# Create dataset objects from generators
# --------------------------------------
def setup_dataset():  # useful for small datasets (that can fit in memory)
    train_generator, valid_generator = setup_data_generator()

    train_dataset = dataset_from_generator(train_generator, num_classes)
    train_dataset = train_dataset.repeat()

    valid_dataset = dataset_from_generator(valid_generator, num_classes)
    valid_dataset = valid_dataset.repeat()

    return train_dataset, valid_dataset


# Create dataset from generator
# -----------------------------
def dataset_from_generator(generator,
                           classes,
                           img_height=img_h,
                           img_width=img_w,
                           img_channels=channels):
    dataset = tf.data.Dataset.from_generator(
        lambda: generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=([None, img_height, img_width,
                        img_channels], [None, classes]))
    return dataset


# Generator for test directory that doesn’t have subdirectories
# the classes of those images are unknown
# -------------------------------------------------------------
def create_test_data_generator():
    test_data_gen = ImageDataGenerator(rescale=1. / 255)

    print('\ntest_gen ... ')

    test_generator = test_data_gen.flow_from_directory(
        dataset_dir,  # specify the parent dir of the test dir
        batch_size=batch_size,
        target_size=(img_w, img_h),
        color_mode='rgb',
        classes=['test'],  # load the test “class”
        # to yield the images in “order”, to predict the outputs
        # and match them with their unique ids or filenames
        shuffle=False,
        seed=seed)

    return test_generator


# Iterate Dataset object to access samples inside it
# --------------------------------------------------
def show_batch(train_data):
    import matplotlib.pyplot as plt
    import numpy as np

    iterator = iter(train_data)
    image_batch, label_batch = next(iterator)

    plt.figure()  # figsize=(10, 10))

    # create grid of subplots
    for i in range(1, 9):
        # create an axes object in the figure (n_rows, n_cols, plot_id)
        plt.subplot(3, 3, i)

        # plot raw pixel data
        image = image_batch[i]  # i-th image
        image = image * 255  # denormalize
        plt.imshow(np.uint8(image))

        # label = tf.where(label_batch[i] == 1)
        # # plt.title(class_list[label])
        plt.axis('off')

    plt.show()  # show the figure


# Visualize accuracy and loss over time
# -------------------------------------
def visualize_performance(trained_model):
    # plt.plot(trained_model.history)

    accuracy = trained_model.history['accuracy']
    validation_accuracy = trained_model.history['val_accuracy']
    loss = trained_model.history['loss']
    validation_loss = trained_model.history['val_loss']

    epochs = range(len(accuracy))

    # Visualize History for Loss.
    plt.title('Model loss')
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, validation_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend(['training', 'validation'], loc='upper right')
    plt.show()

    # # Visualize History for Accuracy.
    plt.title('Model accuracy')
    plt.plot(epochs, accuracy, 'b', label='Training acc')
    plt.plot(epochs, validation_accuracy, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend(['training', 'validation'], loc='lower right')
    plt.show()


# Compute predictions (probabilities -- the output of the last layer)
# -------------------------------------------------------------------
def generate_predictions(model, model_name):
    results = {}
    results_str = {}

    image_filenames = next(
        os.walk(test_dir))[2]  # s[:10] predict until 10th image

    for filename in image_filenames:
        img = Image.open(os.path.join(test_dir,
                                      filename)).convert('RGB')  # open as RGB
        img = img.resize((img_h, img_w))  # target size

        # data_normalization
        img_array = np.array(img)  #
        img_array = img_array * 1. / 255  # normalization
        img_array = np.expand_dims(img_array,
                                   axis=0)  # to fix dims of input in the model

        print("prediction for {}...".format(filename))
        predictions = model.predict(img_array)

        # Get predicted class as the index corresponding to the maximum value in the vector probability
        predicted_class = np.argmax(predictions,
                                    axis=-1)  # multiple categories
        predicted_class = predicted_class[0]

        results[filename] = predicted_class
        results_str[filename] = class_list[predicted_class]

    create_csv(results, model_name)

    # Prints the nicely formatted dictionary
    from pprint import pprint
    pprint(results_str)

    print('Num. of labeled images', results.__len__())


# Create submission csv file
# --------------------------
def create_csv(results, model_name):
    print("\nGenerating submission csv ... ")

    # save on a different dir according to the classifier used
    results_dir = 'image_classification/submissions/' + model_name

    # If directory for the classifier does not exist, create
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    csv_fname = 'results_'
    csv_fname += datetime.now().strftime('%b%d_%H-%M-%S') + '.csv'

    with open(os.path.join(results_dir, csv_fname), 'w') as f:

        fieldnames = 'Id,Category'
        f.write(fieldnames + '\n')

        for key, value in results.items():
            f.write(key + ',' + str(value) + '\n')


def save_model_weights(model, model_filename):
    model.save_weights(model_filename + '.weights', save_format="tf")
