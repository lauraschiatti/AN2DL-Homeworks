# !/usr/bin/env python3
#  -*- coding utf-8 -*-

import os
import tensorflow as tf
import split_folders
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from datetime import datetime

# fix the seed for random operations to make experiments reproducible

seed = 123
tf.random.set_seed(seed)

cwd = os.getcwd()
dataset_dir = os.path.join(cwd, 'image_classification/dataset') # path to dataset

input_dir = os.path.join(dataset_dir, 'training')
dataset_split_dir = os.path.join(dataset_dir, 'dataset_split')
test_dir = os.path.join(dataset_dir, 'test')

# If split dataset does not exist, create
if not os.path.exists(dataset_split_dir):

    # 80% training, 20% validation
    valid_split = 0.2

    # split into training and validation sets using a ratio . e.g. for train/val `.8 .2`.
    split_folders.ratio(
        input_dir,
        output=dataset_split_dir,
        seed=seed,  # allows to reproduce the split
        ratio=(1 - valid_split, valid_split))

# todo: create config dic with params

# Parameters
# params = {'dim': (32,32,32),
#           'batch_size': 64,
#           'num_classes': 6,
#           'n_channels': 1,
#           'shuffle': True}

# define dimensions of input images @todo: which is the correct size for the images?
img_w = 256
img_h = 256

# define channels (color space)
channels = 3  # rgb

# batch size
batch_size = 32  # (default)

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

# number of classes
num_classes = len(class_list)


# Create image generators from directory
# --------------------------------------
def setup_data_generator(with_data_augmentation=False):
    apply_data_augmentation = with_data_augmentation

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
    test_data_gen = ImageDataGenerator(rescale=1. / 255)

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

    print('\ntest_gen ... ')
    # test directory doesn’t have subdirectories the classes of those images are unknown
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

    # todo: create dataset_split.json file indicating how do you split the training set...

    # get config params from train generator
    images, labels = next(train_generator)

    global num_classes, channels

    # inputs (x_train)
    channels = images.shape[3]
    print("channels", channels)

    # labels (y_labels)
    num_classes = labels.shape[1]
    print("num_classes", num_classes)

    return train_generator, valid_generator, test_generator


# Create dataset objects from generators
# --------------------------------------
def setup_dataset(): # useful for small datasets (that can fit in memory)
    train_generator, valid_generator, test_generator = setup_data_generator()

    train_dataset = dataset_from_generator(train_generator, num_classes)
    train_dataset = train_dataset.repeat()

    valid_dataset = dataset_from_generator(valid_generator, num_classes)
    valid_dataset = valid_dataset.repeat()

    test_dataset = dataset_from_generator(test_generator, num_classes)
    test_dataset = test_dataset.repeat()

    return train_dataset, valid_dataset, test_dataset


# Create dataset from generator
# -----------------------
def dataset_from_generator(generator,
                           classes,
                           img_height=256,
                           img_width=256,
                           img_channels=3):
    dataset = tf.data.Dataset.from_generator(
        lambda: generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=([None, img_height, img_width,
                        img_channels], [None, classes]))
    return dataset


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
        plt.subplot(
            3, 3,
            i)  # create an axes object in the figure (n_rows, n_cols, plot_id)

        # plot raw pixel data
        image = image_batch[i]  # i-th image
        image = image * 255  # denormalize
        plt.imshow(np.uint8(image))

        # label = tf.where(label_batch[i] == 1)
        # # plt.title(class_list[label])
        plt.axis('off')

    plt.show()  # show the figure


def create_multilayer_model():
    which_model = 'base_weight_decay'
    # set_which_model(which_model)  # set model for training_callbacks

    # Create base model using functional API Model (e.g., Input -> Hidden -> Out)
    if which_model == 'model':
        # x = tf.keras.Input(shape=[28, 28])  # input tensor
        # flatten = tf.keras.layers.Flatten()(x)
        # h = tf.keras.layers.Dense(units=10, activation=tf.keras.activations.sigmoid)(flatten)  # hidden layers
        # output layer:probabccc of belonging to each class
        # out = tf.keras.layers.Dense(units=10, activation=tf.keras.activations.softmax)(h)
        # model = tf.keras.Model(inputs=x, outputs=out)

        # equivalent formulation:
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=(28,
                                                       28)))  # or as a list
        model.add(
            tf.keras.layers.Dense(units=10,
                                  activation=tf.keras.activations.sigmoid))
        model.add(
            tf.keras.layers.Dense(units=10,
                                  activation=tf.keras.activations.softmax))

    # Create base model using sequential model (e.g., Input -> Hidden -> Out)
    elif which_model == 'base':
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=(28,
                                                       28)))  # or as a list
        model.add(
            tf.keras.layers.Dense(units=1000,
                                  activation=tf.keras.activations.sigmoid))
        model.add(
            tf.keras.layers.Dense(units=10,
                                  activation=tf.keras.activations.softmax))

    # Create model with Dropout layer
    elif which_model == 'base_dropout':

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=(28,
                                                       28)))  # or as a list
        model.add(
            tf.keras.layers.Dense(units=1000,
                                  activation=tf.keras.activations.sigmoid))
        model.add(tf.keras.layers.Dropout(0.3))  # rate (probab): 0.3
        model.add(
            tf.keras.layers.Dense(units=10,
                                  activation=tf.keras.activations.softmax))

    # Create model with weights penalty (L2 regularization)
    elif which_model == 'base_weight_decay':

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=(28,
                                                       28)))  # or as a list
        model.add(
            tf.keras.layers.Dense(
                units=1000,
                activation=tf.keras.activations.sigmoid,
                kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
        model.add(
            tf.keras.layers.Dense(
                units=10,
                activation=tf.keras.activations.softmax,
                kernel_regularizer=tf.keras.regularizers.l2(0.0001)))

    return model

# Visualize history for loss and accuracy
# ---------------------------------------
def visualize_performance(trained_model):
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
def generate_predictions(model):
    target_size = (img_h, img_w)
    results = {}
    results_str = {}

    image_filenames = next(os.walk(test_dir))[2]  # s[:10] predict until 10th image

    for filename in image_filenames:
        # convert the image to RGB
        img = Image.open(os.path.join(test_dir, filename)).convert('RGB')
        # resize the image
        img = img.resize(target_size)

        # data_normalization - convert to array
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)

        print("prediction for {}...".format(filename))
        predictions = model.predict(img_array * 1 / 255.)

        # Get predicted class as the index corresponding to the maximum value in the vector probability
        predicted_class = np.argmax(predictions, axis=-1)  # multiple categories
        predicted_class = predicted_class[0]

        results[filename] = predicted_class
        results_str[filename] = class_list[predicted_class]

    create_csv(results)

    # Prints the nicely formatted dictionary
    from pprint import pprint
    pprint(results_str)

    print('Num. of labeled images', results.__len__())


# Create submission csv file
# --------------------------
def create_csv(results, classifier='CNN'):
    print("\nGenerating submission csv ... ")

    # save on a different dir according to the classifier used
    results_dir = 'image_classification/submissions/' + classifier

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
