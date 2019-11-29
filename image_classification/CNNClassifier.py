# !/usr/bin/env python3.6
#  -*- coding utf-8 -*-

import tensorflow as tf


class CNNClassifier(tf.keras.Model):

    # Define two groups of layers: feature (convolutions) and classification (dense)

    def __init__(self, depth, num_filters, pool_size, kernel_size, num_classes):
        super(CNNClassifier, self).__init__()

        ## Convolution layers ##
        self.feature_extractor = tf.keras.Sequential()

        for i in range(depth):
            self.feature_extractor.add(ConvBlock(num_filters, pool_size, kernel_size))
            num_filters *= 2

        ## Flatten layer to feed data to fully connected layers ##
        self.flatten = tf.keras.layers.Flatten()  # output of a convolutional layer is a n-D tensor

        ## Classification layers ##
        self.classifier = tf.keras.Sequential()

        # Fully connected
        self.classifier.add(tf.keras.layers.Dense(units=512,
                                                  activation='relu'))
        self.classifier.add(tf.keras.layers.Dense(units=512,
                                                  activation='relu'))
        self.classifier.add(tf.keras.layers.Dense(units=512,
                                                  activation='relu'))
        self.classifier.add(tf.keras.layers.Dropout(0.1))

        # output layer with one unit for each class.
        # Use softmax activation because it's a non-binary classification problem
        self.classifier.add(tf.keras.layers.Dense(units=num_classes,
                                                  activation='softmax'))

    # create complete model = Sequential(feature_layers + classification_layers)
    def call(self, inputs):
        x = self.feature_extractor(inputs)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


# Create convolutional layer
class ConvBlock(tf.keras.Model):

    def __init__(self, num_filters, pool_size, kernel_size):
        super(ConvBlock, self).__init__()

        self.conv2d = tf.keras.layers.Conv2D(filters=num_filters,
                                             kernel_size=kernel_size,  # (3, 3)
                                             # how much you move your filter when doing convolution
                                             strides=(1, 1),
                                             # 0 pad the input such that the output
                                             # has the same dimensions as the original input
                                             padding='same')

        self.activation = tf.keras.layers.ReLU()
        self.pooling = tf.keras.layers.MaxPool2D(pool_size=pool_size)  # (2, 2))
        # self.dropout.add(tf.keras.layers.Dropout(0.25))

    def call(self, inputs):
        x = self.conv2d(inputs)
        x = self.activation(x)
        x = self.pooling(x)
        # x = self.dropout(x)
        return x

# model = tf.keras.Sequential()
# model.add(tf.keras.Input(shape=(img_h, img_w, 3)))
# for i in range(depth):
#     model.add(layers.Conv2D(filters=start_f,
#                             kernel_size=(3, 3),
#                             strides=(1, 1),
#                             padding='same'))
#
#     model.add(layers.ReLU())  # we can specify the activation function directly in Conv2D
#     model.add(layers.MaxPool2D(pool_size=(2, 2)))
#     start_f *= 2
