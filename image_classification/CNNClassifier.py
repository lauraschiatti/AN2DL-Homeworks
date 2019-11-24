# !/usr/bin/env python3.6
#  -*- coding: utf-8 -*-

import tensorflow as tf


class CNNClassifier(tf.keras.Model):

    def __init__(self, depth, num_filters, num_classes):
        super(CNNClassifier, self).__init__()

        self.feature_extractor = tf.keras.Sequential()

        for i in range(depth):
            self.feature_extractor.add(ConvBlock(num_filters=num_filters))
            num_filters *= 2

        self.flatten = tf.keras.layers.Flatten()
        self.classifier = tf.keras.Sequential()
        self.classifier.add(tf.keras.layers.Dense(units=512, activation='relu'))
        self.classifier.add(tf.keras.layers.Dense(units=num_classes, activation='softmax'))

    def call(self, inputs):
        x = self.feature_extractor(inputs)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


# Keras Model subclassing
# -----------------------

# Please note that there are many ways to implement a CNN
# Here subclassing is used just for teaching purposes, but you
# can simply write every single layer as usual using Model or Sequential..

# Create convolutional block
class ConvBlock(tf.keras.Model):

    def __init__(self, num_filters):
        super(ConvBlock, self).__init__()
        self.conv2d = tf.keras.layers.Conv2D(filters=num_filters,
                                             kernel_size=(3, 3),
                                             strides=(1, 1),
                                             padding='same')
        self.activation = tf.keras.layers.ReLU()  # we can specify the activation function directly in Conv2D
        self.pooling = tf.keras.layers.MaxPool2D(pool_size=(2, 2))

    def call(self, inputs):
        x = self.conv2d(inputs)
        x = self.activation(x)
        x = self.pooling(x)
        return x
