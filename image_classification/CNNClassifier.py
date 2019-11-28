# !/usr/bin/env python3.6

import tensorflow as tf
from keras import activations


class CNNClassifier(tf.keras.Model):
    def __init__(self, depth, num_filters, num_classes):
        super(CNNClassifier, self).__init__()

        self.feature_extractor = tf.keras.Sequential()

        # Convolutional layers
        for i in range(depth):
            self.feature_extractor.add(ConvBlock(num_filters=num_filters))
            num_filters *= 2

        # Flatten convolutional result so we can feed data to fully connected layers
        self.flatten = tf.keras.layers.Flatten()
        self.classifier = tf.keras.Sequential()

        # Fully connected
        self.classifier.add(tf.keras.layers.Dense(units=512,
                                                  activation='relu'))
        self.classifier.add(tf.keras.layers.Dense(units=512,
                                                  activation='relu'))
        self.classifier.add(tf.keras.layers.Dense(units=512,
                                                  activation='relu'))
        # Dropout Layer
        self.feature_extractor.add(tf.keras.layers.Dropout(0.3))

        # Output Layer
        self.classifier.add(
            # One unit for each class.
            # Use softmax activation because it's a non-binary classification problem
            tf.keras.layers.Dense(units=num_classes, activation='softmax'))

    def call(self, inputs):
        x = self.feature_extractor(inputs)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


# Keras Model subclassing
# -----------------------
# Create convolutional block
class ConvBlock(tf.keras.Model):
    def __init__(self, num_filters):
        super(ConvBlock, self).__init__()
        self.conv2d = tf.keras.layers.Conv2D(
            filters=num_filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            # padding the input such that the output
            # has the same length as the original input
            padding='same')

        # we can specify the activation function directly in Conv2D
        self.activation = tf.keras.layers.ReLU()
        self.pooling = tf.keras.layers.MaxPool2D(pool_size=(2, 2))

    def call(self, inputs):
        x = self.conv2d(inputs)
        x = self.activation(x)
        # x = self.dropout(x)
        x = self.pooling(x)
        return x
