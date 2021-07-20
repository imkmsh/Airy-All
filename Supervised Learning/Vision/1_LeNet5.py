import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

(train_x, train_y), (test_x, test_y) = tf.keras.datasets.fashion_mnist.load_data()

model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(filters=6, kernal_size=(3, 3), activation='relu', input_shape=(32, 32, 1)))

model.add(tf.keras.layers.Conv2D(filters=16, kernal_size=(3, 3), activation='relu'))

model.add(tf.keras.layers.AveragePooling2D())

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(units=120, activation='relu'))

model.add(tf.keras.layers.Dense(units=84, activation='relu'))

model.add(tf.keras.layers.Dense(units=10, activation='softmax'))
