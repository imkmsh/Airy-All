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

n_classes = 10

# Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
mu = 0
sigma = 0.1

weights = {
    # The shape of the filter weight is (height, width, input_depth, output_depth)
    'conv1': tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean=mu, stddev=sigma)),
    'conv2': tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma)),
    'fl1': tf.Variable(tf.truncated_normal(shape=(5 * 5 * 16, 120), mean=mu, stddev=sigma)),
    'fl2': tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma)),
    'out': tf.Variable(tf.truncated_normal(shape=(84, n_classes), mean=mu, stddev=sigma))
}

biases = {
    # The shape of the filter bias is (output_depth,)
    'conv1': tf.Variable(tf.zeros(6)),
    'conv2': tf.Variable(tf.zeros(16)),
    'fl1': tf.Variable(tf.zeros(120)),
    'fl2': tf.Variable(tf.zeros(84)),
    'out': tf.Variable(tf.zeros(n_classes))
}

# Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
conv1 = tf.nn.conv2d(x, weights['conv1'], strides=[1, 1, 1, 1], padding='VALID')
conv1 = tf.nn.bias_add(conv1, biases['conv1'])
# Activation.
conv1 = tf.nn.relu(conv1)
# Pooling. Input = 28x28x6. Output = 14x14x6.
conv1 = tf.nn.avg_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

# Layer 2: Convolutional. Output = 10x10x16.
conv2 = tf.nn.conv2d(conv1, weights['conv2'], strides=[1, 1, 1, 1], padding='VALID')
conv2 = tf.nn.bias_add(conv2, biases['conv2'])
# Activation.
conv2 = tf.nn.relu(conv2)
# Pooling. Input = 10x10x16. Output = 5x5x16.
conv2 = tf.nn.avg_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

# Flatten. Input = 5x5x16. Output = 400.
fl0 = flatten(conv2)

# Layer 3: Fully Connected. Input = 400. Output = 120.
fl1 = tf.add(tf.matmul(fl0, weights['fl1']), biases['fl1'])
# Activation.
fl1 = tf.nn.relu(fl1)

# Layer 4: Fully Connected. Input = 120. Output = 84.
fl2 = tf.add(tf.matmul(fl1, weights['fl2']), biases['fl2'])
# Activation.
fl2 = tf.nn.relu(fl2)

# Layer 5: Fully Connected. Input = 84. Output = 10.
logits = tf.add(tf.matmul(fl2, weights['out']), biases['out'])
