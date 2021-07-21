import tensorflow as tf
import matplotlib.pyplot as plt

(train_x, train_y), (test_x, test_y) = tf.keras.datasets.fashion_mnist.load_data()

train_x = train_x / 255.0
test_x = test_x / 255.0
train_x = tf.expand_dims(train_x, 3)
test_x = tf.expand_dims(test_x, 3)
val_x = train_x[:5000]
val_y = train_y[:5000]

lenet5_model = tf.keras.Sequential()

lenet5_model.add(
    tf.keras.layers.Conv2D(filters=6, kernel_size=5, activation='tanh', input_shape=train_x[0].shape))  # C1

lenet5_model.add(tf.keras.layers.AveragePooling2D())  # S2

lenet5_model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=5, activation='tanh'))  # C3

lenet5_model.add(tf.keras.layers.AveragePooling2D())  # S4

lenet5_model.add(tf.keras.layers.Flatten())  # Flatten

lenet5_model.add(tf.keras.layers.Dense(units=120, activation='tanh'))  # C5

lenet5_model.add(tf.keras.layers.Dense(units=84, activation='tanh'))  # F6

lenet5_model.add(tf.keras.layers.Dense(units=10, activation='softmax'))  # Output

lenet5_model.compile(optimizer='adam', loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

lenet5_model.fit(train_x, train_y, epochs=10, validation_data=(val_x, val_y))

lenet5_model.evaluate(test_x, test_y)
