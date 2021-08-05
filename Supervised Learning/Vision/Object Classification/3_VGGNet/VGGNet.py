import tensorflow as tf
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint

(train_x, train_y), (test_x, test_y) = tf.keras.datasets.fashion_mnist.load_data()
train_x = train_x / 255.0
test_x = test_x / 255.0
train_x = tf.expand_dims(train_x, 3)
test_x = tf.expand_dims(test_x, 3)
val_x = train_x[:5000]
val_y = train_y[:5000]

input_shape = train_x[0].shape
lenet5_model = Sequential()
lenet5_model.add(Conv2D(filters=64, kernel_size=(3,3), strides=1, padding="same", activation="relu", input_shape=input_shape))
lenet5_model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding="same", activation='relu'))
lenet5_model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
lenet5_model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding="same", activation='relu'))
lenet5_model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding="same", activation='relu'))
lenet5_model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
lenet5_model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding="same", activation="relu"))
lenet5_model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding="same", activation="relu"))
lenet5_model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding="same", activation="relu"))
lenet5_model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
lenet5_model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding="same", activation="relu"))
lenet5_model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding="same", activation="relu"))
lenet5_model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding="same", activation="relu"))
lenet5_model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
lenet5_model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding="same", activation='relu'))
lenet5_model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding="same", activation='relu'))
lenet5_model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding="same", activation='relu'))
lenet5_model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
lenet5_model.add(Flatten())
lenet5_model.add(Dense(4096, activation = 'relu'))
lenet5_model.add(Dense(4096, activation = 'relu'))
lenet5_model.add(Dense(1000, activation = 'softmax'))
lenet5_model.summary()
lenet5_model.compile(optimizer='adam', loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

cp = ModelCheckpoint("./weights.h5", save_weights_only=True, save_best_only=True, monitor="val_accuracy")
hist = lenet5_model.fit(x=train_x, y=train_y, epochs=10, validation_data=(val_x, val_y), callbacks=[cp])

lenet5_model.evaluate(test_x, test_y)

fig, loss_ax = plt.subplots()
loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.legend(loc='upper left')
plt.show()

fig, acc_ax = plt.subplots()
acc_ax.plot(hist.history['accuracy'], 'b', label='train accuracy')
acc_ax.plot(hist.history['val_accuracy'], 'g', label='val accuracy')
acc_ax.set_ylabel('accuracy')
acc_ax.legend(loc='upper left')
plt.show()
