import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint

(train_x, train_y), (test_x, test_y) = tf.keras.datasets.fashion_mnist.load_data()
train_x = train_x / 255.0
test_x = test_x / 255.0
train_x = tf.expand_dims(train_x, 3)
test_x = tf.expand_dims(test_x, 3)
val_x = train_x[:5000]
val_y = train_y[:5000]

alexnet_model = tf.keras.Sequential()
alexnet_model.add(tf.keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=4, padding='valid', input_shape=train_x[0].shape))
alexnet_model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2))
alexnet_model.add(tf.keras.layers.BatchNormalization())
alexnet_model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=1, padding='same', activation='relu'))
alexnet_model.add(tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same'))
alexnet_model.add(tf.keras.layers.BatchNormalization())
alexnet_model.add(tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
alexnet_model.add(tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
alexnet_model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
alexnet_model.add(tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same'))
alexnet_model.add(tf.keras.layers.Flatten())
alexnet_model.add(tf.keras.layers.Dense(units=4096, activation='relu'))
alexnet_model.add(tf.keras.layers.Dropout(0.5))
alexnet_model.add(tf.keras.layers.Dense(units=4096, activation='relu'))
alexnet_model.add(tf.keras.layers.Dropout(0.5))
alexnet_model.add(tf.keras.layers.Dense(units=1000, activation='softmax'))
alexnet_model.compile(optimizer='adam', loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

cp = ModelCheckpoint("./weights.h5", save_weights_only=True, save_best_only=True, monitor="val_accuracy")
hist = alexnet_model.fit(x=train_x, y=train_y, epochs=10, validation_data=(val_x, val_y), callbacks=[cp])

alexnet_model.evaluate(test_x, test_y)

fig, loss_ax = plt.subplots()
loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.legend(loc='upper left')
plt.show()
plt.savefig('loss.png')

fig, acc_ax = plt.subplots()
acc_ax.plot(hist.history['accuracy'], 'b', label='train accuracy')
acc_ax.plot(hist.history['val_accuracy'], 'g', label='val accuracy')
acc_ax.set_ylabel('accuracy')
acc_ax.legend(loc='upper left')
plt.show()
plt.savefig('accuracy.png')
