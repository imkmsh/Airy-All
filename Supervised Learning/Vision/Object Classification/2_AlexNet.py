import tensorflow as tf

(train_x, train_y), (test_x, test_y) = tf.keras.datasets.fashion_mnist.load_data()

train_x = train_x / 255.0
test_x = test_x / 255.0
train_x = tf.expand_dims(train_x, 3)
test_x = tf.expand_dims(test_x, 3)
val_x = train_x[:5000]
val_y = train_y[:5000]

alexnet_model = tf.keras.Sequential()

alexnet_model.add(
    tf.keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=4, padding='valid', input_shape=(227, 227, 3)))

alexnet_model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2))

alexnet_model.add(tf.keras.layers.BatchNormalization())

alexnet_model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=1, padding='same', activation='relu'))

alexnet_model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2))

alexnet_model.add(tf.keras.layers.BatchNormalization())

alexnet_model.add(tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))

alexnet_model.add(tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))

alexnet_model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))

alexnet_model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2))

alexnet_model.add(tf.keras.layers.Flatten())

alexnet_model.add(tf.keras.layers.Dense(units=4096, activation='relu'))

alexnet_model.add(tf.keras.layers.Dropout(0.5))

alexnet_model.add(tf.keras.layers.Dense(units=4096, activation='relu'))

alexnet_model.add(tf.keras.layers.Dropout(0.5))

alexnet_model.add(tf.keras.layers.Dense(units=1000, activation='softmax'))

alexnet_model.compile(optimizer='adam', loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

alexnet_model.fit(train_x, train_y, epochs=10, validation_data=(val_x, val_y))

alexnet_model.evaluate(test_x, test_y)
