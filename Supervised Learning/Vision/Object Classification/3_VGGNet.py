from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential


input_shape = (224, 224, 3)
vggnet_model = Sequential()


vggnet_model.add(Conv2D(filters=64, kernel_size=(3,3), strides=1, padding="same", activation="relu", input_shape=input_shape))

vggnet_model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding="same", activation='relu'))

vggnet_model.add(MaxPooling2D(pool_size=(2, 2), strides=2))


vggnet_model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding="same", activation='relu'))

vggnet_model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding="same", activation='relu'))

vggnet_model.add(MaxPooling2D(pool_size=(2, 2), strides=2))


vggnet_model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding="same", activation="relu"))

vggnet_model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding="same", activation="relu"))

vggnet_model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding="same", activation="relu"))

vggnet_model.add(MaxPooling2D(pool_size=(2, 2), strides=2))


vggnet_model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding="same", activation="relu"))

vggnet_model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding="same", activation="relu"))

vggnet_model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding="same", activation="relu"))

vggnet_model.add(MaxPooling2D(pool_size=(2, 2), strides=2))


vggnet_model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding="same", activation='relu'))

vggnet_model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding="same", activation='relu'))

vggnet_model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding="same", activation='relu'))

vggnet_model.add(MaxPooling2D(pool_size=(2, 2), strides=2))


vggnet_model.add(Flatten())


vggnet_model.add(Dense(4096, activation = 'relu'))
vggnet_model.add(Dense(4096, activation = 'relu'))
vggnet_model.add(Dense(1000, activation = 'softmax'))
vggnet_model.summary()
