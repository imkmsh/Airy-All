import \
    tensorflow as tf
import \
    matplotlib.pyplot as plt
import \
    numpy as np
import \
    pathlib
from tensorflow.keras.callbacks import \
    ModelCheckpoint

# 1. 데이터
(
train_x,
train_y), (
test_x,
test_y) = tf.keras.datasets.fashion_mnist.load_data()
train_x = train_x / 255.0
test_x = test_x / 255.0
train_x = tf.expand_dims(train_x, 3)
test_x = tf.expand_dims(test_x, 3)
val_x = train_x[:5000]
val_y = train_y[:5000]

batch_size = 32
img_height = 180
img_width = 180

num_of_classes = 4


# 2. 모델 빌드: resnet50
def resnet50_model_conv1(x):
    x = tf.keras.layers.ZeroPadding2D(padding=3)(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=7, strides=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.ZeroPadding2D(padding=1)(x)

    return x


def resnet50_model_conv2(x):
    org_x = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2)(x)

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='valid')(org_x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=3,
        strides=1,
        padding='same')(
        x)
    x = tf.keras.layers.BatchNormalization()(
        x)
    x = tf.keras.layers.Activation(
        'relu')(
        x)

    x = tf.keras.layers.Conv2D(
        filters=256,
        kernel_size=1,
        strides=1,
        padding='valid')(
        x)
    x = tf.keras.layers.BatchNormalization()(
        x)

    org_x = tf.keras.layers.Conv2D(
        filters=256,
        kernel_size=1,
        strides=1,
        padding='valid')(
        org_x)
    org_x = tf.keras.layers.Activation(
        'relu')(
        org_x)

    x = tf.keras.layers.Add()(
        [
            x,
            org_x])
    x = tf.keras.layers.Activation(
        'relu')(
        x)
    x2 = x

    for i in range(
            2):
        x = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=1,
            strides=1,
            padding='valid')(
            x)
        x = tf.keras.layers.BatchNormalization()(
            x)
        x = tf.keras.layers.Activation(
            'relu')(
            x)

        x = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=3,
            strides=1,
            padding='same')(
            x)
        x = tf.keras.layers.BatchNormalization()(
            x)
        x = tf.keras.layers.Activation(
            'relu')(
            x)

        x = tf.keras.layers.Conv2D(
            filters=256,
            kernel_size=1,
            strides=1,
            padding='valid')(
            x)
        x = tf.keras.layers.BatchNormalization()(
            x)

        x = tf.keras.layers.Add()(
            [
                x,
                x2])
        x = tf.keras.layers.Activation(
            'relu')(
            x)
        x2 = x

    return x


def resnet50_model_conv3(x):
    org_x = x

    x = tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=1,
        strides=2,
        padding='valid')(
        org_x)
    x = tf.keras.layers.BatchNormalization()(
        x)
    x = tf.keras.layers.Activation(
        'relu')(
        x)

    x = tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=3,
        strides=1,
        padding='same')(
        x)
    x = tf.keras.layers.BatchNormalization()(
        x)
    x = tf.keras.layers.Activation(
        'relu')(
        x)

    x = tf.keras.layers.Conv2D(
        filters=512,
        kernel_size=1,
        strides=1,
        padding='valid')(
        x)
    x = tf.keras.layers.BatchNormalization()(
        x)

    ano_x = tf.keras.layers.Conv2D(
        filters=512,
        kernel_size=1,
        strides=2,
        padding='valid')(
        org_x)
    ano_x = tf.keras.layers.BatchNormalization()(
        ano_x)

    x = tf.keras.layers.Add()(
        [
            x,
            ano_x])
    x = tf.keras.layers.Activation(
        'relu')(
        x)
    x2 = x

    for i in range(
            3):
        x = tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=1,
            strides=1,
            padding='valid')(
            x)
        x = tf.keras.layers.BatchNormalization()(
            x)
        x = tf.keras.layers.Activation(
            'relu')(
            x)

        x = tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=3,
            strides=1,
            padding='same')(
            x)
        x = tf.keras.layers.BatchNormalization()(
            x)
        x = tf.keras.layers.Activation(
            'relu')(
            x)

        x = tf.keras.layers.Conv2D(
            filters=512,
            kernel_size=1,
            strides=1,
            padding='valid')(
            x)
        x = tf.keras.layers.BatchNormalization()(
            x)
        x = tf.keras.layers.Activation(
            'relu')(
            x)

        x = tf.keras.layers.Add()(
            [
                x,
                x2])
        x = tf.keras.layers.Activation(
            'relu')(
            x)
        x2 = x

    return x


def resnet50_model_conv4(x):
    org_x = x

    x = tf.keras.layers.Conv2D(
        filters=256,
        kernel_size=1,
        strides=2,
        padding='valid')(
        org_x)
    x = tf.keras.layers.BatchNormalization()(
        x)
    x = tf.keras.layers.Activation(
        'relu')(
        x)

    x = tf.keras.layers.Conv2D(
        filters=256,
        kernel_size=3,
        strides=1,
        padding='same')(
        x)
    x = tf.keras.layers.BatchNormalization()(
        x)
    x = tf.keras.layers.Activation(
        'relu')(
        x)

    x = tf.keras.layers.Conv2D(
        filters=1024,
        kernel_size=1,
        strides=1,
        padding='valid')(
        x)
    x = tf.keras.layers.BatchNormalization()(
        x)

    ano_x = tf.keras.layers.Conv2D(
        filters=1024,
        kernel_size=1,
        strides=2,
        padding='valid')(
        org_x)
    ano_x = tf.keras.layers.BatchNormalization()(
        ano_x)

    x = tf.keras.layers.Add()(
        [
            x,
            ano_x])
    x = tf.keras.layers.Activation(
        'relu')(
        x)
    x2 = x

    for i in range(
            5):
        x = tf.keras.layers.Conv2D(
            filters=256,
            kernel_size=1,
            strides=1,
            padding='valid')(
            x)
        x = tf.keras.layers.BatchNormalization()(
            x)
        x = tf.keras.layers.Activation(
            'relu')(
            x)

        x = tf.keras.layers.Conv2D(
            filters=256,
            kernel_size=3,
            strides=1,
            padding='same')(
            x)
        x = tf.keras.layers.BatchNormalization()(
            x)
        x = tf.keras.layers.Activation(
            'relu')(
            x)

        x = tf.keras.layers.Conv2D(
            filters=1024,
            kernel_size=1,
            strides=1,
            padding='valid')(
            x)
        x = tf.keras.layers.BatchNormalization()(
            x)
        x = tf.keras.layers.Activation(
            'relu')(
            x)

        x = tf.keras.layers.Add()(
            [
                x,
                x2])
        x = tf.keras.layers.Activation(
            'relu')(
            x)
        x2 = x

    return x


def resnet50_model_conv5(x):
    org_x = x

    x = tf.keras.layers.Conv2D(
        filters=512,
        kernel_size=1,
        strides=2,
        padding='valid')(
        org_x)
    x = tf.keras.layers.BatchNormalization()(
        x)
    x = tf.keras.layers.Activation(
        'relu')(
        x)

    x = tf.keras.layers.Conv2D(
        filters=512,
        kernel_size=1,
        strides=1,
        padding='same')(
        x)
    x = tf.keras.layers.BatchNormalization()(
        x)
    x = tf.keras.layers.Activation(
        'relu')(
        x)

    x = tf.keras.layers.Conv2D(
        filters=2048,
        kernel_size=1,
        strides=1,
        padding='valid')(
        x)
    x = tf.keras.layers.BatchNormalization()(
        x)

    ano_x = tf.keras.layers.Conv2D(
        filters=2048,
        kernel_size=1,
        strides=2,
        padding='valid')(
        org_x)
    ano_x = tf.keras.layers.BatchNormalization()(
        ano_x)

    x = tf.keras.layers.Add()(
        [
            x,
            ano_x])
    x = tf.keras.layers.Activation(
        'relu')(
        x)
    x2 = x

    for i in range(
            2):
        x = tf.keras.layers.Conv2D(
            filters=512,
            kernel_size=1,
            strides=1,
            padding='valid')(
            x)
        x = tf.keras.layers.BatchNormalization()(
            x)
        x = tf.keras.layers.Activation(
            'relu')(
            x)

        x = tf.keras.layers.Conv2D(
            filters=512,
            kernel_size=3,
            strides=1,
            padding='same')(
            x)
        x = tf.keras.layers.BatchNormalization()(
            x)
        x = tf.keras.layers.Activation(
            'relu')(
            x)

        x = tf.keras.layers.Conv2D(
            filters=2048,
            kernel_size=1,
            strides=1,
            padding='valid')(
            x)
        x = tf.keras.layers.BatchNormalization()(
            x)
        x = tf.keras.layers.Activation(
            'relu')(
            x)

        x = tf.keras.layers.Add()(
            [
                x,
                x2])
        x = tf.keras.layers.Activation(
            'relu')(
            x)
        x2 = x

    return x


def whole_resnet50_model(x):
    x = resnet50_model_conv1(x)
    x = resnet50_model_conv2(x)
    x = resnet50_model_conv3(x)
    x = resnet50_model_conv4(x)
    x = resnet50_model_conv5(x)
    x = tf.keras.layers.GlobalAvgPool2D()(x)
    x = tf.keras.layers.Dense(num_of_classes, activation='softmax')(x)

    return x

output = whole_resnet50_model(train_x)

resnet50_model = tf.keras.models.Model(test_x, output)


# 3. 모델 학습
resnet50_model.compile(optimizer='adam', loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

cp = ModelCheckpoint("./weights,h5", save_weights_only=True, save_best_only=True, monitor="val_accuracy")
hist = resnet50_model.fit(x=train_x, y=train_y, epochs=10, validation_data=(val_x, val_y), callbacks=[cp])


# 4. 모델 평가
resnet50_model.evaluate(
    test_x,
    test_y)

# 5. 모델 배포
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