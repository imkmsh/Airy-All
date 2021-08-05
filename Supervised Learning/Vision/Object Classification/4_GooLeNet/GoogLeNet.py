import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, concatenate, Flatten, Dropout
from keras.models import Model
from keras.regularizers import l2
import imageio
from PIL import Image


def inception_block(input_layer, filter1x1, filter3x3, filter5x5, reduce3x3, reduce5x5, pool_proj,
                    activation='relu'):
    conv1x1 = Conv2D(filter1x1, kernel_size=(1, 1), padding='same', activation=activation)(input_layer)

    conv3x3_reduce = Conv2D(reduce3x3, kernel_size=(1, 1), padding='same', activation=activation)(input_layer)

    conv3x3 = Conv2D(filter3x3, kernel_size=(3, 3), padding='same', activation=activation)(conv3x3_reduce)

    conv5x5_reduce = Conv2D(reduce5x5, kernel_size=(1, 1), padding='same', activation=activation)(input_layer)

    conv5x5 = Conv2D(filter5x5, kernel_size=(5, 5), padding='same', activation=activation)(conv5x5_reduce)

    pooling = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_layer)

    pool_proj = Conv2D(pool_proj, kernel_size=(1, 1), padding='same', activation=activation)(pooling)

    output_layer = concatenate([conv1x1, conv3x3, conv5x5, pool_proj])

    return output_layer


def make_googlenet(activation='relu'):
    inputs = Input(shape=(224, 224, 3))

    conv1_7x7_s2 = Conv2D(64, activation=activation, kernel_size=(7, 7), name='conv1/7x7_s2', strides=(2, 2),
                          padding="same")(inputs)

    pool1_3x3_s2 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid', name='pool1/3x3_s2')(conv1_7x7_s2)

    pool1_norm1 = tf.nn.lrn(pool1_3x3_s2)

    conv2_3x3_reduce = Conv2D(64, activation='activation', kernel_size=(1, 1), name='conv2/3x3_reduce',
                              W_regularizer=l2(0.0002), padding='same')(pool1_norm1)

    conv2_3x3 = Conv2D(192, activation='activation', kernel_size=(3, 3), W_regularizer=l2(0.0002), padding='same',
                       name='conv2/3x3')(conv2_3x3_reduce)

    conv2_norm2 = tf.nn.lrn(conv2_3x3)

    pool2_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='valid', name='pool2/3x3_s2')(conv2_norm2)

    inception_3a = inception_block(pool2_3x3_s2, 64, 128, 32, 96, 16, 32)

    inception_3b = inception_block(inception_3a, 128, 192, 96, 128, 32, 64)

    pool3_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='valid', name='pool3/3x3_s2')(inception_3b)

    inception_4a = inception_block(pool3_3x3_s2, 192, 208, 48, 96, 16, 64)

    loss1_ave_pool = AveragePooling2D((5, 5), strides=(3, 3), name='loss1/ave_pool')(inception_4a)

    loss1_conv = Conv2D(128, kernel_size=(1, 1), padding='same', activation=activation, name='loss1/conv',
                        W_regularizer=l2(0.0002))(loss1_ave_pool)

    loss1_flat = Flatten()(loss1_conv)

    loss1_fc = Dense(1024, activation=activation, name='loss1/fc', W_regularizer=l2(0.0002))(loss1_flat)

    loss1_drop_fc = Dropout(0.7)(loss1_fc)

    loss1_classifier_act = Dense(1000, name='loss1/classifier', activation='softmax', W_regularizer=l2(0.0002))(
        loss1_drop_fc)

    inception_4b = inception_block(inception_4a, 160, 224, 64, 112, 24, 64)

    inception_4c = inception_block(inception_4b, 128, 256, 64, 128, 24, 64)

    inception_4d = inception_block(inception_4c, 112, 288, 64, 144, 32, 64)

    loss2_ave_pool = AveragePooling2D((5, 5), strides=(3, 3), name='loss2/ave_pool')(inception_4d)

    loss2_conv = Conv2D(128, kernel_size=(1, 1), padding='same', activation=activation, name='loss2/conv',
                        W_regularizer=l2(0.0002))(loss2_ave_pool)

    loss2_flat = Flatten()(loss2_conv)

    loss2_fc = Dense(1024, activation=activation, name='loss2/fc', W_regularizer=l2(0.0002))(loss2_flat)

    loss2_drop_fc = Dropout(0.7)(loss2_fc)

    loss2_classifier_act = Dense(1000, name='loss2/classifier', activation='softmax', W_regularizer=l2(0.0002))(
        loss2_drop_fc)

    inception_4e = inception_block(inception_4d, 256, 320, 128, 160, 32, 128)

    pool4_3x3_s2 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid', name='pool4/3x3_s2')(inception_4e)

    inception_5a = inception_block(pool4_3x3_s2, 256, 320, 128, 160, 32, 128)

    inception_5b = inception_block(inception_5a, 384, 384, 128, 192, 48, 128)

    pool5_7x7_s1 = AveragePooling2D((7, 7), strides=(1, 1), name='pool5/7x7_s2')(inception_5b)

    loss3_flat = Flatten(pool5_7x7_s1)
    
    pool5_drop_7x7_s1 = Dropout(0.4)(loss3_flat)

    loss3_classifier_act = Dense(1000, name='loss3/classifier', activation='softmax', W_regularizer=l2(0.0002))(
        pool5_drop_7x7_s1)

    googlenet = Model(input=inputs, output=[loss1_classifier_act, loss2_classifier_act, loss3_classifier_act])

    return googlenet


if __name__ == "__main__":
    img = imageio.imread('cat.jpg', pilmode='RGB')
    img = np.array(Image.fromarray(img).resize((224, 224))).astype(np.float32)
    img[:, :, 0] -= 123.68
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 103.939
    img[:, :, [0, 1, 2]] = img[:, :, [2, 1, 0]]
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)

    model = make_googlenet()
    sgd = tf.keras.optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    out = model.predict(img)
