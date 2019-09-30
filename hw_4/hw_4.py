import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Add, \
    DepthwiseConv2D

NUM_CATEGORY = 10
lamb = .0001


def shuffle(data_im, data_labels):
    shuffled_index = np.arange(len(data_labels))
    np.random.shuffle(shuffled_index)
    return data_im[shuffled_index], data_labels[shuffled_index]


def format_data(train_images, train_labels):
    train_images = train_images / 255
    train_images = train_images.astype('float32')
    train_labels = train_labels.astype('float32')
    train_labels = tf.keras.utils.to_categorical(train_labels, NUM_CATEGORY)
    return train_images, train_labels


def get_data():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    train_images, train_labels = format_data(train_images, train_labels)
    test_images, test_labels = format_data(test_images, test_labels)

    return shuffle(train_images, train_labels), shuffle(test_images, test_labels)


# from https://towardsdatascience.com/mobilenetv2-inverted-residuals-and-linear-bottlenecks-8a4362f4ffd5
def bottleneck_block(x, filter=3, channels=64, squeeze=16):
    m = Conv2D(channels, (1, 1),data_format="channels_last", kernel_regularizer=tf.keras.regularizers.l2(lamb),
                activity_regularizer=tf.keras.regularizers.l2(lamb))(x)
    m = BatchNormalization()(m)
    m = Activation('relu')(m)
    m = DepthwiseConv2D(30, (filter, filter), data_format="channels_last")(m)
    m = BatchNormalization()(m)
    m = Activation('relu')(m)
    m = Conv2D(128, (1, 1), data_format="channels_last", kernel_regularizer=tf.keras.regularizers.l2(lamb),
                activity_regularizer=tf.keras.regularizers.l2(lamb))(m)
    m = BatchNormalization()(m)
    #x = Add()([m,x])
    return Add()([m,x])


if __name__ == "__main__":
    (train_images, train_labels), (test_images, test_labels) = get_data()
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']


    input = tf.keras.Input(shape=(train_images.shape[1:]))
    x = Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l1(lamb),
                activity_regularizer=tf.keras.regularizers.l1(lamb))(input)
    x = Dropout(.5)(x)
    x = bottleneck_block(x, filter=3, channels=16)
    x = bottleneck_block(x, filter=3, channels= 24)
    x = Dropout(.5)(x)
    x = bottleneck_block(x, filter=3, channels= 24)
    x = bottleneck_block(x, filter=5, channels=40)

    x = bottleneck_block(x, filter=5, channels=40)
    x = bottleneck_block(x, filter=3, channels=80)
    x = Dropout(.5)(x)
    x = bottleneck_block(x, filter=3, channels=80)
    x = bottleneck_block(x, filter=3, channels=80)
    x = bottleneck_block(x, filter=5, channels=112)
    x = Dropout(.5)(x)
    x = bottleneck_block(x, filter=5, channels=112)
    x = bottleneck_block(x, filter=5, channels=112)
    x = bottleneck_block(x, filter=5, channels=192)
    x = bottleneck_block(x, filter=5, channels=192)
    x = Dropout(.5)(x)
    x = bottleneck_block(x, filter=5, channels=192)
    x = bottleneck_block(x, filter=5, channels=192)
    x = bottleneck_block(x, filter=3, channels=320)
    x = Conv2D(64, kernel_size=(1, 1), activation='relu', kernel_regularizer=tf.keras.regularizers.l1(lamb),
                activity_regularizer=tf.keras.regularizers.l2(lamb))(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    output = Dense(NUM_CATEGORY, activation='softmax', kernel_regularizer=tf.keras.regularizers.l1(lamb),
                activity_regularizer=tf.keras.regularizers.l2(lamb))(x)

    model = tf.keras.Model(inputs=input, outputs=output)
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])
    model.summary()
    model_log = model.fit(train_images, train_labels, batch_size=128, epochs=32, verbose=2, validation_split=.1)

    test_stat = model.evaluate(test_images, test_labels)
    print("Test set loss: " + str(test_stat[0]))
    print("Test set accuracy: " + str(test_stat[1]))

    # plotting the metrics
    fig = plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(model_log.history['accuracy'])
    plt.plot(model_log.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='lower right')
    plt.subplot(2, 1, 2)
    plt.plot(model_log.history['loss'])
    plt.plot(model_log.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.tight_layout()

    plt.show()
