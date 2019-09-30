import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Add, DepthwiseConv2D
NUM_CATEGORY = 10

def shuffle(data_im, data_labels):
    shuffled_index = np.arange(len(data_labels))
    np.random.shuffle(shuffled_index)
    return (data_im[shuffled_index], data_labels[shuffled_index])


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

def cust_block(model, filter, out):
    model.add(Conv2D(64, (1,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))


    model.add(DepthwiseConv2D((filter,filter)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(out, (1,1)))
    model.add(BatchNormalization())
    model.add(Dropout(.5))
    return

if __name__ == "__main__":
    (train_images, train_labels), (test_images, test_labels) = get_data()
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']


    model = Sequential()

    lamb = .00001
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=train_images.shape[1:]))
    model.add(Dropout(.5))
    #model.add(Conv2D(16, kernel_size=(2, 2), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(lamb),
    #                 activity_regularizer=tf.keras.regularizers.l2(lamb)))
    #model.add(Dropout(.25))
    cust_block(model, 3, 16)
    cust_block(model, 3, 24)
    cust_block(model, 3, 24)
    cust_block(model, 5, 48)
    cust_block(model, 5, 48)
    cust_block(model, 3, 96)
    #cust_block(model, 3, 96)
    #cust_block(model, 3, 96)
    #cust_block(model, 5, 96)
    #cust_block(model, 5, 96)
    '''
    model.add(Conv2D(32, kernel_size =(2,2),strides=(3,3), activation='relu'))
    
    cust_block(model, 5, 192)
    cust_block(model, 5, 192)
    cust_block(model, 5, 192)
    cust_block(model, 5, 192)
    cust_block(model, 5, 192)
    cust_block(model, 3, 320)
    '''
    model.add(Conv2D(96, (1,1), activation='relu'))
    #model.add(tf.keras.applications.MobileNetv2(include_top=False))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(NUM_CATEGORY, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(lamb)))
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.RMSprop(),
                  metrics=['accuracy'])


    model.summary()
    model_log = model.fit(train_images, train_labels, batch_size=256, epochs=32, verbose=2, validation_split=.1)



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

