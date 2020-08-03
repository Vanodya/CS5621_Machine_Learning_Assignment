# import the libraries needed

from matplotlib import pyplot
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD, Adam
import pandas as pd


# load MNIST data set and split into train and test sets
def load_dataset():
    # load the data set
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # reshape data set to have a single channel
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
    
    # Encode the response values
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # convert from integers to floats
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # normalize to range 0-1
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # return normalized images
    return x_train, y_train, x_test, y_test


# define a CNN model
def define_model():
    # initialize the model
    model = Sequential()

    # Input layer
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())

    # Hidden layer
    model.add(Dense(100, activation='relu'))

    # Output layer
    model.add(Dense(10, activation='softmax'))
    # compile model
    opt1 = SGD(lr=0.01, momentum=0.9)
    opt2 = Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=0.0)
    opt3 = 'rmsprop'

    # Compile the model
    model.compile(optimizer=opt1, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# evaluate the model
def evaluate_model(x_train, y_train, x_test, y_test):

    # create the model
    model = define_model()

    # fit the model
    history = model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test), verbose=0)

    # evaluate the model
    _, acc = model.evaluate(x_test, y_test, verbose=0)
    print('> %.3f' % (acc * 100.0))

    # plot the accuracies over epochs
    pyplot.plot(history.history['accuracy'])
    pyplot.plot(history.history['val_accuracy'])
    pyplot.ylabel('accuracy')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'test'], loc='upper left')
    pyplot.show()

    # save model
    model.save('MNIST_classification_model.h5')

    return history, history.history['val_accuracy']


# Question no 01 [CNN base models]
# ---------------------------------------------------------------------------------------------------------------------
# load data
x_train, y_train, x_test, y_test = load_dataset()

# modelling
history, accuracy_over_epoch = evaluate_model(x_train, y_train, x_test, y_test)
accuracy_over_epoch = pd.DataFrame(accuracy_over_epoch)
accuracy_over_epoch.to_csv("E:/MSc/Sem 02/Machine Learning/Assignment 02/SGD_accuracies_over_epoch_more_layers.csv")
