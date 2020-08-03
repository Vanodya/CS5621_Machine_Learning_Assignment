# import the libraries needed

from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
import numpy as np
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from matplotlib import pyplot
from keras.optimizers import SGD, Adam
from keras.applications.vgg16 import VGG16
import cv2
import pandas as pd
dim = (48, 48)


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


# Function to create model, required for KerasClassifier
def create_model_for_grid_search(optimizer='adam', learn_rate=0.01, momentum=0, neurons = 1):
    # create model

    model = Sequential()

    # input layer
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(neurons, activation='relu'))

    # output layer
    model.add(Dense(10, activation='softmax'))

    # Compile model
    optimizer = SGD(lr=learn_rate, momentum=momentum)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


# transfer learning
def create_model_with_vgg16():
    # initialize the model

    # load pre-trained VGG16 model and specify a new input shape for images
    vgg_model = VGG16(
    	weights='E:/Projects/Dialog/Models/pre_trained_models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
    	include_top=False, input_shape=(48, 48, 3))
    model = Sequential()
    model.add(vgg_model)
    model.add(Conv2D(32, (3, 3), input_shape=(48, 48, 3), padding='same'))
    model.add(Flatten())  # Flatten the input

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
def evaluate_model_vgg16(x_train, y_train, x_test, y_test):
    # create the model
    model = create_model_with_vgg16()

    # fit the model
    history = model.fit(x_train, y_train, epochs=2, batch_size=50, validation_data=(x_test, y_test), verbose=1)

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
    model.save('MNIST_classification_model_noise_75_with VGG.h5')

    return history, history.history['val_accuracy']


# convert 28x28 grayscale to 48x48 rgb channels
def to_rgb(img):
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    img_rgb = np.asarray(np.dstack((img, img, img)), dtype=np.uint8)
    return img_rgb


def convert_images(x_train):
    rgb_list = []
    # convert X_train data to 48x48 rgb values
    for i in range(len(x_train)):
        rgb = to_rgb(x_train[i])
        rgb_list.append(rgb)
        # print(rgb.shape)

    rgb_arr = np.stack([rgb_list], axis=4)
    rgb_arr_to_3d = np.squeeze(rgb_arr, axis=4)
    print(rgb_arr_to_3d.shape)
    return rgb_arr_to_3d


def run_grid_search():
    # create model
    model = KerasClassifier(build_fn=create_model_for_grid_search, epochs=20, batch_size=32, verbose=1)

    # define the grid search parameters
    optimizer = ['SGD', 'RMSprop', 'Adam']
    neurons = [10, 20, 30]
    learn_rate = [0.001, 0.01, 0.1]
    momentum = [0.2, 0.4, 0.6]

    # create the parameter grid
    param_grid = dict(optimizer=optimizer,
                      neurons=neurons,
                      learn_rate=learn_rate,
                      momentum=momentum)

    # grid search for the best results
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
    grid_result = grid.fit(x_train_noisy, y_train)

    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


def run_transfer_learning_vgg16():
    converted_x_train = convert_images(x_train_noisy)
    converted_x_test = convert_images(x_test_noisy)

    converted_x_train = converted_x_train.reshape((converted_x_train.shape[0], 48, 48, 3))
    converted_x_test = converted_x_test.reshape((converted_x_test.shape[0], 48, 48, 3))

    # modelling
    history, accuracy_over_epoch = evaluate_model_vgg16(converted_x_train, y_train, converted_x_test, y_test)
    accuracy_over_epoch = pd.DataFrame(accuracy_over_epoch)
    accuracy_over_epoch.to_csv(
        "E:/MSc/Sem 02/Machine Learning/Assignment 02/SGD_accuracies_over_epoch_more_layers_with_noise_0_75_with_VGG.csv")

# Question no 03 [improve the model accuracy]
# ---------------------------------------------------------------------------------------------------------------------
# load data
x_train, y_train, x_test, y_test = load_dataset()

# define the noise factor
noise_factor = 0.75

# include the noise factor
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# Grid search results
run_grid_search()

# Transfer learning with VGG16
run_transfer_learning_vgg16()
