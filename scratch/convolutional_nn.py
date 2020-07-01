"""Convolutional Neural Network

Program for seting up and testing a CNN based on TensorFlow 2.0.
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interpolate

# Using Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import load_model


def get_mnist_data():
    """Retrieves the mnist handwriting data.

    Returns:
        list(np.ndarrays) -- training, valid and test samples and labels.
    """
    test_data_path = "../datafiles/HandwritingClassification/mnist.pkl"
    with open(test_data_path, "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = "latin1"
        data_train, data_valid, data_test = u.load()

    print("DATA TRAIN: ", data_train[0].shape, data_train[1].shape)
    print("DATA VALID: ", data_valid[0].shape, data_valid[1].shape)
    print("DATA TEST: ", data_test[0].shape, data_test[1].shape)

    def convert_output(label_, output_size):
        """Converts label to output vector."""
        y_ = np.zeros(output_size, dtype=float)
        y_[label_] = 1.0
        return y_

    Nx = 28
    Ny = int(data_train[0].shape[1] / Nx)

    # Converts data to ((N, p-1)) shape.
    data_train_samples = np.asarray(
        [d_.reshape((Nx, Ny, 1)) for d_ in data_train[0]])
    data_valid_samples = np.asarray(
        [d_.reshape((Nx, Ny, 1)) for d_ in data_valid[0]])
    data_test_samples = np.asarray(
        [d_.reshape((Nx, Ny, 1)) for d_ in data_test[0]])

    # Converts labels from single floats to arrays with 1.0 at correct output.
    # Aka, to one-hot vector format.
    data_train_labels = np.asarray(
        [convert_output(l, 10) for l in data_train[1]])
    data_valid_labels = np.asarray(
        [convert_output(l, 10) for l in data_valid[1]])
    data_test_labels = np.asarray(
        [convert_output(l, 10) for l in data_test[1]])

    return [data_train_samples, data_valid_samples, data_test_samples,
            data_train_labels, data_valid_samples, data_test_labels]


# Load data
X_train, X_valid, X_test, Y_train, Y_valid, Y_test = get_mnist_data()

# Set up network


def create_convolutional_neural_network_keras(input_shape, receptive_field,
                                              n_filters,
                                              n_categories, eta, lmbd):
    """Sets up a convolutional network using Keras.
    
    Arguments:
        input_shape {tuple(int, int)} -- input shape.
        receptive_field {int} -- size of the convolution kernel.
        n_filters {int} -- number of convolution kernels(filters).
        n_categories {[type]} -- [description]
        eta {[type]} -- [description]
        lmbd {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """

    print("Input shape:", input_shape)

    model = Sequential()

    # Works
    # model.add(Conv2D(n_filters, (receptive_field, receptive_field),
    #                  input_shape=input_shape, padding='same',
    #                  activation='relu', kernel_regularizer=l2(lmbd)))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Flatten())
    # model.add(Dense(50, activation='tanh',
    #                 kernel_regularizer=l2(lmbd)))
    # model.add(Dropout(0.05))
    # model.add(Dense(n_categories, activation='softmax',
    #                 kernel_regularizer=l2(lmbd)))

    # # Works
    # model.add(Conv2D(n_filters, (receptive_field, receptive_field),
    #                  input_shape=input_shape, padding='same',
    #                  activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.1))
    # model.add(Conv2D(2*n_filters, (receptive_field, receptive_field),
    #                  padding='same', activation='relu'))
    # model.add(Conv2D(4*n_filters, (receptive_field, receptive_field),
    #                  padding='same', activation='relu'))
    # # model.add(Conv2D(8*n_filters, (1, 1)))  # Needed since input is variable
    # # model.add(GlobalMaxPooling2D())         # Needed since input is variable
    # model.add(Flatten())
    # model.add(Dense(n_categories, activation='softmax'))
    # # model.add(Dense(n_categories, activation='softmax'))

    # Works with variable input size
    model.add(Input(input_shape))
    model.add(Conv2D(n_filters, (receptive_field, receptive_field),
                     padding='same', activation='relu',
                     kernel_initializer='glorot_uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))
    model.add(Conv2D(2*n_filters, (receptive_field, receptive_field),
                     padding='same', activation='relu',
                     kernel_initializer='glorot_uniform'))
    model.add(Conv2D(4*n_filters, (receptive_field, receptive_field),
                     padding='same', activation='relu',
                     kernel_initializer='glorot_uniform'))
    model.add(Conv2D(16*n_filters, (1, 1), 
                     kernel_initializer='glorot_uniform'))  # Needed since input is variable
    model.add(GlobalMaxPooling2D())         # Needed since input is variable
    model.add(Dense(n_categories, activation='softmax',
                    kernel_initializer='glorot_uniform'))

    # sgd = SGD(learning_rate=eta)
    adam = Adam(learning_rate=eta)
    model.compile(loss='categorical_crossentropy',
                  # optimizer=sgd,
                  optimizer=adam,
                  metrics=['accuracy'])

    return model


def cnn_tf_setup():

    # PARAMS

    # Sets up the layers
    with tf.variable_scope("cnn"):
        num_hidden_layers = np.size(num_hidden_neurons)

        previous_layer = points

        for l in range(num_hidden_layers):
            current_layer = tf.layers.dense(
                previous_layer, num_hidden_neurons[l],
                activation=hidden_activation_function)

            # current_layer = tf.layers.dense(
            #     previous_layer, num_hidden_neurons[l],
            #     activation=tf.nn.sigmoid)

            if dropout_rate != 0.0:
                current_layer = tf.nn.dropout(current_layer, dropout_rate)

            previous_layer = current_layer

        dnn_output = tf.layers.dense(previous_layer, 1)


# Run network
epochs = 2
batch_size = 50
input_shape = X_train.shape[1:]
input_shape = (None, None, 1)
receptive_field = 3
n_filters = 10
n_categories = 10

eta_vals = np.logspace(-5, 1, 7)  # Learning rate
lmbd_vals = np.logspace(-5, 1, 7)  # Lambda regularizer

# Since there is no optimization of hyper parameters
eta = eta_vals[2]
lmbd = lmbd_vals[2]

# CNN = load_model("cnn_model2")
# CNN = load_model("cnn_model2_none")
CNN = create_convolutional_neural_network_keras(input_shape, receptive_field,
                                                n_filters, n_categories, 
                                                eta, lmbd)

CNN.summary()

CNN.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=1)
scores = CNN.evaluate(X_test, Y_test, verbose=0)

# CNN.save("cnn_model2")
# CNN.save("cnn_model2_none")


# Output
print("Default input shape:", X_test[0:1, :, :, :].shape)
predicted_value = CNN.predict(X_test[0:1, :, :, :])
print("Default input shape:", np.argmax(predicted_value))

if input_shape == (None, None, 1):
    X_test_upscaled = X_test[0, :, :, 0].repeat(2, axis=0).repeat(2, axis=1)

    # # Verifies the up-scaled output
    # fig, ax = plt.subplots(1, 2)
    # ax[0].imshow(X_test[0, :, :, 0])
    # ax[1].imshow(X_test_upscaled)
    # plt.show()

    print("Scaled up input shape:",
          X_test_upscaled[np.newaxis, :, :, np.newaxis].shape)
    predicted_interp_value = CNN.predict(
        X_test_upscaled[np.newaxis, :, :, np.newaxis])
    print("Scaled up predicted value: %d" % np.argmax(predicted_interp_value))

print("Learning rate = ", eta)
print("Lambda = ", lmbd)
print("Test accuracy:", scores)

# Saving the CNN
