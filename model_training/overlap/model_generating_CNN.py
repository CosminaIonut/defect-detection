import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.optimizers import RMSprop, SGD, Adam
from keras.regularizers import l2
from wandb.keras import WandbCallback
from keras.layers import Dropout, Conv2D, Flatten, Dense, MaxPooling2D, Conv1D, MaxPooling1D
from keras.constraints import MaxNorm
import tensorflow as tf
from keras import backend as K, Sequential


def R2(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


def build_optimizer(optimizer, learning_rate):
    if optimizer == "sgd":
        opti = SGD(learning_rate=learning_rate, momentum=0.8)
    elif optimizer == "adam":
        opti = Adam(learning_rate=learning_rate)
    elif optimizer == "rmsprop":
        opti = RMSprop(learning_rate=learning_rate, rho=0.9, epsilon=1e-08, decay=0.0)
    return opti


def build_cnn_sweep(optimizer, learning_rate, cnn_layer_size, input_nodes_cnn, dense_units, length=8):
    network = Sequential()
    network.add(Conv2D(input_nodes_cnn, kernel_size=(3,1), activation='relu', input_shape =(length,1,1)))
    # network.add(Conv2D(input_nodes_cnn, kernel_size=(2,1), activation='relu', input_shape =(4,2,1), data_format='channels_first'))

    for neurons_numbers in cnn_layer_size:
        network.add(Conv2D(neurons_numbers, kernel_size=(3,1), activation='relu'))
        # network.add(Conv2D(neurons_numbers, kernel_size=(2,1), activation='relu'))

    network.add(Flatten())
    network.add(Dense(dense_units, activation='softmax'))
    optimizer = build_optimizer(optimizer, learning_rate)
    network.compile(optimizer=optimizer, loss='mse', metrics=['mae','mse',tf.keras.metrics.RootMeanSquaredError(
    name="root_mean_squared_error", dtype=None)])
    return network

def build_cnn_sweep_maxpooling(optimizer, learning_rate, cnn_layer_size, input_nodes_cnn, dense_units, length=8):
    network = Sequential()
    # network.add(Conv2D(input_nodes_cnn, kernel_size=(2,1), activation='relu', input_shape =(4,2,1), data_format='channels_first'))
    network.add(Conv2D(input_nodes_cnn, kernel_size=(3, 1), activation='relu', input_shape=(length, 1, 1)))

    for neurons_numbers in cnn_layer_size:
        # network.add(MaxPooling2D((2, 1)))
        network.add(MaxPooling2D((2, 1)))
        # network.add(Conv2D(neurons_numbers, kernel_size=(2,1), activation='relu'))
        network.add(Conv2D(neurons_numbers, kernel_size=(3, 1), activation='relu'))

    network.add(Flatten())
    network.add(Dense(cnn_layer_size[0], activation='relu'))
    network.add(Dense(dense_units))

    optimizer = build_optimizer(optimizer, learning_rate)
    network.compile(optimizer=optimizer, loss='mse', metrics=['mae','mse',tf.keras.metrics.RootMeanSquaredError(
    name="root_mean_squared_error", dtype=None)])
    return network

def build_1Dcnn_sweep_maxpooling(optimizer, learning_rate, cnn_layer_size, input_nodes_cnn, dense_units, n_timesteps,length=8):
    network = Sequential()
    network.add(Conv1D(filters=input_nodes_cnn, kernel_size=4, activation='relu', input_shape=(n_timesteps, 1)))
    network.add(Conv1D(filters=input_nodes_cnn, kernel_size=4, activation='relu'))
    network.add(Dropout(0.5))
    for neurons_numbers in cnn_layer_size:
        network.add(MaxPooling1D(pool_size=2))
        network.add(Conv1D(filters=neurons_numbers, kernel_size=1, activation='relu'))

    network.add(Flatten())
    network.add(Dense(cnn_layer_size[0], activation='relu'))
    network.add(Dense(dense_units, activation='softmax'))

    optimizer = build_optimizer(optimizer, learning_rate)
    network.compile(optimizer=optimizer, loss='mse', metrics=['mae','mse',tf.keras.metrics.RootMeanSquaredError(
    name="root_mean_squared_error", dtype=None)])
    return network

def train_model_sweep(network, x_train, y_train,x_test, y_test, epochs, batch_size,patience,monitor):
    learning_rate_reduction = ReduceLROnPlateau(monitor= monitor,
                                                patience=patience,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.00001)

    # early_stop = EarlyStopping(monitor='val_mse', patience=3, verbose=1)
    csv_logger = CSVLogger('train_log.csv', separator=",", append=False)

    history = network.fit(x_train, y_train, epochs=epochs, batch_size = batch_size, validation_data=(x_test, y_test),
                          verbose=1, callbacks=[learning_rate_reduction, csv_logger, WandbCallback()])
    return network, history

def train_model(network, x_train, y_train,x_test, y_test, epochs):
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_mse',
                                                patience=3,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.00001)


    history = network.fit(x_train, y_train, epochs=epochs, batch_size = 30, validation_data=(x_test, y_test),
                          verbose=1, callbacks=[learning_rate_reduction, WandbCallback()])
    return network, history


def split_data(data, test_size):
    df_samples_slim = pd.read_excel(data, engine='openpyxl')
    df_samples_slim.dropna()

    x = df_samples_slim.iloc[:, 3:].values
    y = df_samples_slim.iloc[:, np.r_[0]].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=0)
    return x_train, x_test, y_train, y_test
