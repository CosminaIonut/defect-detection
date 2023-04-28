from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import RMSprop, SGD, Adam
from keras.regularizers import l2
from wandb.keras import WandbCallback

def build_nn(length=8, filters=(512, 256), regularizer=None):
    network = Sequential()
    network.add(Dense(length, input_shape=(length,), activation='relu'))

    for i, f in enumerate(filters):
        network.add(Dense(f, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))

    network.add(Dense(1, activation='linear'))

    optimizer = RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    network.compile(optimizer=optimizer, loss='mse', metrics=['mae','mse'])

    return network

def build_optimizer(optimizer, learning_rate):
    if optimizer == "sgd":
        opti = SGD(learning_rate=learning_rate, momentum=0.8)
    elif optimizer == "adam":
        opti = Adam(learning_rate=learning_rate)
    elif optimizer == "rmsprop":
        opti = RMSprop(learning_rate=learning_rate, rho=0.9, epsilon=1e-08, decay=0.0)
    return opti

def build_nn_sweep(optimizer, learning_rate, hidden_layer_size, length=8):
    network = Sequential()
    network.add(Dense(length, input_shape=(length,), activation='relu'))

    for neurons_numbers in hidden_layer_size:
        network.add(Dense(neurons_numbers, activation='relu', kernel_initializer='he_uniform',
                          kernel_regularizer=l2(0.01),bias_regularizer=l2(0.01)))
    network.add(Dense(1, activation='sigmoid'))
    optimizer = build_optimizer(optimizer, learning_rate)
    network.compile(optimizer=optimizer, loss='mse', metrics=['mae','mse'])
    return network

def train_model(network, x_train, y_train,x_test, y_test, epochs):
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_mse',
                                                patience=3,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.00001)
    history = network.fit(x_train, y_train, epochs=epochs, batch_size=20, validation_data=(x_test, y_test),
                          verbose=1, callbacks=[learning_rate_reduction,WandbCallback()])

    return network, history

def train_model_sweep(network, x_train, y_train,x_test, y_test, epochs, batch_size,patience,monitor):
    learning_rate_reduction = ReduceLROnPlateau(monitor= monitor,
                                                patience=patience,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.00001)

    # early_stop = EarlyStopping(monitor='val_mse', patience=3, verbose=1)
    # csv_logger = CSVLogger('train_log.csv', separator=",", append=False)

    history = network.fit(x_train, y_train, epochs=epochs, batch_size = batch_size, validation_data=(x_test, y_test),
                          verbose=1, callbacks=[learning_rate_reduction, WandbCallback()])
    return network, history


def split_data(data, test_size):
    df_samples_slim = pd.read_excel(data, engine='openpyxl')
    df_samples_slim.dropna()

    x = df_samples_slim.iloc[:, 3:].values
    y = df_samples_slim.iloc[:, np.r_[0, 1]].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size, random_state=0)
    return x_train, x_test, y_train, y_test




