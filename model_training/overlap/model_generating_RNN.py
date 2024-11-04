from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, SimpleRNN, LSTM, GRU, Bidirectional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.optimizers import RMSprop, SGD, Adam
from keras.regularizers import l2
from wandb.keras import WandbCallback
from keras.layers import Dropout
from keras.constraints import MaxNorm
import tensorflow as tf
from keras import backend as K

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

def build_RNN_sweep(optimizer, learning_rate, hidden_layer_size, dense_units,activation, length=8, time_steps=8):
    network = Sequential()
    for hidden_units in hidden_layer_size:
        network.add(SimpleRNN(hidden_units, input_shape=(time_steps,1), activation=activation[0]))

    network.add(Dense(units=dense_units, activation=activation[1]))

    optimizer = build_optimizer(optimizer, learning_rate)
    network.compile(optimizer=optimizer, loss='mse', metrics=['mae','mse',tf.keras.metrics.RootMeanSquaredError(
    name="root_mean_squared_error", dtype=None)])

    return network

def build_RNN_sweep_vertical(optimizer, learning_rate, hidden_layer_size, dense_units,activation, length=8, time_steps=8):
    network = Sequential()
    for hidden_units in hidden_layer_size:
        network.add(SimpleRNN(hidden_units, input_shape=(time_steps,1), activation=activation[0], return_sequences=True))

    network.add(SimpleRNN(hidden_layer_size[0], activation=activation[0], return_sequences=True))
    network.add(SimpleRNN(hidden_layer_size[0], activation=activation[0], return_sequences=False))
    network.add(Dense(units=dense_units, activation=activation[1]))
    network.add(Dense(units=1, activation=activation[2]))

    optimizer = build_optimizer(optimizer, learning_rate)
    network.compile(optimizer=optimizer, loss='mse', metrics=['mae','mse',tf.keras.metrics.RootMeanSquaredError(
    name="root_mean_squared_error", dtype=None)])

    return network


def build_RNN_sweep_LSTM(optimizer, learning_rate, hidden_layer_size, dense_units, activation, length=8, time_steps=8):
    network = Sequential()
    for hidden_units in hidden_layer_size:
        network.add(
            LSTM(hidden_units, input_shape=(time_steps, 1), activation=activation[0]))

    # network.add(SimpleRNN(hidden_layer_size[0], activation=activation[0], return_sequences=True))
    # network.add(SimpleRNN(hidden_layer_size[0], activation=activation[0], return_sequences=False))
    network.add(Dense(units=dense_units, activation=activation[1]))
    # network.add(Dense(units=1, activation=activation[2]))

    optimizer = build_optimizer(optimizer, learning_rate)
    network.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse', tf.keras.metrics.RootMeanSquaredError(
        name="root_mean_squared_error", dtype=None)])

    return network

def build_RNN_sweep_LSTM_bidirectional(optimizer, learning_rate, hidden_layer_size, dense_units, activation, length=8, time_steps=8):
    network = Sequential()
    for hidden_units in hidden_layer_size:
        network.add(
            Bidirectional(LSTM(hidden_units, return_sequences=False, input_shape=(time_steps, 1), activation=activation[0])))

    # network.add(SimpleRNN(hidden_layer_size[0], activation=activation[0], return_sequences=True))
    # network.add(SimpleRNN(hidden_layer_size[0], activation=activation[0], return_sequences=False))
    network.add(Dense(units=dense_units, activation=activation[1]))
    # network.add(Dense(units=1, activation=activation[2]))

    optimizer = build_optimizer(optimizer, learning_rate)
    network.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])

    return network

def build_RNN_sweep_GRU(optimizer, learning_rate, hidden_layer_size, dense_units, activation, length=8, time_steps=8):
    network = Sequential()
    for hidden_units in hidden_layer_size:
        network.add(
            GRU(hidden_units, input_shape=(time_steps, 1), activation=activation[0]))

    # network.add(SimpleRNN(hidden_layer_size[0], activation=activation[0], return_sequences=True))
    # network.add(SimpleRNN(hidden_layer_size[0], activation=activation[0], return_sequences=False))
    network.add(Dense(units=dense_units, activation=activation[1]))
    # network.add(Dense(units=1, activation=activation[2]))

    optimizer = build_optimizer(optimizer, learning_rate)
    network.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse', tf.keras.metrics.RootMeanSquaredError(
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
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    history = network.fit(x_train, y_train, epochs=epochs, batch_size = batch_size, validation_data=(x_test, y_test),
                          verbose=1, callbacks=[learning_rate_reduction, WandbCallback()])
    return network, history

def split_data(data, test_size):
    df_samples_slim = pd.read_excel(data, engine='openpyxl')
    df_samples_slim.dropna()

    x = df_samples_slim.iloc[:, 3:].values
    y = df_samples_slim.iloc[:, np.r_[0]].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=0)
    return x_train, x_test, y_train, y_test

# def get_train_test(url, split_percent=0.8):
#     df = read_csv(url, usecols=[1], engine='python')
#     data = np.array(df.values.astype('float32'))
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     data = scaler.fit_transform(data).flatten()
#     n = len(data)
#     # Point for splitting data into train and test
#     split = int(n*split_percent)
#     train_data = data[range(split)]
#     test_data = data[split:]
#     return train_data, test_data, data
