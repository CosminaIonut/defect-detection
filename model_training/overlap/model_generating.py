from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
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
import sys
import os
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp

from model_training.overlap.levenberg_marquardt import ModelWrapper, MeanSquaredError

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

FEATURE_NAMES = [
    "Mod 1",
    "Mod 2",
    "Mod 3",
    "Mod 4",
    "Mod 5",
    "Mod 6",
    "Mod 7",
    "Mod 8",
]


def R2(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )
def distribution(kernel_size, bias_size):
    n = kernel_size + bias_size
    return tfp.layers.DistributionLambda(
        lambda t: tfp.distributions.MultivariateNormalDiag(
            loc=tf.zeros(n), scale_diag=tf.ones(n)
        )
    )

def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    print("N",n)
    zeros=[]
    ones=[]
    for i in range(0,n):
        zeros.append(0)
        ones.append(1)
    tfd = tfp.distributions
    prior_model = keras.Sequential(
        [
            tfp.layers.DistributionLambda(
                make_distribution_fn=lambda t: tfd.Normal(
                    loc=zeros, scale=ones))
        ]
            )

    return prior_model


# Define variational posterior weight distribution as multivariate Gaussian.
# Note that the learnable parameters for this distribution are the means,
# variances, and covariances.
def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    posterior_model = keras.Sequential(
        [
            tfp.layers.VariableLayer(
                tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
            ),
            tfp.layers.MultivariateNormalTriL(n),
        ]
    )
    return posterior_model

def build_baseline_nn(length=8):
    # create model
    model = Sequential()
    model.add(Dense(length, input_shape=(length,), activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    optimizer = SGD(learning_rate=0.01, momentum=0.8)
    # optimizer = RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae','mse'])
    return model

def build_dropout_visible_nn(length=8,filters=(512, 256)):
    # create model
    model = Sequential()
    model.add(Dropout(0.2, input_shape=(8,)))  #dropout
    model.add(Dense(length, input_shape=(length,), activation='relu'))
    for i, f in enumerate(filters):
        model.add(Dense(f, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), kernel_constraint=MaxNorm(3)))
    model.add(Dense(1, activation='linear'))

    # Compile model
    optimizer = SGD(learning_rate=0.1, momentum=0.9)
    # optimizer = RMSprop(learning_rate=0.01, rho=0.9, epsilon=1e-08, decay=0.0)  # learning rate was lifted by one order of magnitude
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae','mse'])
    return model


def build_dropout_hidden_nn(length=8,filters=(512, 256)):
    # create model
    model = Sequential()
    model.add(Dense(length, input_shape=(length,), activation='relu'))
    model.add(Dropout(0.2))
    for i, f in enumerate(filters):
        # print(i)
        if i % 2:
            model.add(Dropout(0.2))
        model.add(Dense(f, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), kernel_constraint=MaxNorm(3)))

    model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))

    # Compile model
    optimizer = SGD(learning_rate=0.1, momentum=0.9)
    # optimizer = RMSprop(learning_rate=0.01, rho=0.9, epsilon=1e-08, decay=0.0)  # learning rate was lifted by one order of magnitude
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae','mse'])
    return model

def build_nn(length=8, filters=(512, 256), regularizer=None):
    network = Sequential()
    network.add(Dense(length, input_shape=(length,), activation='relu'))

    for i, f in enumerate(filters):
        network.add(Dense(f, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))

    network.add(Dense(1, activation='linear'))

    optimizer = SGD(learning_rate=0.01, momentum=0.8)
    # optimizer = RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    network.compile(optimizer=optimizer, loss='mse', metrics=['mae','mse'])

    return network

def build_nn_sigmoid(length=8, filters=(512, 256), regularizer=None):
    network = Sequential()
    network.add(Dense(length, input_shape=(length,), activation='relu'))

    for i, f in enumerate(filters):
        network.add(Dense(f, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))

    network.add(Dense(1, activation='sigmoid'))

    optimizer = SGD(learning_rate=0.01, momentum=0.8)
    # optimizer = RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    network.compile(optimizer=optimizer, loss='mse', metrics=['mae','mse'])

    return network

def build_nn_sigmoid_new_network(length=8, filters=(512, 256), regularizer=None):
    network = Sequential()
    network.add(Dense(12, input_shape=(length,), activation='relu'))
    network.add(Dense(8, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    network.add(Dense(1, activation='sigmoid'))

    # optimizer = SGD(learning_rate=0.01, momentum=0.8)
    # optimizer = RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    network.compile(optimizer='adam', loss='mse', metrics=['mae','mse'])

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
    network.add(Dense(hidden_layer_size[0], input_shape=(length,), activation='relu'))
    hidden_nodes = hidden_layer_size[1:]
    for neurons_numbers in hidden_nodes:
        network.add(Dense(neurons_numbers, activation='relu', kernel_initializer='he_uniform',
                          kernel_regularizer=l2(0.001),bias_regularizer=l2(0.001)))
    network.add(Dense(1, activation='sigmoid'))
    optimizer = build_optimizer(optimizer, learning_rate)
    network.compile(optimizer=optimizer, loss='mse', metrics=['mae','mse',tf.keras.metrics.RootMeanSquaredError(
    name="root_mean_squared_error", dtype=None)])
    return network

def create_model_inputs():
    inputs = {}
    for feature_name in FEATURE_NAMES:
        inputs[feature_name] = layers.Input(
            name=feature_name, shape=(1,), dtype=tf.float32
        )
    return inputs
def build_nn_sweep_BNN(optimizer, learning_rate, hidden_layer_size,sample_size, length=8):
    inputs2 = create_model_inputs()
    # print(inputs)
    inputs = layers.Input(shape=(8,))
    features = layers.Dense(8, activation="relu")(inputs)


    # Create hidden layers with weight uncertainty using the DenseVariational layer.
    for units in hidden_layer_size:
        features = tfp.layers.DenseVariational(
            units=units,
            make_prior_fn=prior,
            make_posterior_fn=posterior,
            kl_weight=1 / sample_size,
            activation="sigmoid",
        )(features)

    # The output is deterministic: a single point estimate.
    outputs = layers.Dense(units=1)(features)
    model = keras.Model(inputs=inputs, outputs=outputs)
    optimizer = build_optimizer(optimizer, learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae','mse'])
    return model

def build_nn_sweep_levenberg(optimizer, learning_rate, hidden_layer_size, length=8):
    network = Sequential()
    network.add(Dense(length, input_shape=(length,), activation='relu'))

    for neurons_numbers in hidden_layer_size:
        network.add(Dense(neurons_numbers, activation='relu', kernel_initializer='he_uniform',
                          kernel_regularizer=l2(0.001),bias_regularizer=l2(0.001)))
    network.add(Dense(1, activation='sigmoid'))

    model_wrapper = ModelWrapper(
        tf.keras.models.clone_model(network))

    model_wrapper.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=1.0),
        loss=MeanSquaredError())
    return model_wrapper

def build_nn_baseline():
    network = Sequential()
    network.add(
        Dense(100, input_shape=(8,), activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(0.01),
              bias_regularizer=l2(0.01)))
    network.add(Dense(20, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(0.01),
                    bias_regularizer=l2(0.01)))
    network.add(Dense(1, activation='sigmoid'))
    optimizer = RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    network.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])

    return network

def build_nn_sigmoid_new_network_more_neurons(length=8, filters=(512, 256), regularizer=None):
    network = Sequential()

    # 20 input
    # network.add(Dense(20, input_shape=(length,), activation='relu'))
    # network.add(Dense(8, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(0.01),
    #                   bias_regularizer=l2(0.01)))
    # 50 input
    # network.add(Dense(50, input_shape=(length,), activation='relu'))
    # network.add(Dense(26, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(0.01),
    #                   bias_regularizer=l2(0.01)))
    # # 8 input 5 hidden
    # network.add(Dense(8, input_shape=(length,), activation='relu'))
    # network.add(Dense(5, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(0.01),
    #                   bias_regularizer=l2(0.01)))

    # # 8 input 4 hidden
    # network.add(Dense(8, input_shape=(length,), activation='relu'))
    # network.add(Dense(4, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(0.01),
    #                   bias_regularizer=l2(0.01)))

    # # 8 input 20 hidden
    network.add(Dense(length, input_shape=(length,), activation='relu'))
    network.add(Dense(20, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(0.01),
                      bias_regularizer=l2(0.01)))
    #
    # # 8 input 50 hidden
    # network.add(Dense(length, input_shape=(length,), activation='relu'))
    # network.add(Dense(50, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(0.01),
    #                   bias_regularizer=l2(0.01)))

    # # 8 input 100 hidden
    # network.add(Dense(length, input_shape=(length,), activation='relu'))
    # network.add(Dense(100, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(0.01),
    #                   bias_regularizer=l2(0.01)))

    network.add(Dense(1, activation='sigmoid'))

    # optimizer = SGD(learning_rate=0.01, momentum=0.8)
    # optimizer = RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    network.compile(optimizer='adam', loss='mse', metrics=['mae','mse'])

    return network

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




