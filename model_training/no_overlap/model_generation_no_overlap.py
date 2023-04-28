from keras.constraints import MaxNorm
from keras.models import Sequential
from keras.layers import Dense
from wandb.keras import WandbCallback
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import RMSprop, SGD
from keras.regularizers import l2
from keras.layers import Dropout


def build_baseline_nn(length=8):
    # create model
    model = Sequential()
    model.add(Dense(length, input_shape=(length,), activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    # optimizer = SGD(learning_rate=0.01, momentum=0.8)
    optimizer = RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
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
    # optimizer = SGD(learning_rate=0.01, momentum=0.8)
    optimizer = RMSprop(learning_rate=0.01, rho=0.9, epsilon=1e-08, decay=0.0)  # learning rate was lifted by one order of magnitude
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
    # optimizer = SGD(learning_rate=0.01, momentum=0.8)
    optimizer = RMSprop(learning_rate=0.01, rho=0.9, epsilon=1e-08, decay=0.0)  # learning rate was lifted by one order of magnitude
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae','mse'])
    return model

def build_nn(length=8, filters=(512, 256), regularizer=None):
    network = Sequential()
    network.add(Dense(length, input_shape=(length,), activation='relu'))

    for i, f in enumerate(filters):
        # print(i)
        # if i % 2:
        #     network.add(Dropout(0.2))
        network.add(Dense(f, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))

    # network.add(Dropout(0.2))
    network.add(Dense(1, activation='linear'))


    optimizer = RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    network.compile(optimizer=optimizer, loss='mse', metrics=['mae','mse'])
    return network


def train_model(network, x_train, y_train, x_test, y_test, epochs):
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_mse',
                                                patience=3,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.00001)
    history = network.fit(x_train, y_train, epochs=epochs, batch_size=30, validation_data=(x_test, y_test),
                          verbose=1, callbacks=[learning_rate_reduction, WandbCallback()])
    return network, history

def train_model_k_fold(network, x_train, y_train, x_test, y_test, epochs):
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_mse',
                                                patience=3,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.00001)
    history = network.fit(x_train, y_train, epochs=epochs, batch_size=30, validation_data=(x_test, y_test),
                          verbose=1, callbacks=[learning_rate_reduction, WandbCallback()])

    kfold = StratifiedKFold(n_splits=10, shuffle=True)
    results = cross_val_score(network, x_train, y_train, cv=kfold)

    print("Baseline:",results)
    return network, history

def split_data(data, test_size):
    df_samples_slim = pd.read_excel(data, engine='openpyxl')
    df_samples_slim.dropna()

    x = df_samples_slim.iloc[:, 3:].values
    y = df_samples_slim.iloc[:, np.r_[0]].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size, random_state=0)
    return x_train, x_test, y_train, y_test


# # load the data
# data_no_overlap = ['../../data/no_overlap_data/1_0-100.xlsx', '../../data/no_overlap_data/2_100-200.xlsx',
#                    '../../data/no_overlap_data/3_200-300.xlsx','../../data/no_overlap_data/4_300-400.xlsx',
#                    '../../data/no_overlap_data/5_400-500.xlsx', '../../data/no_overlap_data/6_500-600.xlsx',
#                    '../../data/no_overlap_data/7_600-700.xlsx', '../../data/no_overlap_data/8_700-800.xlsx',
#                    '../../data/no_overlap_data/9_800-900.xlsx',
#                    '../../data/no_overlap_data/10_900-1000.xlsx']
#
# # create the 10 models
# # fit and save models
# n_members = 10
# epochs = 1
# for i in range(n_members):
#     # fit model
#     x_train, x_test, y_train, y_test = split_data(data_no_overlap[i], 0.3)
#     network, history = train_model(build_nn(), x_train, y_train,x_test, y_test, epochs)
#     # save model
#     filename = '../../trained_models/models_segments_test/model_' + str(i + 1) + '.h5'
#     network.save(filename)
#     print('>Saved %s' % filename)



