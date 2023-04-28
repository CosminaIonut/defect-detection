# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from copy import deepcopy
import seaborn as sn
import csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import sklearn
from sklearn.model_selection import train_test_split
import os
import random as rn

os.environ['PYTHONHASHSEED']= '0'
np.random.seed(1)
rn.seed(1)
#tf.set_random_seed(1)

import tensorflow as tf

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import RMSprop
from keras import regularizers
from keras.regularizers import l2
from statistics import mean
from keras.models import load_model
from sklearn.metrics import mean_squared_error




def build_nn(length=8, filters=(512, 256), regularizer=None):
    network = Sequential()
    network.add(Dense(length, input_shape=(length,), activation='relu'))

    for i, f in enumerate(filters):
        #if i % 2:
            #network.add(Dropout(0.2))
        network.add(Dense(f, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
        #network.add(Dropout(0.2))

    # network.add(Dropout(0.2))
    network.add(Dense(1, activation='linear'))

    optimizer = RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    #https://colab.research.google.com/github/fabiodimarco/tf-levenberg-marquardt/blob/main/tf-levenberg-marquardt.ipynb#scrollTo=RnvOIrjSkcEo

    # model_wrapper = lm.ModelWrapper(network)
    #
    # model_wrapper.compile(
    #     optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
    #     loss=lm.MeanSquaredError,
    #     metrics=['mae', 'mse'])

    network.compile(optimizer=optimizer, loss='mse', metrics=['mae','mse'])

    return network


def build_data(data, pos):
    df_samples_slim = pd.read_excel(data, engine='openpyxl')
    df_samples_slim.dropna()

    x_slim = df_samples_slim.iloc[:, 3:].values
    y_slim = df_samples_slim.iloc[:, np.r_[0]].values

    y = np.zeros((len(y_slim), 9))

    for i in range(len(y_slim)):
        y[i][pos] = y_slim[i][0] * 1000

    return x_slim, y


def train_model(network, data):
    df_samples_slim = pd.read_excel(data, engine='openpyxl')
    df_samples_slim.dropna()

    x = df_samples_slim.iloc[:, 3:].values
    y = df_samples_slim.iloc[:, np.r_[0]].values

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_mse',
                                                patience=3,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.00001)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=0)

    history = network.fit(x_train, y_train, epochs=1, batch_size=30, validation_data=(x_test, y_test),
                          verbose=1, callbacks=[learning_rate_reduction])

    return network, history

def read_test_data():
    df_samples_slim = pd.read_excel('test_segment_400-500.xlsx', engine='openpyxl')
    df_samples_slim.dropna()

    x_test_input = df_samples_slim.iloc[:, 2:].values
    y_test_input = df_samples_slim.iloc[:, np.r_[0]].values

    return x_test_input, y_test_input



data = ['1_0-150.xlsx', '2_100-300.xlsx', '3_250-400.xlsx', '4_350-500.xlsx', '5_450-600.xlsx', '6_550-700.xlsx', '7_650-800.xlsx', '8_750-900.xlsx', '9_850-1000.xlsx']
data_no_overlap = ['no_overlap_data/1_0-100.xlsx', 'no_overlap_data/2_100-200.xlsx','no_overlap_data/3_200-300.xlsx',
                   'no_overlap_data/4_300-400.xlsx', 'no_overlap_data/5_400-500.xlsx', 'no_overlap_data/6_500-600.xlsx',
                   'no_overlap_data/7_600-700.xlsx', 'no_overlap_data/8_700-800.xlsx', 'no_overlap_data/9_800-900.xlsx',
                   'no_overlap_data/10_900-1000.xlsx']
models = [train_model(build_nn(), data_no_overlap[i]) for i in range(10)]



def test_model(networks, test_data, y_target):
    i = 0
    y_pred=[]
    for row in test_data:

        segment = 4
        pos_rf = -1
        # filename = 'models_segments/model_' + str(segment) + '.h5'
        filename = 'models_segments_no_overlap/model_' + str(segment + 1) + '.h5'
        # load model from file
        model = load_model(filename)
        yhat = model.predict([row.tolist()])
        pos_rf = yhat[0][0]
        # for model in networks:
        #     if networks.index(model) == 4:
        #         yhat = model[0].predict([row.tolist()])
        #         pos_rf = yhat[0][0]

        print('Predicted EN: %s' % (pos_rf), y_target[i])

        print('')
        y_pred.append(pos_rf)
        i += 1

    return y_pred


x_test, y_test = read_test_data()
y_pred = test_model(models, x_test, y_test)

error = mean_squared_error(y_test, y_pred)
print("Error(MSE)", format(float(error), '.7f'))
# # Write predictions in a text file
# f = open("predictions/overlap_test_model400-500.txt", "w")
# for x in y_pred:
#     f.write(str(x)+"\n")
# f.close()