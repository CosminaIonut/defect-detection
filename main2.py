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


# import lm

from tensorflow.keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import RMSprop
from keras import regularizers




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
    df_samples_slim = pd.read_excel('date_test.xlsx', engine='openpyxl')
    df_samples_slim.dropna()

    x_test_input = df_samples_slim.iloc[:, 2:].values
    y_test_input = df_samples_slim.iloc[:, np.r_[0]].values

    return x_test_input, y_test_input



data = ['1_0-150.xlsx', '2_100-300.xlsx', '3_250-400.xlsx', '4_350-500.xlsx', '5_450-600.xlsx', '6_550-700.xlsx', '7_650-800.xlsx', '8_750-900.xlsx', '9_850-1000.xlsx']

models =  [train_model(build_nn(), data[i]) for i in range(9)]

from statistics import mean

rows = [([0.000412, 0.000412, 0.000414, 0.000419, 0.000429, 0.000446, 0.000472, 0.000507], 962, 0.0011911),
        ([0.002246, 0.000832, 0.000174, -6.5E-06, 0.000252, 0.000666, 0.001077, 0.001298], 100, 0.0033459),
        ([0.001899, 0.000254, 3.79E-05, 0.000552, 0.001173, 0.001254, 0.000798, 0.000219], 150, 0.0033459),
        ([0.000644, 0.00102, 0.000683, 0.000299, 0.00148, 3.7E-05, 0.001166, 0.000771], 400, 0.0033459),
        ([0.00024, 0.001535, 0.000256, 0.00108, 0.000632, 0.00064, 0.001072, 0.00024], 550, 0.0033459),
        ([0.000141, 0.001343, 0.000991, 0.000146, 0.001467, 0.000171, 0.000805, 0.001136], 613, 0.0033459),
        ([0.001995056, 0.000384732, 7.27776E-06, 0.000173547, 0.000734034, 0.001227756, 0.000814199, 0.000205494], 133,
         0.0033459),
        ([0.004237513, 0.002804209, 0.004348325, 0.003654791, 0.002554861, 0.003922156, 0.004598436, 0.002963461], 280,
         0.005123),
        ([0.002521808, 0.00306296, 0.002482601, 0.002346224, 0.003380631, 0.001844147, 0.003376766, 0.002300145], 410,
         0.0033459),
        ([0.001851032, 0.004503032, 0.002379226, 0.002995499, 0.003568136, 0.001829615, 0.004420003, 0.001480542], 570,
         0.0033459),
        ([0.020610, 0.007828, 0.001629, 0.000031, 0.002070, 0.005964, 0.009762, 0.011873], 98, 0.026224),
        ([0.001795, 0.000660, 0.002334, 0.000600, 0.000542, 0.002654, 0.001223, 0.000146], 310, 0.005124),
        ([0.002382, 0.017252, 0.005019, 0.009109, 0.011488, 0.002556, 0.016603, 0.000017], 569, 0.026224),
        ([0.023458, 0.005550, 0.000064, 0.002581, 0.008901, 0.014021, 0.014610, 0.010491], 126, 0.026224),
        ([0.000288, 0.005461, 0.017336, 0.017272, 0.004422, 0.000996, 0.012234, 0.016377], 759, 0.026224)
        ]


def test_model(networks, test_data):
    for row in test_data:
        o = row[1]
        s = row[2]

        i = 0
        pos_rf = []
        #sev_rf = []
        #inc_rf = []

        for model in networks:
            yhat = model[0].predict([row[0]])
            pos_rf.append(1000 * yhat[0][0])
            #sev_rf.append(yhat[0][1])
            #inc_rf.append(yhat[0][2])

            i += 1

        pos_rf = mean(pos_rf)
        #sev_rf = mean(sev_rf)
        #inc_rf = mean(inc_rf)

        print('Predicted EN: %s' % (pos_rf), o)
        print('')

test_model(models, rows)