from keras.models import load_model
from numpy import dstack
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from numpy import savetxt


def load_one_model(path):
    filename = '../../trained_models/' + path + '/model_' + str( 1) + '.h5'
    # load model from file
    model = load_model(filename)
    # add to list of members
    print('>loaded %s' % filename)
    return model

def split_data(data, test_size):
    df_samples_slim = pd.read_excel(data, engine='openpyxl')
    df_samples_slim.dropna()

    x = df_samples_slim.iloc[:, 3:].values
    y = df_samples_slim.iloc[:, np.r_[0,1,2]].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=0)
    return x_train, x_test, y_train, y_test


def read_test_data(path):
    df_samples_slim = pd.read_excel(path, engine='openpyxl')
    df_samples_slim.dropna()

    x_test_input = df_samples_slim.iloc[:, 3:].values
    y_test_input = df_samples_slim.iloc[:, np.r_[0,1,2]].values

    return x_test_input, y_test_input
