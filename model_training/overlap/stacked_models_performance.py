from keras.models import load_model
from numpy import dstack
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from numpy import savetxt


def load_all_models(n_models, path):
    all_models = list()
    for i in range(n_models):
        # define filename for this ensemble
        # filename = '../../trained_models/models_segments_overlap-baseline-SGD-30.0_50epochs/model_' + str(i + 1) + '.h5'
        # filename = '../../trained_models/models_segments_overlap-baseline-RMSprop-30.0_50epochs/model_' + str(i + 1) + '.h5'
        # filename = '../../trained_models/models_segments_overlap-dropout-visiblelayer-30.0_50epochs/model_' + str(i + 1) + '.h5'
        # filename = '../../trained_models/models_segments_overlap-dropout-hiddenlayer-30.0_50epochs/model_' + str(i + 1) + '.h5'
        # filename = '../../trained_models/models_segments_overlap30.0_50epochs/model_' + str(i + 1) + '.h5'
        # filename = '../../trained_models/models_segments_overlap30.0_1epochs/model_' + str(i + 1) + '.h5'

        # filename = '../../trained_models/models_segments_overlap-dropout-visiblelayer-SGD30.0_50epochs/model_' + str(i + 1) + '.h5'
        # filename = '../../trained_models/models_segments_overlap-dropout-hiddenlayer-SGD-30.0_50epochs/model_' + str(i + 1) + '.h5'
        # filename = '../../trained_models/models_segments_overlap-normal-SGD-30.0_50epochs/model_' + str(i + 1) + '.h5'
        # filename = '../../trained_models/models_segments_overlap-normal-SGD-30.0_1epochs/model_' + str(i + 1) + '.h5'

        # filename = '../../trained_models/models_segments_overlap-normal-sigmoid-SGD-30.0_50epochs/model_' + str(i + 1) + '.h5'
        # filename = '../../trained_models/models_segments_overlap-new-network-sigmoid-adam-30.0_50epochs/model_' + str(i + 1) + '.h5'
        # --------------------------- Neuron numbers ------------------------------------

        # filename = '../../trained_models/models_segments_overlap-new-network-sigmoid-adam-20IN-30.0_50epochs/model_' + str(i + 1) + '.h5'
        # filename = '../../trained_models/models_segments_overlap-new-network-sigmoid-adam-50IN-30.0_50epochs/model_' + str(i + 1) + '.h5'
        # filename = '../../trained_models/models_segments_overlap-new-network-sigmoid-adam-8IN-5HN-30.0_50epochs/model_' + str(i + 1) + '.h5'
        # filename = '../../trained_models/models_segments_overlap-new-network-sigmoid-adam-8IN-4HN-30.0_50epochs/model_' + str(i + 1) + '.h5'
        # filename = '../../trained_models/models_segments_overlap-new-network-sigmoid-adam-8IN-20HN-30.0_50epochs/model_' + str(i + 1) + '.h5'
        # filename = '../../trained_models/models_segments_overlap-new-network-sigmoid-adam-8IN-50HN-30.0_50epochs/model_' + str(i + 1) + '.h5'
        # filename = '../../trained_models/models_segments_overlap-new-network-sigmoid-adam-8IN-100HN-30.0_50epochs/model_' + str(i + 1) + '.h5'
        filename = '../../trained_models/'+ path +'/model_' + str(i + 1) + '.h5'


        # load model from file
        model = load_model(filename)
        # add to list of members
        all_models.append(model)
        print('>loaded %s' % filename)
    return all_models


# create stacked model input dataset as outputs from the ensemble
def stacked_dataset(members, inputX):
    stackX = None
    for model in members:
        # make prediction
        yhat = model.predict(inputX, verbose=0)
        # stack predictions into [rows, members, probabilities]
        if stackX is None:
            stackX = yhat
        else:
            stackX = dstack((stackX, yhat))
    # flatten predictions to [rows, members x probabilities]
    stackX = stackX.reshape((stackX.shape[0], stackX.shape[1] * stackX.shape[2]))
    return stackX


# fit a model based on the outputs from the ensemble members
def fit_stacked_model(members, inputX, inputy):
    # create dataset using ensemble
    stackedX = stacked_dataset(members, inputX)
    # fit standalone model
    model = KNeighborsRegressor()
    # model = DecisionTreeRegressor()
    # model = SVR()
    # model = LinearRegression()
    model.fit(stackedX, inputy)
    return model

# make a prediction with the stacked model
def stacked_prediction(members, model, inputX):
    # create dataset using ensemble
    stackedX = stacked_dataset(members, inputX)
    # make a prediction
    yhat = model.predict(stackedX)
    return yhat

def split_data(data, test_size):
    df_samples_slim = pd.read_excel(data, engine='openpyxl')
    df_samples_slim.dropna()

    x = df_samples_slim.iloc[:, 3:].values
    y = df_samples_slim.iloc[:, np.r_[0]].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=0)
    return x_train, x_test, y_train, y_test


def read_test_data(path):
    df_samples_slim = pd.read_excel(path, engine='openpyxl')
    df_samples_slim.dropna()

    x_test_input = df_samples_slim.iloc[:, 2:].values
    y_test_input = df_samples_slim.iloc[:, np.r_[0]].values

    return x_test_input, y_test_input

#
# # load all models
# n_members = 9
# members = load_all_models(n_members)
# print('Loaded %d models' % len(members))
# # evaluate standalone models on test dataset
#
# # load the data
# data = ['../../data/overlap_data/1_0-150.xlsx', '../../data/overlap_data/2_100-300.xlsx',
#         '../../data/overlap_data/3_250-400.xlsx', '../../data/overlap_data/4_350-500.xlsx',
#         '../../data/overlap_data/5_450-600.xlsx', '../../data/overlap_data/6_550-700.xlsx',
#         '../../data/overlap_data/7_650-800.xlsx', '../../data/overlap_data/8_750-900.xlsx',
#         '../../data/overlap_data/9_850-1000.xlsx']
# data_no_overlap = ['../../data/no_overlap_data/1_0-100.xlsx', '../../data/no_overlap_data/2_100-200.xlsx',
#                    '../../data/no_overlap_data/3_200-300.xlsx','../../data/no_overlap_data/4_300-400.xlsx',
#                    '../../data/no_overlap_data/5_400-500.xlsx', '../../data/no_overlap_data/6_500-600.xlsx',
#                    '../../data/no_overlap_data/7_600-700.xlsx', '../../data/no_overlap_data/8_700-800.xlsx',
#                    '../../data/no_overlap_data/9_800-900.xlsx',
#                    '../../data/no_overlap_data/10_900-1000.xlsx']
# X_train=[]
# X_test=[]
# Y_train=[]
# Y_test=[]
#
# #overlap
# train_X_list = [0, 0 , 0 ,0 ,0 , 0, 0 ,0 ,0]
# test_X_list = [0, 0 , 0 ,0 ,0 , 0, 0 ,0 ,0]
# train_Y_list = [0, 0 , 0 ,0 ,0 , 0, 0 ,0 ,0]
# test_Y_list = [0, 0 , 0 ,0 ,0 , 0, 0 ,0 ,0]
# test_size=0.30
# for i in range (n_members):
#     train_X_list[i], test_X_list[i], train_Y_list[i], test_Y_list[i] = split_data(data[i],test_size)
#
# X_train = np.concatenate([train_X_list[0],train_X_list[1],train_X_list[2],train_X_list[3],train_X_list[4],train_X_list[5],train_X_list[6],train_X_list[7],train_X_list[8]], axis=0)
# Y_train = np.concatenate([train_Y_list[0],train_Y_list[1],train_Y_list[2],train_Y_list[3],train_Y_list[4],train_Y_list[5],train_Y_list[6],train_Y_list[7],train_Y_list[8]], axis=0)
# X_test = np.concatenate([test_X_list[0],test_X_list[1],test_X_list[2],test_X_list[3],test_X_list[4],test_X_list[5],test_X_list[6],test_X_list[7],test_X_list[8]], axis=0)
# Y_test = np.concatenate([test_Y_list[0],test_Y_list[1],test_Y_list[2],test_Y_list[3],test_Y_list[4],test_Y_list[5],test_Y_list[6],test_Y_list[7],test_Y_list[8]], axis=0)
# savetxt('train_test_csv/train.csv', X_train, delimiter=',')
# savetxt('train_test_csv/train_target.csv', Y_train, delimiter=',')
# savetxt('train_test_csv/test_target.csv', Y_test, delimiter=',')
# savetxt('train_test_csv/test.csv', X_test, delimiter=',')
#
#
# test_data = [([0.000412, 0.000412, 0.000414, 0.000419, 0.000429, 0.000446, 0.000472, 0.000507], 962, 0.0011911),
#         ([0.002246, 0.000832, 0.000174, -6.5E-06, 0.000252, 0.000666, 0.001077, 0.001298], 100, 0.0033459),
#         ([0.001899, 0.000254, 3.79E-05, 0.000552, 0.001173, 0.001254, 0.000798, 0.000219], 150, 0.0033459),
#         ([0.000644, 0.00102, 0.000683, 0.000299, 0.00148, 3.7E-05, 0.001166, 0.000771], 400, 0.0033459),
#         ([0.00024, 0.001535, 0.000256, 0.00108, 0.000632, 0.00064, 0.001072, 0.00024], 550, 0.0033459),
#         ([0.000141, 0.001343, 0.000991, 0.000146, 0.001467, 0.000171, 0.000805, 0.001136], 613, 0.0033459),
#         ([0.001995056, 0.000384732, 7.27776E-06, 0.000173547, 0.000734034, 0.001227756, 0.000814199, 0.000205494], 133,
#          0.0033459),
#         ([0.004237513, 0.002804209, 0.004348325, 0.003654791, 0.002554861, 0.003922156, 0.004598436, 0.002963461], 280,
#          0.005123),
#         ([0.002521808, 0.00306296, 0.002482601, 0.002346224, 0.003380631, 0.001844147, 0.003376766, 0.002300145], 410,
#          0.0033459),
#         ([0.001851032, 0.004503032, 0.002379226, 0.002995499, 0.003568136, 0.001829615, 0.004420003, 0.001480542], 570,
#          0.0033459),
#         ([0.020610, 0.007828, 0.001629, 0.000031, 0.002070, 0.005964, 0.009762, 0.011873], 98, 0.026224),
#         ([0.001795, 0.000660, 0.002334, 0.000600, 0.000542, 0.002654, 0.001223, 0.000146], 310, 0.005124),
#         ([0.002382, 0.017252, 0.005019, 0.009109, 0.011488, 0.002556, 0.016603, 0.000017], 569, 0.026224),
#         ([0.023458, 0.005550, 0.000064, 0.002581, 0.008901, 0.014021, 0.014610, 0.010491], 126, 0.026224),
#         ([0.000288, 0.005461, 0.017336, 0.017272, 0.004422, 0.000996, 0.012234, 0.016377], 759, 0.026224)
#         ]
#
# x_test = []
# y_test = []
#
#
# # fit stacked model using the ensemble
# model = fit_stacked_model(members, X_train, Y_train)
# # # # evaluate model on test set
# test_csv_path = "../../data/test_data/test_segment_400-500.xlsx"
# x_test1, y_test1 = read_test_data(test_csv_path)
#
# y_pred = stacked_prediction(members, model,x_test1)
# fig = plt.figure(figsize=(6, 5))
# for i in range (0,len(y_pred)):
#     print("Predicted pos: ",y_pred[i], "actual pos: ", y_test1[i])
#     plt.plot(i, y_pred[i], 'o', color='red')
#     plt.plot(i, y_test1[i], 'o', color='green')
# error = mean_squared_error(y_test1, y_pred)
# print("Error(MSE)",format(float(error), '.7f'))
#
#
# # # Save plots
# plt.savefig('../../plots/overlap/test.png')
# # # plt.show()
#
#
# # # Write predictions in a text file
# f = open("../../predictions/overlap/test.txt", "w")
# for x in y_pred:
#     stripped_pred = str(x).strip("[]")
#     f.write(stripped_pred+"\n")
# f.close()