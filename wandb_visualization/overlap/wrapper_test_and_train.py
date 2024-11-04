import numpy as np
from model_training.overlap.stacked_models_performance import load_all_models, split_data, fit_stacked_model, \
    read_test_data, stacked_prediction

# load the data
from model_training.overlap.stacked_models_performance import read_test_data
from wandb_visualization.overlap.test_and_train import test_models, train_models_and_save, train_models_and_save_RNN, \
    train_models_and_save_CNN, train_models_and_save_1D_CNN
import tensorflow as tf
# data = ['../../data/overlap_data/1_0-150.xlsx', '../../data/overlap_data/2_100-300.xlsx',
#         '../../data/overlap_data/3_250-400.xlsx', '../../data/overlap_data/4_350-500.xlsx',
#         '../../data/overlap_data/5_450-600.xlsx', '../../data/overlap_data/6_550-700.xlsx',
#         '../../data/overlap_data/7_650-800.xlsx', '../../data/overlap_data/8_750-900.xlsx',
#         '../../data/overlap_data/9_850-1000.xlsx']
# data = ['../../data/overlap_data/0_0-100.xlsx','../../data/overlap_data/1_0-200.xlsx', '../../data/overlap_data/2_0-300.xlsx',
#         '../../data/overlap_data/3_100-400.xlsx', '../../data/overlap_data/4_200-500.xlsx',
#         '../../data/overlap_data/5_300-600.xlsx', '../../data/overlap_data/6_400-700.xlsx',
#         '../../data/overlap_data/7_500-800.xlsx', '../../data/overlap_data/8_600-900.xlsx',
#         '../../data/overlap_data/9_700-1000.xlsx','../../data/overlap_data/10_800-1000.xlsx',
#         '../../data/overlap_data/11_900-1000.xlsx']

# data = ['../../data/overlap_data/0_0-100.xlsx','../../data/overlap_data/0_0-100.xlsx',
#          '../../data/overlap_data/0_0-100.xlsx',
#         '../../data/overlap_data/0_0-100.xlsx', '../../data/overlap_data/0_0-100.xlsx',
#         '../../data/overlap_data/0_0-100.xlsx', '../../data/overlap_data/0_0-100.xlsx',
#         '../../data/overlap_data/0_0-100.xlsx', '../../data/overlap_data/0_0-100.xlsx',
#         '../../data/overlap_data/0_0-100.xlsx','../../data/overlap_data/0_0-100.xlsx',
#         '../../data/overlap_data/0_0-100.xlsx']
#
# data = ['../../data/total_data.xlsx','../../data/total_data.xlsx',
#         '../../data/total_data.xlsx',
#         '../../data/total_data.xlsx', '../../data/total_data.xlsx',
#         '../../data/total_data.xlsx', '../../data/total_data.xlsx',
#         '../../data/total_data.xlsx', '../../data/total_data.xlsx',
#         '../../data/total_data.xlsx','../../data/total_data.xlsx',
#         '../../data/total_data.xlsx']

  # 14 with 0-1
from wandb_visualization.overlap.test_bnn import train_models_and_save_bnn

data = ['../../data/overlap_data/0-50.xlsx','../../data/overlap_data/0_0-100.xlsx',
        '../../data/overlap_data/1_0-200.xlsx', '../../data/overlap_data/2_0-300.xlsx',
        '../../data/overlap_data/3_100-400.xlsx', '../../data/overlap_data/4_200-500.xlsx',
        '../../data/overlap_data/5_300-600.xlsx', '../../data/overlap_data/6_400-700.xlsx',
        '../../data/overlap_data/7_500-800.xlsx', '../../data/overlap_data/8_600-900.xlsx',
        '../../data/overlap_data/9_700-1000.xlsx','../../data/overlap_data/10_800-1000.xlsx',
        '../../data/overlap_data/11_900-1000.xlsx','../../data/overlap_data/950-1000.xlsx']

#   14 with 0-1000
# data = ['../../data/overlap_data_1000/0-50.xlsx','../../data/overlap_data_1000/0_0-100.xlsx',
#         '../../data/overlap_data_1000/1_0-200.xlsx', '../../data/overlap_data_1000/2_0-300.xlsx',
#         '../../data/overlap_data_1000/3_100-400.xlsx', '../../data/overlap_data_1000/4_200-500.xlsx',
#         '../../data/overlap_data_1000/5_300-600.xlsx', '../../data/overlap_data_1000/6_400-700.xlsx',
#         '../../data/overlap_data_1000/7_500-800.xlsx', '../../data/overlap_data_1000/8_600-900.xlsx',
#         '../../data/overlap_data_1000/9_700-1000.xlsx','../../data/overlap_data_1000/10_800-1000.xlsx',
#         '../../data/overlap_data_1000/11_900-1000.xlsx','../../data/overlap_data_1000/950-1000.xlsx']
#
# data = ['../../data/overlap_sev0/0-50.xlsx','../../data/overlap_sev0/0_0-100.xlsx',
#         '../../data/overlap_sev0/1_0-200.xlsx', '../../data/overlap_sev0/2_0-300.xlsx',
#         '../../data/overlap_sev0/3_100-400.xlsx', '../../data/overlap_sev0/4_200-500.xlsx',
#         '../../data/overlap_sev0/5_300-600.xlsx', '../../data/overlap_sev0/6_400-700.xlsx',
#         '../../data/overlap_sev0/7_500-800.xlsx', '../../data/overlap_sev0/8_600-900.xlsx',
#         '../../data/overlap_sev0/9_700-1000.xlsx','../../data/overlap_sev0/10_800-1000.xlsx',
#         '../../data/overlap_sev0/11_900-1000.xlsx','../../data/overlap_sev0/950-1000.xlsx']

# create the 9 models
# fit and save models
n_members = 14
test_size = 0.20

X_train = []
X_test = []
Y_train = []
Y_test = []

# overlap
# train_X_list = [0, 0, 0, 0, 0, 0, 0, 0, 0]
# test_X_list = [0, 0, 0, 0, 0, 0, 0, 0, 0]
# train_Y_list = [0, 0, 0, 0, 0, 0, 0, 0, 0]
# test_Y_list = [0, 0, 0, 0, 0, 0, 0, 0, 0]

# overlap --10
# train_X_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# test_X_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# train_Y_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# test_Y_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# overlap ---12
# train_X_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0]
# test_X_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0]
# train_Y_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0]
# test_Y_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0]

# overlap ---14
train_X_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0]
test_X_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0]
train_Y_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0]
test_Y_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0]


for i in range(n_members):
    train_X_list[i], test_X_list[i], train_Y_list[i], test_Y_list[i] = split_data(data[i], test_size)
#
# X_train = np.concatenate(
#     [train_X_list[0], train_X_list[1], train_X_list[2], train_X_list[3], train_X_list[4], train_X_list[5],
#      train_X_list[6], train_X_list[7], train_X_list[8]], axis=0)
# Y_train = np.concatenate(
#     [train_Y_list[0], train_Y_list[1], train_Y_list[2], train_Y_list[3], train_Y_list[4], train_Y_list[5],
#      train_Y_list[6], train_Y_list[7], train_Y_list[8]], axis=0)
# X_test = np.concatenate(
#     [test_X_list[0], test_X_list[1], test_X_list[2], test_X_list[3], test_X_list[4], test_X_list[5], test_X_list[6],
#      test_X_list[7], test_X_list[8]], axis=0)
# Y_test = np.concatenate(
#     [test_Y_list[0], test_Y_list[1], test_Y_list[2], test_Y_list[3], test_Y_list[4], test_Y_list[5], test_Y_list[6],
#      test_Y_list[7], test_Y_list[8]], axis=0)

# ----------- 10 models more overlap
# X_train = np.concatenate(
#     [train_X_list[0], train_X_list[1], train_X_list[2], train_X_list[3], train_X_list[4], train_X_list[5],
#      train_X_list[6], train_X_list[7], train_X_list[8],train_X_list[9]], axis=0)
# Y_train = np.concatenate(
#     [train_Y_list[0], train_Y_list[1], train_Y_list[2], train_Y_list[3], train_Y_list[4], train_Y_list[5],
#      train_Y_list[6], train_Y_list[7], train_Y_list[8],train_Y_list[9]], axis=0)
# X_test = np.concatenate(
#     [test_X_list[0], test_X_list[1], test_X_list[2], test_X_list[3], test_X_list[4], test_X_list[5], test_X_list[6],
#      test_X_list[7], test_X_list[8],test_X_list[9]], axis=0)
# Y_test = np.concatenate(
#     [test_Y_list[0], test_Y_list[1], test_Y_list[2], test_Y_list[3], test_Y_list[4], test_Y_list[5], test_Y_list[6],
#      test_Y_list[7], test_Y_list[8],test_Y_list[9]], axis=0)

# # ----------- 12 models more overlap
# X_train = np.concatenate(
#     [train_X_list[0], train_X_list[1], train_X_list[2], train_X_list[3], train_X_list[4], train_X_list[5],
#      train_X_list[6], train_X_list[7], train_X_list[8],train_X_list[9],train_X_list[10],train_X_list[11]], axis=0)
# Y_train = np.concatenate(
#     [train_Y_list[0], train_Y_list[1], train_Y_list[2], train_Y_list[3], train_Y_list[4], train_Y_list[5],
#      train_Y_list[6], train_Y_list[7], train_Y_list[8],train_Y_list[9],train_Y_list[10],train_Y_list[11]], axis=0)
# X_test = np.concatenate(
#     [test_X_list[0], test_X_list[1], test_X_list[2], test_X_list[3], test_X_list[4], test_X_list[5], test_X_list[6],
#      test_X_list[7], test_X_list[8],test_X_list[9],test_X_list[10],test_X_list[11]], axis=0)
# Y_test = np.concatenate(
#     [test_Y_list[0], test_Y_list[1], test_Y_list[2], test_Y_list[3], test_Y_list[4], test_Y_list[5], test_Y_list[6],
#      test_Y_list[7], test_Y_list[8],test_Y_list[9],test_Y_list[10],test_Y_list[11]], axis=0)

# ----------- 14 models more overlap
X_train = np.concatenate(
    [train_X_list[0], train_X_list[1], train_X_list[2], train_X_list[3], train_X_list[4], train_X_list[5],
     train_X_list[6], train_X_list[7], train_X_list[8],train_X_list[9],train_X_list[10],train_X_list[11],train_X_list[12],train_X_list[13]], axis=0)
Y_train = np.concatenate(
    [train_Y_list[0], train_Y_list[1], train_Y_list[2], train_Y_list[3], train_Y_list[4], train_Y_list[5],
     train_Y_list[6], train_Y_list[7], train_Y_list[8],train_Y_list[9],train_Y_list[10],train_Y_list[11],train_Y_list[12],train_Y_list[13]], axis=0)
X_test = np.concatenate(
    [test_X_list[0], test_X_list[1], test_X_list[2], test_X_list[3], test_X_list[4], test_X_list[5], test_X_list[6],
     test_X_list[7], test_X_list[8],test_X_list[9],test_X_list[10],test_X_list[11],test_X_list[12],test_X_list[13]], axis=0)
Y_test = np.concatenate(
    [test_Y_list[0], test_Y_list[1], test_Y_list[2], test_Y_list[3], test_Y_list[4], test_Y_list[5], test_Y_list[6],
     test_Y_list[7], test_Y_list[8],test_Y_list[9],test_Y_list[10],test_Y_list[11],test_Y_list[12],test_Y_list[13]], axis=0)



test_csv_path = "../../data/test_data/date_test.xlsx"
# test_csv_path = "../../data/test_data/test_data_FEM.xlsx"
# test_csv_path = "../../data/test_data/test_data_measured.xlsx"
# test_csv_path = "../../data/test_data/date_test_RNN.xlsx"
# test_csv_path = "../../data/test_data/date_test_1000.xlsx"
# test_csv_path = "../../data/test_data/test_segment_400-500.xlsx"
x_test, y_test = read_test_data(test_csv_path)

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def trainin_param_ANN():
    batch_size = 100
    # batch_size = 30
    optimizer = 'adam'
    hidden_layer_size = [36]
    epochs = 300
    # learning_rate = 0.09925
    # learning_rate = 0.001
    learning_rate = 0.04270
    patience = 10
    monitor = 'val_mse'
    return batch_size, epochs, learning_rate, hidden_layer_size, optimizer, patience, monitor

def trainin_param_RNN():
    batch_size = 8
    optimizer = 'adam'
    hidden_layer_size = [10]
    epochs = 150
    learning_rate = 0.06896
    patience = 15
    monitor = 'val_mse'
    dense_units = 8
    activation = ['sigmoid', 'tanh','tanh']
    return batch_size, epochs, learning_rate, hidden_layer_size, dense_units, optimizer, patience, monitor,activation

def trainin_param_CNN():
    batch_size = 64
    epochs = 150
    # learning_rate = 0.1176
    # cnn_layer_size = [124,516]
    # cnn_input_nodes = 32
    learning_rate = 0.01337803
    cnn_layer_size = [23]
    cnn_input_nodes = 100
    optimizer = 'adam'
    patience = 15
    monitor = 'val_loss'
    dense_units = 100
    return batch_size, epochs, learning_rate, cnn_layer_size, cnn_input_nodes, optimizer, patience, monitor,dense_units

def trainin_param_1DCNN():
    batch_size = 64
    epochs = 300
    learning_rate = 0.01337803
    cnn_layer_size = [23]
    cnn_input_nodes = 100
    optimizer = 'adam'
    patience = 15
    monitor = 'val_loss'
    dense_units = 100

    return batch_size, epochs, learning_rate, cnn_layer_size, cnn_input_nodes, optimizer, patience, monitor,dense_units



# ---------------------------------------------------- TRAIN  ANN --------------------------------------------

def train_ANN():

    # training parameters
    batch_size, epochs, learning_rate, hidden_layer_size, optimizer, patience, monitor = trainin_param_ANN()
    train_group = "best-overlap-models-" + 'ANN' + '_' + str(optimizer) + '_' + str(learning_rate) + 'LR_' \
                  + str(hidden_layer_size) + 'HN_' + str(batch_size) + 'BS_' \
                  + str(patience) + 'P_' + str(monitor) + 'M_' \
                  + str(epochs) + 'epochs'
    train_models_and_save(n_members,data,test_size,train_group, epochs, batch_size, learning_rate,optimizer,hidden_layer_size,patience,monitor)

# ---------------------------------------------------- TRAIN RNN --------------------------------------------

def train_RNN():
    # training parameters
    batch_size, epochs, learning_rate, hidden_layer_size, dense_units, optimizer, patience, monitor, activation = trainin_param_RNN()
    train_group_RNN = "more-overlap-9-" + 'RNN' + '_' + str(optimizer) + '_' + str(learning_rate) + 'LR_' \
                      + str(hidden_layer_size) + 'HU_' + str(batch_size) + 'BS_' \
                      + str(dense_units) + 'DU_' + str(activation) + '_' \
                      + str(patience) + 'P_' + str(monitor) + 'M_' + str(dense_units) + 'DU_' \
                      + str(epochs)+'epochs'

    train_models_and_save_RNN(n_members,data,test_size,train_group_RNN, epochs, batch_size, learning_rate,optimizer,hidden_layer_size,patience,monitor,activation,dense_units)


# ---------------------------------------------------- TRAIN  CNN --------------------------------------------
def train_CNN():
    # training parameters

    batch_size,epochs,learning_rate,cnn_layer_size,cnn_input_nodes,optimizer,patience,monitor,dense_units = trainin_param_CNN()
    train_group_cnn = "overlap-test-" + 'CNN' + '_' + str(optimizer) + '_' + str(learning_rate) + 'LR_' \
                      + str(cnn_layer_size) + 'CNNLS_' + str(cnn_input_nodes) + 'CNNIN_'+ str(batch_size) + 'BS_' \
                      + str(patience) + 'P_' + str(monitor) + 'M_'+ str(dense_units) + 'DU_' \
                      + str(epochs)+'epochs'

    train_models_and_save_CNN(n_members,data,test_size,train_group_cnn, epochs, batch_size, learning_rate,optimizer,cnn_layer_size, cnn_input_nodes, patience,monitor,dense_units)




def train_1DCNN():
    # training parameters

    batch_size,epochs,learning_rate,cnn_layer_size,cnn_input_nodes,optimizer,patience,monitor,dense_units = trainin_param_1DCNN()
    train_group_cnn = "overlap-test-" + '1DCNN' + '_' + str(optimizer) + '_' + str(learning_rate) + 'LR_' \
                      + str(cnn_layer_size) + 'CNNLS_' + str(cnn_input_nodes) + 'CNNIN_'+ str(batch_size) + 'BS_' \
                      + str(patience) + 'P_' + str(monitor) + 'M_'+ str(dense_units) + 'DU_' \
                      + str(epochs)+'epochs'

    train_models_and_save_1D_CNN(n_members,data,test_size,train_group_cnn, epochs, batch_size, learning_rate,optimizer,cnn_layer_size, cnn_input_nodes, patience,monitor,dense_units)

test_group = "test-model-overlap"


#  -------------------------------------------------- TEST --------------------------------------------

# --------------------------------------------ANN ----------------------------------------------------------------


def test_ANN():

    # test_model_path_ANN = 'ANN/models_segments_overlap_baseline'
    # test_run_name_ANN ='ANN-baseline'
    # test_model_path_ANN = "ANN/models_segments_overlap_adam_0.04270280954958349LR_[36]HN_32BS_10P_val_lossM_200epochs"
    # test_run_name_ANN = 'ANN'+ test_model_path_ANN

    batch_size, epochs, learning_rate, hidden_layer_size, optimizer, patience, monitor = trainin_param_ANN()


    # test_model_path_ANN = 'ANN/best-models_segments_overlap' \
    #            + '_' + str(optimizer) + '_' + str(learning_rate) + 'LR_' \
    #            + str(hidden_layer_size) + 'HN_' + str(batch_size) + 'BS_' \
    #            + str(patience) + 'P_' + str(monitor) + 'M_' \
    #            + str(epochs) + 'epochs'
    test_model_path_ANN = 'ANN/best-models_segments_overlap_adam_0.0427LR_[36]HN_100BS_10P_val_mseM_300epochs'
    test_run_name_ANN = '- '+ test_model_path_ANN
    # test_model_path_ANN = 'ANN/models_segments_overlap'


    test_models(n_members, X_train, Y_train, x_test, y_test, test_group, test_run_name_ANN, test_model_path_ANN)



# --------------------------------------------CNN ----------------------------------------------------------------

def test_CNN():
    # test_model_path_cnn = 'models_segments_overlap-cnn_adam_0.08991095538972839LR_[23]CHN_32CNNI_120BS_1P_val_lossM_20epochs'
    # test_run_name_cnn = 'CNN-'+ test_model_path_cnn
    # test_model_path_cnn = 'CNN/models_segments_overlap-cnn_adam_0.0797465282647894LR_[20]CHN_20CNNI_240BS_1DU_5P_val_mseM_20epochs'
    # test_model_path_cnn = 'CNN/models_segments_overlap-cnn_rmsprop_0.011832556194175592LR_[32]CHN_50CNNI_56BS_40DU_15P_val_lossM_50epochs'
    # test_model_path_cnn = 'CNN/models_segments_overlap-cnn_adam_0.02770054981221334LR_[37]CHN_64CNNI_16BS_1000DU_10P_val_lossM_50epochs'
    # test_model_path_cnn = 'CNN/models_segments_overlap-cnn_adam_0.028809837774000313LR_[30]CHN_32CNNI_200BS_1DU_5P_val_mseM_50epochs'
    # test_model_path_cnn = 'CNN/models_segments_overlap-cnn-more_adam_0.013378030192505152LR_[23]CHN_100CNNI_64BS_100DU_15P_val_lossM_150epochs'
    # test_run_name_cnn = 'CNN-normal' + test_model_path_cnn
    # test_run_name_cnn = 'CNN-maxpooling' + test_model_path_cnn
    # test_run_name_cnn = 'CNN-maxpooling_4,2,1shape' + test_model_path_cnn
    # test_run_name_cnn = 'CNN-normal-4,2,1shape' + test_model_path_cnn
    # test_run_name_cnn = 'CNN-more-overlap-maxpooling-8,1,1shape' + test_model_path_cnn
    batch_size, epochs, learning_rate, cnn_layer_size, cnn_input_nodes, optimizer, patience, monitor,dense_units = trainin_param_CNN()
    # test_run_name_cnn = 'CNN-' + '_' + str(optimizer) + '_' + str(learning_rate) + 'LR_' \
    #                    + str(cnn_layer_size) + 'HN_' + str(cnn_layer_size) + 'HN_'+ str(batch_size) + 'BS_' \
    #                    + str(patience) + 'P_' + str(monitor) + 'M_' + str(dense_units) + 'DU_'\
    #                    + str(epochs) + 'epochs'
    #
    test_model_path_cnn = 'CNN/models_segments_overlap' \
                       + '_' + str(optimizer) + '_' + str(learning_rate) + 'LR_' \
                       + str(cnn_layer_size) + 'HN_' + str(cnn_layer_size) + 'HN_'+ str(batch_size) + 'BS_' \
                       + str(patience) + 'P_' + str(monitor) + 'M_'\
                       + str(epochs) + 'epochs'


    test_run_name_cnn = 'FEM' + test_model_path_cnn
    # x_train_cnn = X_train.reshape(X_train.shape[0], 4, 2, 1)
    # x_test_cnn = x_test.reshape(x_test.shape[0], 4, 2, 1)
    x_train_cnn = X_train.reshape(X_train.shape[0], 8, 1, 1)
    x_test_cnn = x_test.reshape(x_test.shape[0], 8, 1 , 1)
    test_models(n_members, x_train_cnn, Y_train, x_test_cnn, y_test, test_group, test_run_name_cnn, test_model_path_cnn)

def test_1DCNN():
    batch_size, epochs, learning_rate, cnn_layer_size, cnn_input_nodes, optimizer, patience, monitor,dense_units = trainin_param_CNN()
    # test_run_name_cnn = 'CNN-' + '_' + str(optimizer) + '_' + str(learning_rate) + 'LR_' \
    #                    + str(cnn_layer_size) + 'HN_' + str(cnn_layer_size) + 'HN_'+ str(batch_size) + 'BS_' \
    #                    + str(patience) + 'P_' + str(monitor) + 'M_' + str(dense_units) + 'DU_'\
    #                    + str(epochs) + 'epochs'
    #
    test_model_path_cnn = '1D-CNN/models_segments_overlap' \
                       + '_' + str(optimizer) + '_' + str(learning_rate) + 'LR_' \
                       + str(cnn_input_nodes) + 'IN_' + str(cnn_layer_size) + 'HN_'+ str(batch_size) + 'BS_' \
                       + str(patience) + 'P_' + str(monitor) + 'M_'\
                       + str(epochs) + 'epochs'
    test_run_name_cnn = test_model_path_cnn
    x_train_cnn = X_train.reshape(X_train.shape[0], 8, 1)
    x_test_cnn = x_test.reshape(x_test.shape[0], 8, 1)
    test_models(n_members, x_train_cnn, Y_train, x_test_cnn, y_test, test_group, test_run_name_cnn, test_model_path_cnn)



# --------------------------------------------RNN ----------------------------------------------------------------

def test_RNN():

    # test_model_path_rnn = 'models_segments_overlap-cnn_adam_0.08991095538972839LR_[23]CHN_32CNNI_120BS_1P_val_lossM_20epochs'
    # test_model_path_rnn = 'RNN/models_segments_overlap-rnn_rmsprop_0.0777768335966716LR_[6]HL2DU_16BS_10P_val_mseM_150epochs'
    # test_model_path_rnn = 'RNN/models_segments_overlap-rnn_rmsprop_0.0664462509744353LR_[8]HL2DU_8BS_15P_val_mseM_50epochs'
    # test_model_path_rnn = 'RNN/models_segments_overlap-rnn_adam_0.09588550860047763LR_[8]HL8DU_8BS_10P_val_lossM_150epochs'
    # test_model_path_rnn = 'RNN/models_segments_overlap-rnn_sgd_0.08062488847758388LR_[40]HL8DU_0BS_3P_val_lossM_50epochs'
    # test_model_path_rnn = 'RNN/models_segments_overlap-rnn_adam_0.06896211375291401LR_[10]HL8DU_8BS_15P_val_mseM_150epochs'
    # test_run_name_rnn = 'RNN-'+ test_model_path_rnn
    # test_run_name_rnn = 'LSTM-'+ test_model_path_rnn
    # test_run_name_rnn = 'LSTM-bidirectional-'+ test_model_path_rnn
    # test_run_name_rnn = 'GRU-'+ test_model_path_rnn
    # test_run_name_rnn = 'RNN-vertical'+ test_model_path_rnn
    # test_run_name_rnn = 'RNN-vertical'+ test_model_path_rnn
    # test_model_path_rnn = "RNN/models_segments_overlap-LSTMBI_adam_0.06896LR_[10]HL8DU_8BS_['sigmoid', 'tanh', 'tanh']_15P_val_mseM_8DU_150epochs"
    # test_model_path_rnn = "RNN/models_segments_overlap-rnn_adam_0.06896LR_[10]HL8DU_8BS_['sigmoid', 'tanh', 'tanh']_15P_val_mseM_8DU_150epochs"  --very goood  the best RNN
    #  more overlap
    # test_model_path_rnn = "RNN/models_segments_overlap-rnn_rmsprop_0.05783114821551175LR_[4]HL5DU_80BS_20P_val_mseM_150epochs"
    # test_model_path_rnn = "RNN/models_segments_overlap-rnn_adam_0.03469306864744246LR_[2]HL2DU_8BS_15P_val_lossM_150epochs"
    # test_model_path_rnn = "RNN/models_segments_overlap-rnn_adam_0.03864336762277235LR_[2]HL5DU_24BS_15P_val_lossM_150epochs"


    # test_run_name_rnn = 'RNN-more-overlap'+ test_model_path_rnn
    # test_run_name_rnn = 'LSTM-bidirectional-more-overlap'+ test_model_path_rnn




    batch_size, epochs, learning_rate, hidden_layer_size, dense_units, optimizer, patience, monitor, activation = trainin_param_RNN()
    # test_run_name_rnn = 'more-overlap-regulizers0.001-LSTM-BI' + '_' + str(optimizer) + '_' + str(learning_rate) + 'LR_' \
    #                    + str(hidden_layer_size) + 'HL' + str(dense_units) + 'DU_' \
    #                    + str(batch_size) + 'BS_' + str(activation) + '_' \
    #                    + str(patience) + 'P_' + str(monitor) + 'M_' \
    #                    + str(epochs) + 'epochs'

    # test_model_path_rnn = 'RNN/models_segments_overlap-9-RNN' \
    #                    + '_' + str(optimizer) + '_' + str(learning_rate) + 'LR_' \
    #                    + str(hidden_layer_size) + 'HL' + str(dense_units) + 'DU_' \
    #                    + str(batch_size) + 'BS_' + str(activation) + '_' \
    #                    + str(patience) + 'P_' + str(monitor) + 'M_' + str(dense_units) + 'DU_' \
    #                    + str(epochs) + 'epochs'
    test_model_path_rnn = "RNN/models_segments_overlap-14-LSTM-BI_adam_0.06896LR_[10]HL8DU_8BS_['sigmoid', 'tanh', 'tanh']_15P_val_mseM_8DU_150epochs"
    test_run_name_rnn = test_model_path_rnn
    x_train_rnn = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    x_test_rnn = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    test_models(n_members, x_train_rnn, Y_train, x_test_rnn, y_test, test_group, test_run_name_rnn, test_model_path_rnn)


# train_ANN()
# test_ANN()

# train_CNN()
# test_CNN()
#
# train_RNN()
test_RNN()


# train_1DCNN()
# test_1DCNN()
# train_models_and_save_bnn(n_members,data,test_size,"BNN")
