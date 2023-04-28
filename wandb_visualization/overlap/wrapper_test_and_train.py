import numpy as np
from model_training.overlap.stacked_models_performance import load_all_models, split_data, fit_stacked_model, \
    read_test_data, stacked_prediction

# load the data
from model_training.overlap.stacked_models_performance import read_test_data
from wandb_visualization.overlap.test_and_train import test_models, train_models_and_save, train_models_and_save_RNN, \
    train_models_and_save_CNN
import tensorflow as tf
data = ['../../data/overlap_data/1_0-150.xlsx', '../../data/overlap_data/2_100-300.xlsx',
        '../../data/overlap_data/3_250-400.xlsx', '../../data/overlap_data/4_350-500.xlsx',
        '../../data/overlap_data/5_450-600.xlsx', '../../data/overlap_data/6_550-700.xlsx',
        '../../data/overlap_data/7_650-800.xlsx', '../../data/overlap_data/8_750-900.xlsx',
        '../../data/overlap_data/9_850-1000.xlsx']

# create the 9 models
# fit and save models
n_members = 9
test_size = 0.30

X_train = []
X_test = []
Y_train = []
Y_test = []

# overlap
train_X_list = [0, 0, 0, 0, 0, 0, 0, 0, 0]
test_X_list = [0, 0, 0, 0, 0, 0, 0, 0, 0]
train_Y_list = [0, 0, 0, 0, 0, 0, 0, 0, 0]
test_Y_list = [0, 0, 0, 0, 0, 0, 0, 0, 0]

for i in range(n_members):
    train_X_list[i], test_X_list[i], train_Y_list[i], test_Y_list[i] = split_data(data[i], test_size)

X_train = np.concatenate(
    [train_X_list[0], train_X_list[1], train_X_list[2], train_X_list[3], train_X_list[4], train_X_list[5],
     train_X_list[6], train_X_list[7], train_X_list[8]], axis=0)
Y_train = np.concatenate(
    [train_Y_list[0], train_Y_list[1], train_Y_list[2], train_Y_list[3], train_Y_list[4], train_Y_list[5],
     train_Y_list[6], train_Y_list[7], train_Y_list[8]], axis=0)
X_test = np.concatenate(
    [test_X_list[0], test_X_list[1], test_X_list[2], test_X_list[3], test_X_list[4], test_X_list[5], test_X_list[6],
     test_X_list[7], test_X_list[8]], axis=0)
Y_test = np.concatenate(
    [test_Y_list[0], test_Y_list[1], test_Y_list[2], test_Y_list[3], test_Y_list[4], test_Y_list[5], test_Y_list[6],
     test_Y_list[7], test_Y_list[8]], axis=0)

test_csv_path = "../../data/test_data/date_test.xlsx"
# test_csv_path = "../../data/test_data/test_segment_400-500.xlsx"
x_test, y_test = read_test_data(test_csv_path)

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def trainin_param_ANN():
    batch_size = 32
    optimizer = 'adam'
    hidden_layer_size = [34]
    epochs = 10
    learning_rate = 0.09925
    patience = 10
    monitor = 'val_mse'
    return batch_size, epochs, learning_rate, hidden_layer_size, optimizer, patience, monitor

def trainin_param_RNN():
    batch_size = 8
    optimizer = 'adam'
    hidden_layer_size = [10]
    epochs = 100
    learning_rate = 0.06896
    patience = 15
    monitor = 'val_mse'
    dense_units = 8
    activation = ['sigmoid', 'tanh']
    return batch_size, epochs, learning_rate, hidden_layer_size, dense_units, optimizer, patience, monitor,activation

def trainin_param_CNN():
    batch_size = 40
    epochs = 20
    learning_rate = 0.1176
    cnn_layer_size = [124,516]
    cnn_input_nodes = 32
    optimizer = 'sgd'
    patience = 1
    monitor = 'val_loss'
    dense_units = 1

    return batch_size, epochs, learning_rate, cnn_layer_size, cnn_input_nodes, optimizer, patience, monitor,dense_units

# ---------------------------------------------------- TRAIN  ANN --------------------------------------------

def train_ANN():

    # training parameters
    batch_size, epochs, learning_rate, hidden_layer_size, optimizer, patience, monitor = trainin_param_ANN()
    train_group = "overlap-test-" + 'ANN' + '_' + str(optimizer) + '_' + str(learning_rate) + 'LR_' \
                      + str(hidden_layer_size) + 'HN_' + str(batch_size) + 'BS_' \
                      + str(patience) + 'P_' + str(monitor) + 'M_' \
                      + str(epochs)+'epochs'

    train_models_and_save(n_members,data,test_size,train_group, epochs, batch_size, learning_rate,optimizer,hidden_layer_size,patience,monitor)

# ---------------------------------------------------- TRAIN RNN --------------------------------------------

def train_RNN():
    # training parameters
    batch_size, epochs, learning_rate, hidden_layer_size, dense_units, optimizer, patience, monitor, activation = trainin_param_RNN()
    train_group_RNN = "overlap-test-" + 'RNN' + '_' + str(optimizer) + '_' + str(learning_rate) + 'LR_' \
                      + str(hidden_layer_size) + 'HU_' + str(batch_size) + 'BS_' \
                      + str(dense_units) + 'DU_' + str(activation) + '_' \
                      + str(patience) + 'P_' + str(monitor) + 'M_'  \
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

test_group = "test-model-overlap"

#  -------------------------------------------------- TEST --------------------------------------------

# --------------------------------------------ANN ----------------------------------------------------------------


def test_ANN():

    # test_run_name_ANN ='ANN-adam_0.03837015216285987LR_[35]HN_16BS_10P_val_mseM_200epochs1234'
    # test_model_path_ANN = 'models_segments_overlap_adam_0.03837015216285987LR_[35]HN_16BS_10P_val_mseM_200epochs'

    batch_size, epochs, learning_rate, hidden_layer_size, optimizer, patience, monitor = trainin_param_ANN()
    test_run_name_ANN = 'ANN-' + '_' + str(optimizer) + '_' + str(learning_rate) + 'LR_' \
                        + str(hidden_layer_size) + 'HN_' + str(batch_size) + 'BS_' \
                        + str(patience) + 'P_' + str(monitor) + 'M_' \
                        + str(epochs)+'epochs'

    test_model_path_ANN = 'ANN/models_segments_overlap' \
               + '_' + str(optimizer) + '_' + str(learning_rate) + 'LR_' \
               + str(hidden_layer_size) + 'HN_' + str(batch_size) + 'BS_' \
               + str(patience) + 'P_' + str(monitor) + 'M_' \
               + str(epochs) + 'epochs'


    test_models(n_members, X_train, Y_train, x_test, y_test, test_group, test_run_name_ANN, test_model_path_ANN)



# --------------------------------------------CNN ----------------------------------------------------------------

def test_CNN():
    # test_model_path_cnn = 'models_segments_overlap-cnn_adam_0.08991095538972839LR_[23]CHN_32CNNI_120BS_1P_val_lossM_20epochs'
    # test_run_name_cnn = 'CNN-'+ test_model_path_cnn

    batch_size, epochs, learning_rate, cnn_layer_size, cnn_input_nodes, optimizer, patience, monitor,dense_units = trainin_param_CNN()
    test_run_name_cnn = 'CNN-' + '_' + str(optimizer) + '_' + str(learning_rate) + 'LR_' \
                       + str(cnn_layer_size) + 'HN_' + str(cnn_layer_size) + 'HN_'+ str(batch_size) + 'BS_' \
                       + str(patience) + 'P_' + str(monitor) + 'M_' + str(dense_units) + 'DU_'\
                       + str(epochs) + 'epochs'

    test_model_path_cnn = 'CNN/models_segments_overlap' \
                       + '_' + str(optimizer) + '_' + str(learning_rate) + 'LR_' \
                       + str(cnn_layer_size) + 'HN_' + str(cnn_layer_size) + 'HN_'+ str(batch_size) + 'BS_' \
                       + str(patience) + 'P_' + str(monitor) + 'M_' + str(dense_units) + 'DU_'\
                       + str(epochs) + 'epochs'

    x_train_cnn = X_train.reshape(X_train.shape[0], 8, 1, 1)
    x_test_cnn = x_test.reshape(x_test.shape[0], 8, 1, 1)
    test_models(n_members, x_train_cnn, Y_train, x_test_cnn, y_test, test_group, test_run_name_cnn, test_model_path_cnn)



# --------------------------------------------RNN ----------------------------------------------------------------

def test_RNN():

    # test_model_path_rnn = 'models_segments_overlap-cnn_adam_0.08991095538972839LR_[23]CHN_32CNNI_120BS_1P_val_lossM_20epochs'
    # test_run_name_rnn = 'RNN-'+ test_model_path_rnn

    batch_size, epochs, learning_rate, hidden_layer_size, dense_units, optimizer, patience, monitor, activation = trainin_param_RNN()
    test_run_name_rnn = 'RNN-' + '_' + str(optimizer) + '_' + str(learning_rate) + 'LR_' \
                       + str(hidden_layer_size) + 'HL' + str(dense_units) + 'DU_' \
                       + str(batch_size) + 'BS_' + str(activation) + '_' \
                       + str(patience) + 'P_' + str(monitor) + 'M_' \
                       + str(epochs) + 'epochs'

    test_model_path_rnn = 'RNN/models_segments_overlap-rnn' \
                       + '_' + str(optimizer) + '_' + str(learning_rate) + 'LR_' \
                       + str(hidden_layer_size) + 'HL' + str(dense_units) + 'DU_' \
                       + str(batch_size) + 'BS_' + str(activation) + '_' \
                       + str(patience) + 'P_' + str(monitor) + 'M_' \
                       + str(epochs) + 'epochs'

    x_train_rnn = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    x_test_rnn = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    test_models(n_members, x_train_rnn, Y_train, x_test_rnn, y_test, test_group, test_run_name_rnn, test_model_path_rnn)


# train_ANN()
# test_ANN()
#
# train_CNN()
# test_CNN()
#
# train_RNN()
# test_RNN()
