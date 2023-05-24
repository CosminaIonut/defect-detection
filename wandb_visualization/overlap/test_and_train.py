import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd

from model_training.overlap.bayesian_models import Pbnn
from model_training.overlap.model_generating import build_nn_sweep, build_nn_baseline, train_model_sweep, \
    build_nn_sweep_levenberg, build_nn_sweep_BNN
from model_training.overlap.model_generating_CNN import build_cnn_sweep, train_model_sweep, build_cnn_sweep_maxpooling, \
    build_1Dcnn_sweep_maxpooling
from model_training.overlap.model_generating_RNN import build_RNN_sweep, train_model_sweep, build_RNN_sweep_LSTM, \
    build_RNN_sweep_LSTM_bidirectional, build_RNN_sweep_GRU
from model_training.overlap.stacked_models_performance import load_all_models, split_data, fit_stacked_model,stacked_prediction
from wandb_visualization.wandb_config import wandb_init
import matplotlib.pyplot as plt
import plotly.express as px

def get_XY(x_train, y_train, time_steps):
    print(x_train.shape)
    X_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    return X_train, y_train

def train_models_and_save(n_members,data,test_size,group, epochs, batch_size, learning_rate, optimizer, hidden_layer_size,patience,monitor):
    for i in range(n_members):
        # fit model
        name = "model-"+str(i+1)
        wandb_config = wandb_init(name, group)
        x_train, x_test, y_train, y_test = split_data(data[i], test_size)
        # config = {"n_infeatures": 8,
        #           "n_outfeatures": 1,
        #           "n_samples": len(x_train),
        #           "learn_all_params": True,
        #           "fixed_param": 0.3}
        #
        # mybnn = Pbnn(config)
        # mybnn.build_bnn(2, [8, 36])

        # network, history = train_model_sweep(build_nn_sweep(optimizer, learning_rate, hidden_layer_size), x_train,
        #                                      y_train, x_test, y_test, epochs, batch_size,patience,monitor)
        # save model
        network, history = train_model_sweep(build_nn_sweep(optimizer, learning_rate, hidden_layer_size), x_train,
                                             y_train, x_test, y_test, epochs, batch_size, patience, monitor)
        # sample_size = int(len(x_train)*test_size)
        # network, history = train_model_sweep(build_nn_sweep_BNN(optimizer, learning_rate, hidden_layer_size,sample_size), x_train,
        #                                      y_train, x_test, y_test, epochs, batch_size, patience, monitor)
        filename = '../../trained_models/ANN/best-models_segments_overlap' \
                   + '_' + str(optimizer) + '_' + str(learning_rate) + 'LR_' \
                   + str(hidden_layer_size) + 'HN_' + str(batch_size) + 'BS_' \
                   + str(patience) + 'P_' + str(monitor) + 'M_' \
                   + str(epochs) + 'epochs/model_' + str(i + 1) + '.h5'
        # filename = '../../trained_models/ANN/models_segments_overlap/model_' + str(i + 1) + '.h5'
        network.save(filename)
        print('>Saved %s' % filename)
        print(history.history.keys())
        wandb_config.finish()

def train_models_and_save_RNN(n_members,data,test_size,group, epochs, batch_size, learning_rate, optimizer, hidden_layer_size,patience,monitor, activation, dense_units):

    for i in range(9,n_members):
        # fit model
        name = "model-"+str(i+1)
        print(name)
        wandb_config = wandb_init(name, group)
        x_train, x_test, y_train, y_test = split_data(data[i], test_size)
        time_steps = x_train.shape[1]
        X, Y = get_XY(x_train, y_train, time_steps)
        X_test, Y_test = get_XY(x_test, y_test, time_steps)
        # --------------------------RNN ----------------------------
        # (optimizer, learning_rate, hidden_layer_size, dense_units,activation, length=8):
        network, history = train_model_sweep(
            build_RNN_sweep_LSTM_bidirectional(optimizer, learning_rate, hidden_layer_size, dense_units,
                            activation), X, Y,
            X_test, Y_test, epochs, batch_size, patience, monitor)
        filename = '../../trained_models/RNN/models_segments_overlap-14-LSTM-BI' \
                   + '_' + str(optimizer) + '_' + str(learning_rate) + 'LR_' \
                   + str(hidden_layer_size) + 'HL' + str(dense_units) + 'DU_' \
                   + str(batch_size) + 'BS_' + str(activation) + '_' \
                   + str(patience) + 'P_' + str(monitor) + 'M_' + str(dense_units) + 'DU_'\
                   + str(epochs) + 'epochs/model_' + str(i + 1) + '.h5'


        network.save(filename)
        print('>Saved %s' % filename)
        print(history.history.keys())
        wandb_config.finish()

def train_models_and_save_CNN(n_members,data,test_size,train_group_cnn, epochs, batch_size, learning_rate, optimizer, cnn_layer_size, cnn_input_nodes, patience,monitor,dense_units):
    for i in range(n_members):
        # fit model
        name = "model-"+str(i+1)
        wandb_config = wandb_init(name, train_group_cnn)
        x_train, x_test, y_train, y_test = split_data(data[i], test_size)
        x_train_cnn = x_train.reshape(x_train.shape[0], 8, 1, 1)
        x_test_cnn = x_test.reshape(x_test.shape[0], 8, 1, 1)
        network, history = train_model_sweep(build_cnn_sweep_maxpooling(optimizer, learning_rate, cnn_layer_size,cnn_input_nodes,dense_units), x_train_cnn,
                                             y_train, x_test_cnn, y_test, epochs, batch_size,patience,monitor)
        # save model
        filename = '../../trained_models/CNN/models_segments_overlap' \
                   + '_' + str(optimizer) + '_' + str(learning_rate) + 'LR_' \
                   + str(cnn_layer_size) + 'HN_' + str(cnn_layer_size) + 'HN_'+ str(batch_size) + 'BS_' \
                   + str(patience) + 'P_' + str(monitor) + 'M_' \
                   + str(epochs) + 'epochs/model_' + str(i + 1) + '.h5'

        network.save(filename)
        print('>Saved %s' % filename)
        print(history.history.keys())
        wandb_config.finish()

def train_models_and_save_1D_CNN(n_members,data,test_size,train_group_cnn, epochs, batch_size, learning_rate, optimizer, cnn_layer_size, cnn_input_nodes, patience,monitor,dense_units):
    for i in range(n_members):
        # fit model
        name = "model-"+str(i+1)
        wandb_config = wandb_init(name, train_group_cnn)
        x_train, x_test, y_train, y_test = split_data(data[i], test_size)
        x_train_cnn = x_train.reshape(x_train.shape[0], 8, 1)
        x_test_cnn = x_test.reshape(x_test.shape[0], 8, 1)
        n_timesteps = x_train_cnn.shape[1]

        network, history = train_model_sweep(build_1Dcnn_sweep_maxpooling(optimizer, learning_rate, cnn_layer_size,cnn_input_nodes,dense_units,n_timesteps), x_train_cnn,
                                             y_train, x_test_cnn, y_test, epochs, batch_size,patience,monitor)
        # save model
        filename = '../../trained_models/1D-CNN/models_segments_overlap' \
                   + '_' + str(optimizer) + '_' + str(learning_rate) + 'LR_' \
                   + str(cnn_input_nodes) + 'IN_' + str(cnn_layer_size) + 'HN_'+ str(batch_size) + 'BS_' \
                   + str(patience) + 'P_' + str(monitor) + 'M_' \
                   + str(epochs) + 'epochs/model_' + str(i + 1) + '.h5'

        network.save(filename)
        print('>Saved %s' % filename)
        print(history.history.keys())
        wandb_config.finish()

def test_models(n_members, x_train, y_train, x_test, y_test, group, run_name, path):
    members = load_all_models(n_members, path)
    # fit stacked model using the ensemble
    model = fit_stacked_model(members, x_train, y_train)
    # # # evaluate model on test set
    y_pred = stacked_prediction(members, model, x_test)

    list1 = []
    list2 = []
    list3 = []
    error_list = []
    i = 1
    for x in y_pred:
        stripped_pred = str(x).strip("[]")
        list1.append(float(stripped_pred))
        list3.append(i)
        i += 1
    for x in y_test:
        stripped_pred = str(x).strip("[]")
        list2.append(float(stripped_pred))

    for i in range(len(list1)):
        error = (abs(list1[i] - list2[i])) * 1000 *100 / 1000 # the error in %
        error_list.append(error)

    data_table = {'index_id': list3,
            'actual_pos': list2,
            'prediction': list1,
            'error(%)': error_list}

    df = pd.DataFrame(data_table)
    wandb_config = wandb_init(run_name, group)
    wandb_config.log({'dataset': df})

    # plot with actual vs predicted
    fig = px.scatter(
        data_table, x='actual_pos', y='prediction', size_max=7,labels=dict(actual_pos="Actual Position", prediction="Prediction"))
    fig.update_traces(marker={'size': 7})
    x1 = max(list2)
    y1 = max(list1)
    max_axes = max(y1, x1) + 0.1
    x1 = min(list2)
    y1 = min(list1)
    min_axes = min(y1, x1) - 0.1

    fig.update_layout(
        shapes=[
            dict(type="line",
                 x0=min_axes, y0=min_axes, x1=max_axes, y1=max_axes)])

    # plot error of prediction
    df_actual_pos_sort = pd.DataFrame(data_table)
    print(df_actual_pos_sort)
    sort = df_actual_pos_sort.sort_values(by=['actual_pos'])
    print(df_actual_pos_sort)
    error_plot = px.line(x=sort['actual_pos'], y=sort['error(%)'], markers=True,labels=dict(x="Actual Position", y="Error (%)"))
    y_max = max(error_list)
    if (y_max < 1):
        y_max = 1
    error_plot.update_yaxes(range=[0, y_max])

    #  second plot with actual and predicted
    for i in range(0, len(y_pred)):
        print("Predicted pos: ", y_pred[i], "actual pos: ", y_test[i])
        plt.plot(i, y_pred[i], 'o', color='red')
        plt.plot(i, y_test[i], 'o', color='green')

    error = mean_squared_error(y_test, y_pred)
    print("Error(MSE)", format(float(error), '.7f'))

    wandb_config.log({"actual_vs_predicted": fig})
    wandb_config.log({"actual_and_predicted_vs_index": plt})
    wandb_config.log({"Prediction Error": error_plot})
    wandb_config.log({"test_mse": float(error)})
    wandb_config.finish()

