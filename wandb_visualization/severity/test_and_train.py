import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd

from model_training.overlap.model_generating import build_nn_sweep, build_nn_baseline, train_model_sweep
from model_training.overlap.model_generating_CNN import build_cnn_sweep, train_model_sweep, build_cnn_sweep_maxpooling
from model_training.overlap.model_generating_RNN import build_RNN_sweep, train_model_sweep, build_RNN_sweep_LSTM, \
    build_RNN_sweep_LSTM_bidirectional, build_RNN_sweep_GRU
from model_training.overlap.stacked_models_performance import load_all_models, fit_stacked_model,stacked_prediction
from model_training.severity.model_generation_severity import split_data
from wandb_visualization.wandb_config import wandb_init
import matplotlib.pyplot as plt
import plotly.express as px

def get_XY(x_train, y_train, time_steps):
    print(x_train.shape)
    X_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    # Y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))
    return X_train, y_train

def train_models_and_save(n_members,data,test_size,group, epochs, batch_size, learning_rate, optimizer, hidden_layer_size,patience,monitor):
    for i in range(n_members):
        # fit model
        name = "model-"+str(i+1)
        wandb_config = wandb_init(name, group)
        x_train, x_test, y_train, y_test = split_data(data[i], test_size)

        # network, history = train_model_sweep(build_nn_sweep(optimizer, learning_rate, hidden_layer_size), x_train,
        #                                      y_train, x_test, y_test, epochs, batch_size,patience,monitor)
        # save model
        network, history = train_model_sweep(build_nn_sweep(optimizer, learning_rate, hidden_layer_size), x_train,
                                             y_train, x_test, y_test, epochs, batch_size, patience, monitor)
        filename = '../../trained_models/severity/ANN/models_segments_overlap' \
                   + '_' + str(optimizer) + '_' + str(learning_rate) + 'LR_' \
                   + str(hidden_layer_size) + 'HN_' + str(batch_size) + 'BS_' \
                   + str(patience) + 'P_' + str(monitor) + 'M_' \
                   + str(epochs) + 'epochs/model_' + str(i + 1) + '.h5'
        # filename = '../../trained_models/severity/ANN/models_segments_overlap_baseline/model_' + str(i + 1) + '.h5'
        network.save(filename)
        print('>Saved %s' % filename)
        print(history.history.keys())
        wandb_config.finish()

def train_models_and_save_RNN(n_members,data,test_size,group, epochs, batch_size, learning_rate, optimizer, hidden_layer_size,patience,monitor, activation, dense_units):

    for i in range(n_members):
        # fit model
        name = "model-"+str(i+1)
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
        filename = '../../trained_models/severity//RNN/models_segments_overlap-LSTM-BI' \
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
        filename = '../../trained_models/severity/CNN/models_segments_overlap' \
                   + '_' + str(optimizer) + '_' + str(learning_rate) + 'LR_' \
                   + str(cnn_layer_size) + 'HN_' + str(cnn_layer_size) + 'HN_'+ str(batch_size) + 'BS_' \
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

    list_pred_pos = []
    list_actual_pos = []
    list_index = []

    list_pred_severity = []
    list_actual_severity = []
    error_list_pos = []
    error_list_sev = []
    i = 1
    for x in y_pred:
        list_pred_pos.append(float(x[0]))
        list_index.append(i)
        i += 1
    for x in y_test:
        list_actual_pos.append(float(x[0]))

    for x in y_pred:
        list_pred_severity.append(float(x[1]))
    for x in y_test:
        list_actual_severity.append(float(x[1]))

    for i in range(len(list_pred_pos)):
        error_pos = (abs(list_pred_pos[i] - list_actual_pos[i])) * 1000
        error_sev = (abs(list_pred_severity[i] - list_actual_severity[i])) * 1000
        error_list_pos.append(error_pos)
        error_list_sev.append(error_sev)

    data_table = {'index_id': list_index,
            'actual_pos': list_actual_pos,
            'prediction_pos': list_pred_pos,
            'actual_severity': list_actual_severity,
            'prediction_severity': list_pred_severity,
            'error_pos(%)': error_list_pos,
            'error_severity(%)': error_list_sev,
            }

    predicted_data = {'index_id': list_index,
                      'prediction_pos': list_pred_pos}

    actual_data = {'index_id': list_index,
                   'actual_pos': list_actual_pos}

    df = pd.DataFrame(data_table)
    wandb_config = wandb_init(run_name, group)
    wandb_config.log({'dataset': df})

    predicted_data_df = pd.DataFrame(predicted_data)
    actual_data_df = pd.DataFrame(actual_data)

    # wandb_config.log({"predicted_pos" : predicted_data_df})
    # wandb_config.log({"actual_pos" : actual_data_df})

    # plot with actual vs predicted position
    fig_pos = px.scatter(
        data_table, x='actual_pos', y='prediction_pos', size_max=7)
    fig_pos.update_traces(marker={'size': 7})
    x1 = max(list_actual_pos)
    y1 = max(list_pred_pos)
    max_axes = max(y1, x1) + 0.1
    x1 = min(list_actual_pos)
    y1 = min(list_pred_pos)
    min_axes = min(y1, x1) - 0.1

    fig_pos.update_layout(
        shapes=[
            dict(type="line",
                 x0=min_axes, y0=min_axes, x1=max_axes, y1=max_axes)])
    # fig.show()

    # plot with actual vs predicted severity
    fig_severity = px.scatter(
        data_table, x='actual_severity', y='prediction_severity', size_max=7)
    fig_severity.update_traces(marker={'size': 7})
    x1 = max(list_actual_severity)
    y1 = max(list_pred_severity)
    max_axes = max(y1, x1) + 0.01
    x1 = min(list_actual_severity)
    y1 = min(list_pred_severity)
    min_axes = min(y1, x1) - 0.01

    fig_severity.update_layout(
        shapes=[
            dict(type="line",
                 x0=min_axes, y0=min_axes, x1=max_axes, y1=max_axes)])

    # plot error of prediction
    df_actual_pos_sort = pd.DataFrame(data_table)
    print(df_actual_pos_sort)
    sort = df_actual_pos_sort.sort_values(by=['actual_pos'])
    print(df_actual_pos_sort)
    error_plot_pos = px.line(x=sort['actual_pos'], y=sort['error_pos(%)'], markers=True)

    # plot error of prediction severity
    df_actual_pos_sort = pd.DataFrame(data_table)
    print(df_actual_pos_sort)
    sort = df_actual_pos_sort.sort_values(by=['actual_severity'])
    print(df_actual_pos_sort)
    error_plot_severity = px.line(x=sort['actual_severity'], y=sort['error_severity(%)'], markers=True)

    error = mean_squared_error(y_test, y_pred)
    print("Error(MSE)", format(float(error), '.7f'))

    wandb_config.log({"Actual vs Predicted Position": fig_pos})
    wandb_config.log({"Actual vs predicted Severity": fig_severity})
    wandb_config.log({"Prediction Error Position": error_plot_pos})
    wandb_config.log({"Prediction Error Severity": error_plot_severity})
    wandb_config.log({"test_mse": float(error)})
    wandb_config.finish()


