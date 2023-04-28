import sys
import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from model_training.overlap.model_generating_RNN import train_model_sweep, split_data, build_RNN_sweep, \
    build_RNN_sweep_LSTM, build_RNN_sweep_vertical, build_RNN_sweep_GRU, build_RNN_sweep_LSTM_bidirectional

import wandb

sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'val_mse',
        'goal': 'minimize',
        },
    'parameters': {
        'batch_size': {
            # with evenly-distributed logarithms
            'distribution': 'q_log_uniform_values',
            'q': 8,
            'min': 1,
            'max': 256,
        },
        'epochs': {'values': [20, 30, 50, 150]},
        'learning_rate': {
            # a flat distribution between 0 and 0.1
            'distribution': 'uniform',
            'min': 0,
            'max': 0.1
        },
        # 'hidden_layer_size': {'values': [[20], [30], [70], [20,10],[32,64],[64,64,64],[32,64,128],[128,128,128],[32,64,128,256]]},
        # 'hidden_layer_size': {'values': [[20], [21],[22],[23],[24],[25],[26],[27],[28],[29],
        #                                  [30], [31],[32],[33],[34],[35],[36],[37],[38],[39],[40]
        #                                  ]},
        'hidden_layer_size': {'values': [[20], [3],[10],[40],
                                      [8], [15], [6], [5]]},
        'dense_units': {'values': [1,8,5,3,2,4]},
        'optimizer': {'values': ['adam', 'sgd', 'rmsprop']},
        'patience': {'values': [1, 3, 5, 10, 15, 20]},
        'monitor': {'values': ['val_loss', 'val_mse']},
        'activation':{'values': [['tanh', 'tanh','tanh'],['tanh','sigmoid','softmax'],['sigmoid','tanh','sigmoid'],['relu','tanh','softmax']]}

     }
}
wandb.login(key="44af7bf1f24c6aab99ae33b0ae4fa5a5c8a59590", relogin=True)
sweep_id = wandb.sweep(sweep_config,   project="Defect-Detection-Sweep", entity="cosminaionut",)

data = ['../data/overlap_data/1_0-150.xlsx', '../data/overlap_data/2_100-300.xlsx',
        '../data/overlap_data/3_250-400.xlsx', '../data/overlap_data/4_350-500.xlsx',
        '../data/overlap_data/5_450-600.xlsx', '../data/overlap_data/6_550-700.xlsx',
        '../data/overlap_data/7_650-800.xlsx', '../data/overlap_data/8_750-900.xlsx',
        '../data/overlap_data/9_850-1000.xlsx']

# create the 9 models
# fit and save models
n_members = 9

# training parameters
test_size = 0.30



def get_XY(x_train, y_train, time_steps):
    print(x_train.shape)
    X_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    return X_train, y_train

# start a new wandb run to track this script
def train(config=None):
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
        for i in range(n_members):
            # fit model
            x_train, x_test, y_train, y_test = split_data(data[i], test_size)
            time_steps = x_train.shape[1]
            X, Y = get_XY(x_train,y_train,time_steps)
            X_test, Y_test = get_XY(x_test,y_test,time_steps)

            # --------------------------RNN ----------------------------
            # (optimizer, learning_rate, hidden_layer_size, dense_units,activation, length=8):
            network, history = train_model_sweep(
                build_RNN_sweep_vertical(config.optimizer, config.learning_rate, config.hidden_layer_size, config.dense_units,config.activation), X, Y,
                X_test,Y_test, config.epochs, config.batch_size, config.patience, config.monitor)
            filename = '../trained_models/RNN/models_segments_overlap-rnn' \
                       + '_' + str(config.optimizer) + '_' + str(config.learning_rate) + 'LR_' \
                       + str(config.hidden_layer_size) + 'HL'+ str(config.dense_units) + 'DU_' \
                       + str(config.batch_size) + 'BS_' \
                       + str(config.patience) + 'P_' + str(config.monitor) + 'M_' \
                       + str(config.epochs) + 'epochs/model_' + str(i + 1) + '.h5'
            #  -------------------------------------------------------------

            network.save(filename)
            print('>Saved %s' % filename)
            print(history.history.keys())

wandb.finish()
wandb.agent(sweep_id, train, count=10)