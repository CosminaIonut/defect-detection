import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import wandb
from model_training.overlap.model_generating import split_data, train_model_sweep, build_nn_sweep

sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'val_mse',
        'goal': 'minimize',
        },
    'parameters': {
        'batch_size': {
            # integers between 32 and 256
            # with evenly-distributed logarithms
            'distribution': 'q_log_uniform_values',
            'q': 8,
            'min': 13,
            'max': 256,
        },
        'epochs': {'values': [20, 30, 50, 150]},
        'learning_rate': {
            # a flat distribution between 0 and 0.1
            'distribution': 'uniform',
            'min': 0,
            'max': 0.1
        },
        'hidden_layer_size': {'values': [[8,36], [20], [70],[128,512,90],[8]]},
        #
        # 'hidden_layer_size': {'values': [[20], [21],[22],[23],[24],[25],[26],[27],[28],[29],
        #                                  [30], [31],[32],[33],[34],[35],[36],[37],[38],[39],[40]
        #                                  ]},
        'optimizer': {'values': ['sgd','rmsprop','adam']},
        'patience': {'values': [1, 3, 5, 10, 15]},
        'monitor': {'values': ['val_loss', 'val_mse']}


     }
}
wandb.login(key="44af7bf1f24c6aab99ae33b0ae4fa5a5c8a59590", relogin=True)
sweep_id = wandb.sweep(sweep_config,   project="Defect-Detection-Sweep", entity="cosminaionut",)
#
# data = ['../data/overlap_data/1_0-150.xlsx', '../data/overlap_data/2_100-300.xlsx',
#         '../data/overlap_data/3_250-400.xlsx', '../data/overlap_data/4_350-500.xlsx',
#         '../data/overlap_data/5_450-600.xlsx', '../data/overlap_data/6_550-700.xlsx',
#         '../data/overlap_data/7_650-800.xlsx', '../data/overlap_data/8_750-900.xlsx',
#         '../data/overlap_data/9_850-1000.xlsx']
# data = ['../data/overlap_data/1_0-200.xlsx', '../data/overlap_data/2_0-300.xlsx',
#         '../data/overlap_data/3_100-400.xlsx', '../data/overlap_data/4_200-500.xlsx',
#         '../data/overlap_data/5_300-600.xlsx', '../data/overlap_data/6_400-700.xlsx',
#         '../data/overlap_data/7_500-800.xlsx', '../data/overlap_data/8_600-900.xlsx',
#         '../data/overlap_data/9_700-1000.xlsx','../data/overlap_data/10_800-1000.xlsx']

data = ['../data/overlap_data/0-50.xlsx','../data/overlap_data/0_0-100.xlsx',
        '../data/overlap_data/1_0-200.xlsx', '../data/overlap_data/2_0-300.xlsx',
        '../data/overlap_data/3_100-400.xlsx', '../data/overlap_data/4_200-500.xlsx',
        '../data/overlap_data/5_300-600.xlsx', '../data/overlap_data/6_400-700.xlsx',
        '../data/overlap_data/7_500-800.xlsx', '../data/overlap_data/8_600-900.xlsx',
        '../data/overlap_data/9_700-1000.xlsx','../data/overlap_data/10_800-1000.xlsx',
        '../data/overlap_data/11_900-1000.xlsx','../data/overlap_data/950-1000.xlsx']

# create the 9 models
# fit and save models
n_members = 14
# training parameters
test_size = 0.30

# start a new wandb run to track this script
def train(config=None):
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
        for i in range(n_members):
            # fit model
            x_train, x_test, y_train, y_test = split_data(data[i], test_size)

            # --------------------------ANN ----------------------------
            network, history = train_model_sweep(build_nn_sweep(config.optimizer, config.learning_rate, config.hidden_layer_size), x_train, y_train, x_test,
                                           y_test, config.epochs, config.batch_size, config.patience, config.monitor)


            filename = '../trained_models/ANN/models_segments_overlap-sweep' \
                       + '_' + str(config.optimizer) + '_' + str(config.learning_rate) + 'LR_' \
                       + str(config.hidden_layer_size) + 'HN_' + str(config.batch_size) + 'BS_' \
                       + str(config.patience) + 'P_' + str(config.monitor) + 'M_' \
                       + str(config.epochs) + 'epochs/model_' + str(i + 1) + '.h5'

            network.save(filename)
            print('>Saved %s' % filename)
            print(history.history.keys())



wandb.finish()
wandb.agent(sweep_id, train, count=20)