import wandb

from model_training.severity.model_generation_severity import split_data, train_model, build_nn
from wandb_visualization.wandb_config import wandb_init

# load the data
data_no_overlap = ['../../data/no_overlap_data/1_0-100.xlsx', '../../data/no_overlap_data/2_100-200.xlsx',
                   '../../data/no_overlap_data/3_200-300.xlsx','../../data/no_overlap_data/4_300-400.xlsx',
                   '../../data/no_overlap_data/5_400-500.xlsx', '../../data/no_overlap_data/6_500-600.xlsx',
                   '../../data/no_overlap_data/7_600-700.xlsx', '../../data/no_overlap_data/8_700-800.xlsx',
                   '../../data/no_overlap_data/9_800-900.xlsx',
                   '../../data/no_overlap_data/10_900-1000.xlsx']

# create the 10 models
# fit and save models
n_members = 10

# training parameters
test_size = 0.30
epochs = 2
learning_rate =0.001
# learning_rate =0.0001
# learning_rate =0.01
# group = "experiment-" + wandb.util.generate_id()
group = "experiment-severity2-" + "lr0.001"
for i in range(n_members):
    # fit model
    name = "model-"+str(i+1)
    wandb_config = wandb_init(name, group)
    x_train, x_test, y_train, y_test = split_data(data_no_overlap[i], test_size)
    network, history = train_model(build_nn(), x_train, y_train, x_test, y_test, epochs)
    # save model
    filename = '../../trained_models/models_segments_severity' + str(test_size*100) + '_'+str(learning_rate) + 'epochs/model_' + str(i + 1) + '.h5'
    network.save(filename)
    print('>Saved %s' % filename)
    print(history.history.keys())
    wandb_config.finish()


