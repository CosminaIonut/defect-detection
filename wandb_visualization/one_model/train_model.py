import wandb

from model_training.one_model.one_model_generating import split_data, train_model, build_nn
from wandb_visualization.wandb_config import wandb_init

# load the data
data = ['../../data/total_data.xlsx']

# create the 10 models
# fit and save models

# training parameters
test_size = 0.30
epochs = 50
learning_rate =0.001
# learning_rate =0.0001
# learning_rate =0.01
# group = "experiment-" + wandb.util.generate_id()
# group = "one-model" +"dropout-visiblelayer" +str(epochs)+"epochs"
# group = "one-model" +"dropout-hiddenlayer" +str(epochs)+"epochs"
# group = "one-model" +"-baseline-SGD-" +str(epochs)+"epochs"
group = "one-model" +"-" +str(epochs)+"epochs"

# fit model
name = "model-1"
wandb_config = wandb_init(name, group)
x_train, x_test, y_train, y_test = split_data(data[0], test_size)
network, history = train_model(build_nn(), x_train, y_train, x_test, y_test, epochs)
# save model
filename = '../../trained_models/models_segments_one_model_' + str(test_size*100) + '_'+str(epochs) + 'epochs/model_' + str(1) + '.h5'
network.save(filename)
print('>Saved %s' % filename)
print(history.history.keys())
wandb_config.finish()


