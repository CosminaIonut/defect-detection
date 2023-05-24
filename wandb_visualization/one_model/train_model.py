import wandb

from model_training.one_model.one_model_generating import split_data, train_model, build_nn, build_nn_sweep, \
    train_model_sweep
from wandb_visualization.wandb_config import wandb_init

# load the data
data = ['../../data/overlap_data/4_350-500.xlsx']

# create the 10 models
# fit and save models

# training parameters
test_size = 0.30

# learning_rate =0.0001
# learning_rate =0.01
# group = "experiment-" + wandb.util.generate_id()
# group = "one-model" +"dropout-visiblelayer" +str(epochs)+"epochs"
# group = "one-model" +"dropout-hiddenlayer" +str(epochs)+"epochs"
# group = "one-model" +"-baseline-SGD-" +str(epochs)+"epochs"


optimizer = 'adam',
hidden_layer_size = [30,30]
batch_size = 16
patience = 10
monitor ='val_mse'
epochs = 100
learning_rate = 0.001


group = "one-model" +"-" +str(epochs)+"epochs"

# fit model
name = "model-1"
wandb_config = wandb_init(name, group)
x_train, x_test, y_train, y_test = split_data(data[0], test_size)
print(y_train)
network, history = train_model_sweep(build_nn_sweep(optimizer, learning_rate, hidden_layer_size), x_train,
                                             y_train, x_test, y_test, epochs, batch_size, patience, monitor)
# save model
filename = '../../trained_models/one_model/models_segments_one_model_sec4'+ str(learning_rate)+"LR_"+ str(hidden_layer_size)+"HN" + '_'+str(epochs) + 'epochs/model_' + str(1) + '.h5'
network.save(filename)
print('>Saved %s' % filename)
print(history.history.keys())
wandb_config.finish()


