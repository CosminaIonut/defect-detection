import wandb
from model_training.overlap.model_generating import split_data, build_nn, train_model, build_baseline_nn, \
    build_dropout_visible_nn, build_dropout_hidden_nn, build_nn_sigmoid, build_nn_sigmoid_new_network, \
    build_nn_sigmoid_new_network_more_neurons
from wandb_visualization.wandb_config import wandb_init

# load the data
data = ['../../data/overlap_data/1_0-150.xlsx', '../../data/overlap_data/2_100-300.xlsx',
        '../../data/overlap_data/3_250-400.xlsx', '../../data/overlap_data/4_350-500.xlsx',
        '../../data/overlap_data/5_450-600.xlsx', '../../data/overlap_data/6_550-700.xlsx',
        '../../data/overlap_data/7_650-800.xlsx', '../../data/overlap_data/8_750-900.xlsx',
        '../../data/overlap_data/9_850-1000.xlsx']

# create the 10 models
# fit and save models
n_members = 9

# training parameters
test_size = 0.30
epochs = 150
learning_rate =0.001
# learning_rate =0.0001
# learning_rate =0.01
# group = "experiment-" + wandb.util.generate_id()
# group = "overlap" +"dropout-visiblelayer" +str(epochs)+"epochs"
# group = "overlap" +"dropout-hiddenlayer" +str(epochs)+"epochs"
# group = "overlap" +"-baseline-SGD-" +str(epochs)+"epochs"

# group = "overlap" +"-normal-SGD-" +str(epochs)+"epochs"
# group = "overlap" +"-dropout-visiblelayer-SGD-" +str(epochs)+"epochs"
# group = "overlap" +"-dropout-hiddenlayer-SGD-" +str(epochs)+"epochs"

# group = "overlap" +"-normal-sigmoid-SGD-" +str(epochs)+"epochs"
# group = "overlap" +"-new-network-sigmoid-adam" +str(epochs)+"epochs"

# ----------------------More Neurons ------------------------------------

# group = "overlap" +"-new-network-sigmoid-adam-20IN-" +str(epochs)+"epochs"
# group = "overlap" +"-new-network-sigmoid-adam-50IN-" +str(epochs)+"epochs"
# group = "overlap" +"-new-network-sigmoid-adam-8IN-5HN-" +str(epochs)+"epochs"
# group = "overlap" +"-new-network-sigmoid-adam-8IN-4HN-" +str(epochs)+"epochs"
group = "overlap" +"-new-network-sigmoid-adam-8IN-20HN-" +str(epochs)+"epochs"
# group = "overlap" +"-new-network-sigmoid-adam-8IN-50HN-" +str(epochs)+"epochs"
# group = "overlap" +"-new-network-sigmoid-adam-8IN-100HN-" +str(epochs)+"epochs"

for i in range(n_members):
    # fit model
    name = "model-"+str(i+1)
    wandb_config = wandb_init(name, group)
    x_train, x_test, y_train, y_test = split_data(data[i], test_size)
    network, history = train_model(build_nn_sigmoid_new_network_more_neurons(), x_train, y_train, x_test, y_test, epochs)
    # save model
    # filename = '../../trained_models/models_segments_overlap-normal-sigmoid-SGD-' + str(test_size*100) + '_'+str(epochs) + 'epochs/model_' + str(i + 1) + '.h5'
    # filename = '../../trained_models/models_segments_overlap-new-network-sigmoid-adam-' + str(test_size*100) + '_'+str(epochs) + 'epochs/model_' + str(i + 1) + '.h5'
    # filename = '../../trained_models/models_segments_overlap-dropout-visiblelayer-SGD' + str(test_size*100) + '_'+str(epochs) + 'epochs/model_' + str(i + 1) + '.h5'

    # ------------------------------ MORE NEURONS ---------------------------------
    # filename = '../../trained_models/models_segments_overlap-new-network-sigmoid-adam-20IN-' + str(test_size*100) + '_'+str(epochs) + 'epochs/model_' + str(i + 1) + '.h5'
    # filename = '../../trained_models/models_segments_overlap-new-network-sigmoid-adam-50IN-' + str(test_size*100) + '_'+str(epochs) + 'epochs/model_' + str(i + 1) + '.h5'
    # filename = '../../trained_models/models_segments_overlap-new-network-sigmoid-adam-8IN-5HN-' + str(test_size*100) + '_'+str(epochs) + 'epochs/model_' + str(i + 1) + '.h5'
    # filename = '../../trained_models/models_segments_overlap-new-network-sigmoid-adam-8IN-4HN-' + str(test_size*100) + '_'+str(epochs) + 'epochs/model_' + str(i + 1) + '.h5'
    filename = '../../trained_models/models_segments_overlap-new-network-sigmoid-adam-8IN-20HN-' + str(test_size*100) + '_'+str(epochs) + 'epochs/model_' + str(i + 1) + '.h5'
    # filename = '../../trained_models/models_segments_overlap-new-network-sigmoid-adam-8IN-50HN-' + str(test_size*100) + '_'+str(epochs) + 'epochs/model_' + str(i + 1) + '.h5'
    # filename = '../../trained_models/models_segments_overlap-new-network-sigmoid-adam-8IN-100HN-' + str(test_size*100) + '_'+str(epochs) + 'epochs/model_' + str(i + 1) + '.h5'

    network.save(filename)
    print('>Saved %s' % filename)
    print(history.history.keys())
    wandb_config.finish()


