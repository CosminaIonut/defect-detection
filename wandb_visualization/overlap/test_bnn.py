import numpy as np

from model_training.overlap.bayesian_models import Pbnn
from model_training.overlap.model_generating import split_data
from model_training.overlap.stacked_models_performance import read_test_data
from wandb_visualization.wandb_config import wandb_init
import pickle

def train_models_and_save_bnn(n_members,data,test_size,group):
    n_members=1
    data = ['../../data/total_data.xlsx']
    for i in range(n_members):
        # fit model
        name = "model-"+str(i+1)
        wandb_config = wandb_init(name, group)
        x_train, X_test, y_train, Y_test = split_data(data[i], test_size)
        config = {"n_infeatures": 8,
                  "n_outfeatures": 1,
                  "n_samples": len(x_train),
                  "learn_all_params": True,
                  "fixed_param": 0.3}

        mybnn = Pbnn(config)
        mybnn.build_bnn(2, [8, 36])
        train_env = {"batch_size": 150,
                     "epochs": 1000,
                     "callback_patience": 10000,
                     "verbose": 1}
        mybnn.train_bnn(x_train, y_train, train_env)
        with open("test", "wb") as fp:   #Pickling
            pickle.dump(mybnn.weights, fp)
        with open("test", "rb") as fp:  # Unpickling
            b = pickle.load(fp)
        # mybnn.weights = b
        #     TEST ---
        test_csv_path = "../../data/test_data/date_test_RNN.xlsx"
        # test_csv_path = "../../data/test_data/date_test_1000.xlsx"
        # test_csv_path = "../../data/test_data/test_segment_400-500.xlsx"
        x_test, y_test = read_test_data(test_csv_path)

        Mean_Y, Stdv_Y = mybnn.test_bnn(x_test)
        for i in range(0,len(Mean_Y)):
            print(Mean_Y[i], "---------",y_test[i])
        Mean_LL = mybnn.evaluate_bnn(x_test, y_test)
        print(np.mean(Mean_LL, axis=0))

        # # Write predictions in a text file
        f = open("../../predictions/overlap/BNN.txt", "w")
        for x in Mean_Y:
            stripped_pred = str(x).strip("[]")
            f.write(stripped_pred+"\n")
        f.close()
