import numpy as np
import pandas as pd
from keras.losses import mean_squared_error

from model_training.overlap.stacked_models_performance import load_all_models, split_data, fit_stacked_model, \
    read_test_data, stacked_prediction
from wandb_visualization.wandb_config import wandb_init
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# group = "experiment-" + wandb.util.generate_id()

n_members = 9
members = load_all_models(n_members)
print('Loaded %d models' % len(members))

# load the data
data = ['../../data/overlap_data/1_0-150.xlsx', '../../data/overlap_data/2_100-300.xlsx',
        '../../data/overlap_data/3_250-400.xlsx', '../../data/overlap_data/4_350-500.xlsx',
        '../../data/overlap_data/5_450-600.xlsx', '../../data/overlap_data/6_550-700.xlsx',
        '../../data/overlap_data/7_650-800.xlsx', '../../data/overlap_data/8_750-900.xlsx',
        '../../data/overlap_data/9_850-1000.xlsx']

X_train=[]
X_test=[]
Y_train=[]
Y_test=[]

#overlap
train_X_list = [0, 0 , 0 ,0 ,0 , 0, 0 ,0 ,0]
test_X_list = [0, 0 , 0 ,0 ,0 , 0, 0 ,0 ,0]
train_Y_list = [0, 0 , 0 ,0 ,0 , 0, 0 ,0 ,0]
test_Y_list = [0, 0 , 0 ,0 ,0 , 0, 0 ,0 ,0]
test_size = 0.30
for i in range (n_members):
    train_X_list[i], test_X_list[i], train_Y_list[i], test_Y_list[i] = split_data(data[i],test_size)

X_train = np.concatenate([train_X_list[0],train_X_list[1],train_X_list[2],train_X_list[3],train_X_list[4],train_X_list[5],train_X_list[6],train_X_list[7],train_X_list[8]], axis=0)
Y_train = np.concatenate([train_Y_list[0],train_Y_list[1],train_Y_list[2],train_Y_list[3],train_Y_list[4],train_Y_list[5],train_Y_list[6],train_Y_list[7],train_Y_list[8]], axis=0)
X_test = np.concatenate([test_X_list[0],test_X_list[1],test_X_list[2],test_X_list[3],test_X_list[4],test_X_list[5],test_X_list[6],test_X_list[7],test_X_list[8]], axis=0)
Y_test = np.concatenate([test_Y_list[0],test_Y_list[1],test_Y_list[2],test_Y_list[3],test_Y_list[4],test_Y_list[5],test_Y_list[6],test_Y_list[7],test_Y_list[8]], axis=0)

# fit stacked model using the ensemble
model = fit_stacked_model(members, X_train, Y_train)
# # # evaluate model on test set
test_csv_path = "../../data/test_data/date_test.xlsx"
# test_csv_path = "../../data/test_data/test_segment_400-500.xlsx"
x_test, y_test = read_test_data(test_csv_path)

y_pred = stacked_prediction(members, model, x_test)
list1=[]
list2=[]
list3=[]
error_list = []
i=1
for x in y_pred:
    stripped_pred = str(x).strip("[]")
    list1.append(float(stripped_pred))
    list3.append(i)
    i += 1
for x in y_test:
    stripped_pred = str(x).strip("[]")
    list2.append(float(stripped_pred))

for i in range(len(list1)):
    error = (abs(list1[i] - list2[i])) * 1000
    error_list.append(error)

data = {'index_id' : list3,
        'actual_pos' : list2,
        'prediction' : list1,
        'error':error_list}

predicted_data = {'index_id': list3,
        'prediction' : list1}

actual_data = {'index_id': list3,
        'actual_pos' : list2}

df = pd.DataFrame(data)
predicted_data_df = pd.DataFrame(predicted_data)
actual_data_df = pd.DataFrame(actual_data)

group = "test-model-overlap"

# ------------------------- Test Baseline  -------------------------------------

# name = 'baseline-sgd-datetest-50epochs'
# name = 'baseline-RMSprop-datetest-50epochs'

# ------------------------- Test with RMSprop optimizer -------------------------------------

# name = 'dropout-visiblelayer-RMSprop-datetest-50epochs'
# name = 'dropout-hiddenlayer-RMSprop-datetest-50epochs'
# name = 'normal-RMSprop-datetest-50epochs'
# name = 'normal-RMSprop-datetest-1epoch'

# ------------------------- Test with SGD optimizer -------------------------------------

# name = 'dropout-visiblelayer-SGD-datetest-50epochs'
# name = 'dropout-hiddenlayer-SGD-datetest-50epochs'
# name = 'normal-SGD-datetest-50epochs'
# name = 'normal-SGD-datetest-1epoch'

# ------------------------- Test with new network -------------------------------------
# name = 'normal-sigmoid-SGD-datatest'
# name = 'new-network-sigmoid-adam-datatest-50epochs'
#  ---------------------------Neurons Number -------------------------------------------
# name = 'new-network-sigmoid-adam-20IN-50epochs'
# name = 'new-network-sigmoid-adam-50IN-50epochs'
# name = 'new-network-sigmoid-adam-8IN-5HN-50epochs'
# name = 'new-network-sigmoid-adam-8IN-4HN-50epochs'
# name = 'new-network-sigmoid-adam-8IN-20HN-50epochs'
# name = 'new-network-sigmoid-adam-8IN-50HN-50epochs'
# name = 'new-network-sigmoid-adam-8IN-100HN-50epochs'
name = 'new-network-sigmoid-adam-8IN-20HN-150epochs'



wandb_config = wandb_init(name, group)
wandb_config.log({'dataset': df})
# wandb_config.log({"predicted" : predicted_data_df})
# wandb_config.log({"actual" : actual_data_df})

# plot with actual vs predicted
fig = px.scatter(
    data, x='actual_pos', y='prediction', size_max=7)
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

#  second plot with actual and predicted
for i in range (0,len(y_pred)):
    print("Predicted pos: ",y_pred[i], "actual pos: ", y_test[i])
    plt.plot(i, y_pred[i], 'o', color='red')
    plt.plot(i, y_test[i], 'o', color='green')
error = mean_squared_error(y_test, y_pred)
print("Error(MSE)", format(float(error), '.7f'))
wandb_config.log({"actual_vs_predicted" : fig})
wandb_config.log({"actual_and_predicted_vs_index" : plt})
wandb_config.log({"test_mse": float(error)})
wandb_config.finish()
