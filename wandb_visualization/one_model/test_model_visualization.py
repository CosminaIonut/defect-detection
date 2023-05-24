import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from model_training.one_model.model_performance import load_one_model, split_data, read_test_data
from wandb_visualization.wandb_config import wandb_init
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


# test_model_path_ANN = "one_model/models_segments_one_model_0.03837LR_[35]HN_200epochs"
# test_model_path_ANN = "one_model/models_segments_one_model_new_ann0.03837LR_[35]HN_200epochs"
test_model_path_ANN = "one_model/models_segments_one_model_new_ann0.001LR_[10]HN_80epochs"
test_run_name_ANN = 'one-model-ANN'+ test_model_path_ANN
model = load_one_model(test_model_path_ANN)

# load the data
data = ['../../data/total_data.xlsx']

X_train=[]
X_test=[]
Y_train=[]
Y_test=[]

#one model
test_size = 0.30
X_train, X_test, Y_train, Y_test = split_data(data[0],test_size)
# fit stacked model using the ensemble
# model = fit_stacked_model(members, X_train, Y_train)
# # # evaluate model on test set

# test_csv_path = "../../data/test_data/date_test.xlsx"
test_csv_path = "../../data/test_data/data_seg4.xlsx"
x_test, y_test = read_test_data(test_csv_path)
y_pred = model.predict(x_test)
print(y_pred)

group = "test-model-one-model"

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
wandb_config = wandb_init(test_run_name_ANN, group)
wandb_config.log({'dataset': df})

# plot with actual vs predicted
fig = px.scatter(
        data_table, x='actual_pos', y='prediction', size_max=7)
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
error_plot = px.line(x=sort['actual_pos'], y=sort['error(%)'], markers=True)


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