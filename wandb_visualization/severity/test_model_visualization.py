import numpy as np
import pandas as pd
from model_training.severity.stacked_models_severity import load_all_models, split_data, fit_stacked_model, \
    read_test_data, stacked_prediction
from wandb_visualization.wandb_config import wandb_init
import matplotlib.pyplot as plt
import plotly.express as px

n_members = 9
members = load_all_models(n_members)
print('Loaded %d models' % len(members))

# load the data
# data_no_overlap = ['../../data/no_overlap_data/1_0-100.xlsx', '../../data/no_overlap_data/2_100-200.xlsx',
#                    '../../data/no_overlap_data/3_200-300.xlsx','../../data/no_overlap_data/4_300-400.xlsx',
#                    '../../data/no_overlap_data/5_400-500.xlsx', '../../data/no_overlap_data/6_500-600.xlsx',
#                    '../../data/no_overlap_data/7_600-700.xlsx', '../../data/no_overlap_data/8_700-800.xlsx',
#                    '../../data/no_overlap_data/9_800-900.xlsx',
#                    '../../data/no_overlap_data/10_900-1000.xlsx']

data = ['../../data/overlap_data/1_0-150.xlsx', '../../data/overlap_data/2_100-300.xlsx',
        '../../data/overlap_data/3_250-400.xlsx', '../../data/overlap_data/4_350-500.xlsx',
        '../../data/overlap_data/5_450-600.xlsx', '../../data/overlap_data/6_550-700.xlsx',
        '../../data/overlap_data/7_650-800.xlsx', '../../data/overlap_data/8_750-900.xlsx',
        '../../data/overlap_data/9_850-1000.xlsx']

X_train=[]
X_test=[]
Y_train=[]
Y_test=[]
# no_overlap
train_X_list = [0, 0 , 0 ,0 ,0 , 0, 0 ,0 ,0]
test_X_list = [0, 0 , 0 ,0 ,0 , 0, 0 ,0 ,0]
train_Y_list = [0, 0 , 0 ,0 ,0 , 0, 0 ,0 ,0 ]
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
test_csv_path = "../../data/test_data/test_segment_400-500.xlsx"
x_test, y_test = read_test_data(test_csv_path)

y_pred = stacked_prediction(members, model, x_test)
list_pred_pos=[]
list_actual_pos=[]
list_index=[]

list_pred_severity=[]
list_actual_severity=[]
error_list_pos = []
error_list_sev = []
i=1
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

data= {'index_id' : list_index,
        'actual_pos' : list_actual_pos,
        'prediction_pos' : list_pred_pos,
        'actual_severity' : list_actual_severity,
        'prediction_severity' : list_pred_severity,
        'error_pos':error_list_pos,
        'error_severity': error_list_sev,
            }

predicted_data = {'index_id': list_index,
        'prediction_pos' : list_pred_pos}

actual_data = {'index_id': list_index,
        'actual_pos' : list_actual_pos}

df = pd.DataFrame(data)
predicted_data_df = pd.DataFrame(predicted_data)
actual_data_df = pd.DataFrame(actual_data)
group = "test-model-severity"
name = 'test-segment4-bayes-severity-HN-20-40'
wandb_config = wandb_init(name, group)
wandb_config.log({'dataset': df})
# wandb_config.log({"predicted_pos" : predicted_data_df})
# wandb_config.log({"actual_pos" : actual_data_df})

# plot with actual vs predicted position
fig_pos = px.scatter(
    data, x='actual_pos', y='prediction_pos', size_max=7)
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
    data, x='actual_severity', y='prediction_severity', size_max=7)
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
# fig.show()


#  second plot with actual and predicted
for i in range (0,len(y_pred)):
    # print("Predicted pos: ",y_pred[i], "actual pos: ", y_test[i])
    plt.plot(i, y_pred[i][0], 'o', color='red')
    plt.plot(i, y_test[i][0], 'o', color='green')

wandb_config.log({"actual_vs_predicted_position" : fig_pos})
wandb_config.log({"actual_vs_predicted_severity" : fig_severity})
wandb_config.log({"actual_and_predicted_position" : plt})

plt.figure()
for i in range (0,len(y_pred)):
    # print("Predicted pos: ",y_pred[i], "actual pos: ", y_test[i])
    plt.plot(i, y_pred[i][1], 'o', color='red')
    plt.plot(i, y_test[i][1], 'o', color='green')

wandb_config.log({"actual_and_predicted_severity" : plt})
wandb_config.finish()

# # # Write predictions in a text file
# f = open("../../predictions/no_overlap/test.txt", "w")
# for x in y_pred:
#     stripped_pred = str(x).strip("[]")
#     f.write(stripped_pred+"\n")
# f.close()
