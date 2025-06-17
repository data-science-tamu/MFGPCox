import torch
import torch.nn as nn
import torch.nn.functional as F
import Model
import utils_simulation
# import utils_real
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import numpy as np

data, min, max = utils_simulation.load_data(Normalize=False)

# data  = pd.read_csv('./train_sina.csv')

train_data = data[data.columns.difference(['id', 'start', 'stop', 'died', 'x'])]

node_in = len(train_data.columns)

times = torch.tensor(data['stop'].values)
events = torch.tensor(data['died'].values)
train_data = torch.tensor(train_data.values).float()

# define model
node_hidden1 = 40
node_hidden2 = 20
node_out = 1
model = Model.CoxNet(input_dim=node_in, node_hidden1=node_hidden1, node_hidden2=node_hidden2, out_dim=node_out)

# loss
loss_type = 2

if loss_type == 1:
    criterion = Model.Efron_loss()
elif loss_type == 2:
    criterion = Model.Efron_loss_penalty(penalty=0.05)

learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

is_train = False
use_gpu = False
path = f'trained_model_loss{loss_type}.pth'
if is_train:
    # train model
    if use_gpu:
        model = model.cuda()
        criterion = criterion.cuda()
        times = times.cuda()
        events = times.cuda()
        train_data = train_data.cuda()

    Epochs = 600
    loss_list = []
    for t in range(Epochs):
        risk = model(train_data)
        loss, tie_counts, cum_exp_risk, failure_time = criterion(times, events, risk)
        print(t, loss.item())
        loss_list.append((loss.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(model, path)
else:
    model = torch.load(path, map_location="cpu")
    risk = model(train_data)
    loss, tie_counts, cum_exp_risk, failure_time = criterion(times, events, risk)

# H0_t


cumulative_sum = np.cumsum(cum_exp_risk[::-1])[::-1]

if loss_type == 1:
    smoothed_H0t = UnivariateSpline(failure_time, tie_counts / cum_exp_risk, s=0.2)

if loss_type == 2:
    smoothed_H0t = UnivariateSpline(failure_time, tie_counts / cumulative_sum, s=0.095)

# x_range = np.linspace(np.min(failure_time), np.max(failure_time), 1000)
#
# plt.figure(figsize=(10, 6))
# plt.plot(x_range, smoothed_H0t(x_range), label='Smoothed Cumulative Hazard')
# plt.scatter(failure_time, tie_counts / cumulative_sum, color='red', alpha=0.5, label='Original Data Points')
#
# plt.xlabel('Time')
# plt.ylabel('Cumulative Hazard')
# plt.title('Smoothed Cumulative Hazard Function')
# plt.legend()
# plt.grid(True)
# plt.show()
#
# ys = smoothed_H0t(failure_time)
#
# # utils_simulation.Compare_hazard_plot(data, id=68, H0t=smoothed_H0t, model=model)
#
#
# # Prediction
#
# # [est_sz,true_sz]=utils_simulation.compare_pre_result(model, node_in, smoothed_H0t, min, max)
# # est_sz,true_sz=utils_simulation.compare_pre_result(model, node_in, smoothed_H0t, min, max)
#
# est_sz, est_sz_lower, est_sz_upper, mrl, survival_probs = utils_simulation.compare_pre_result(10,
#                                                                                               1,
#                                                                                               model,
#                                                                                               node_in,
#                                                                                               smoothed_H0t,
#                                                                                               min,
#                                                                                               max)
#
# import matplotlib.pyplot as plt
#
# time_steps = range(10, 10 + len(est_sz[0]))
#
# # plt.plot(time_steps, true_sz[0], label="true", color="orange")
# plt.plot(time_steps, est_sz[0], label="Mean Survival Estimate", color="blue")
# # plt.fill_between(time_steps, est_sz_lower[0], est_sz_upper[0], color="blue", alpha=0.2, label="95% Confidence Interval")
# plt.plot(time_steps, est_sz_lower[0], label="interval", color="black", linestyle="--")
# plt.plot(time_steps, est_sz_upper[0], label="interval", color="black", linestyle="--")
#
# plt.xlabel("Time Steps")
# plt.ylabel("Survival Estimate")
# plt.title("Survival Estimate with Confidence Interval")
# plt.legend()
# plt.show()
#
# print()




import os
# tstar_time = 20
num_train_data = 150
for tstar_time in [50]:
    directory_sp = f'survivals_probs_loss{loss_type}_t{tstar_time}_ndata{num_train_data}'

    if os.path.exists(directory_sp):
        raise FileExistsError(f"The directory '{directory_sp}' already exists.")
    else:
        os.makedirs(directory_sp)
        print(f"The directory '{directory_sp}' has been created.")


    directory_mrl = f'mrl_loss{loss_type}_t{tstar_time}_ndata{num_train_data}'

    if os.path.exists(directory_mrl):
        raise FileExistsError(f"The directory '{directory_mrl}' already exists.")
    else:
        os.makedirs(directory_mrl)
        print(f"The directory '{directory_mrl}' has been created.")

    for unit in range(1, 151):
        est_sz, est_sz_lower, est_sz_upper, mrl, survival_probs = utils_simulation.compare_pre_result(tstar_time,
                                                                                                      unit,
                                                                                                      model,
                                                                                                      node_in,
                                                                                                      smoothed_H0t,
                                                                                                      min,
                                                                                                      max)

        sp_file_name = f"sp_{unit}.npy"
        file_path = os.path.join(directory_sp, sp_file_name)
        np.save(file_path, survival_probs)
        print(f"Array saved as {sp_file_name} in {directory_sp}")


        mrl_file_name = f"mrl_{unit}.npy"
        file_path = os.path.join(directory_mrl, mrl_file_name)
        np.save(file_path, mrl)
        print(f"Array saved as {mrl_file_name} in {directory_mrl}")




# ra=50
# t=np.linspace(1,ra,ra)
# plt.plot(t,np.quantile(est_sz,0.025,axis=0)[0:ra])
# plt.plot(t,np.mean(est_sz,axis=0)[0:ra])
# plt.plot(t,np.quantile(est_sz,0.975,axis=0)[0:ra])
# plt.plot(t,true_sz[0][0:ra])
# plt.show()


