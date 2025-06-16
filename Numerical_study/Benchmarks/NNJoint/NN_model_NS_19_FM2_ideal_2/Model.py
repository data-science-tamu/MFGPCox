import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 
class CoxNet(nn.Module):
    def __init__(self, input_dim, node_hidden1, node_hidden2, out_dim):
        super(CoxNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, node_hidden1),
            nn.Sigmoid(),
            nn.Linear(node_hidden1, node_hidden2),
            nn.Sigmoid(),
            nn.Linear(node_hidden2, out_dim)
        )

    def forward(self, x):
        x = self.model(x)
        return x



class Efron_loss(nn.Module):
    def __init__(self):
        super(Efron_loss, self).__init__()

    def forward(self, times, events, risk):
        loss = torch.tensor([0])
        exp_risk = torch.exp(risk)
        T = torch.unique(times)
        tie_count_list = []
        cum_exp_risk = []
        failure_time = []

        for t in range(len(T)):
            index = times == T[t]
            tie_index = (times == T[t]) & (events == 1)
            tie_counts = len(times[tie_index])
            temp_loss = torch.tensor([0])
            tie_risk = risk[tie_index]
            tie_exp_risk = torch.sum(exp_risk[tie_index])
            if tie_counts > 0:
                failure_time.append(T[t].cpu().detach().numpy())
                tie_count_list.append(tie_counts)
                cum_exp_risk.append(torch.sum(exp_risk[index]).cpu().detach().numpy())

            for j in range(tie_counts):
                temp = torch.sub(tie_risk[j],
                                 torch.log(torch.sub(torch.sum(exp_risk[index]), j / tie_counts * tie_exp_risk)))
                temp_loss = torch.sub(temp_loss.to(device='cuda'), temp.to(device='cuda'))

            loss = torch.add(temp_loss.to(device='cuda'), loss.to(device='cuda'))

        return loss, np.array(tie_count_list), np.array(cum_exp_risk), failure_time


class Efron_loss_penalty(nn.Module):
    def __init__(self, penalty):
        super(Efron_loss_penalty, self).__init__()
        self.penalty = penalty

    def forward(self, times, events, risk):
        loss = torch.tensor([0])
        exp_risk = torch.exp(risk)
        T = torch.unique(times)
        tie_count_list = []
        cum_exp_risk = []
        failure_time = []

        for t in range(len(T)):
            index = times == T[t]
            tie_index = (times == T[t]) & (events == 1)
            tie_counts = len(times[tie_index])
            temp_loss = torch.tensor([0])
            tie_risk = risk[tie_index]
            tie_exp_risk = torch.sum(exp_risk[tie_index])
            if tie_counts > 0:
                temp_loss = torch.add(temp_loss.to(device="cuda"), self.penalty * torch.sum(torch.abs(risk[index])))

                failure_time.append(T[t].detach().cpu().numpy())
                tie_count_list.append(tie_counts)
                cum_exp_risk.append(torch.sum(exp_risk[index]).cpu().detach().numpy())

            for j in range(tie_counts):
                temp = torch.sub(tie_risk[j],
                                 torch.log(torch.sub(torch.sum(exp_risk[index]), j / tie_counts * tie_exp_risk)))
                temp_loss = torch.sub(temp_loss.to(device="cuda"), temp.to(device="cuda"))

            loss = torch.add(temp_loss.to(device="cuda"), loss.to(device="cuda"))

        return loss, np.array(tie_count_list), np.array(cum_exp_risk), np.array(failure_time)



class Efron_loss_penalty_no_exp(nn.Module):
    def __init__(self, penalty):
        super(Efron_loss_penalty_no_exp, self).__init__()
        self.penalty = penalty

    def forward(self, times, events, risk):
        loss = torch.tensor([0])
        exp_risk = risk
        T = torch.unique(times)
        tie_count_list = []
        cum_exp_risk = []
        failure_time = []

        for t in range(len(T)):
            index = times == T[t]
            tie_index = (times == T[t]) & (events == 1)
            tie_counts = len(times[tie_index])
            temp_loss = torch.tensor([0])
            tie_risk = risk[tie_index]
            tie_exp_risk = torch.sum(exp_risk[tie_index])
            if tie_counts > 0:
                temp_loss = torch.add(temp_loss, self.penalty * torch.sum(torch.abs(risk[index])))

            failure_time.append(T[t].detach().numpy())
            tie_count_list.append(tie_counts)
            cum_exp_risk.append(torch.sum(exp_risk[index]).detach().numpy())

            for j in range(tie_counts):
                temp = torch.sub(tie_risk[j],
                                 torch.log(torch.sub(torch.sum(exp_risk[index]), j / tie_counts * tie_exp_risk)))
                temp_loss = torch.sub(temp_loss, temp)

            loss = torch.add(temp_loss, loss)

        return loss, np.array(tie_count_list), np.array(cum_exp_risk), np.array(failure_time)







