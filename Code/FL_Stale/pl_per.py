import numpy as np
import torch
import math
import copy
import torch
from options import args_parser
from torch import nn
import torch.nn.functional as F
import operator

args = args_parser()

def Client_State_Matrix(all_client, active_client):
    client_states = []
    for i in range(0, len(all_client)):
        empty_list = [0] * len(all_client[0])
        index_i = np.where(np.in1d(np.array(all_client[i]), np.array(active_client[i])))[0]
        index_i.tolist()
        for index in index_i:
            empty_list[index] = 1
        client_states.append(copy.deepcopy(empty_list))
    print('clien-status--------------', client_states)
    return client_states

def find(lst, a):
    return [i for i, x in enumerate(lst) if x== a]

def PL_Update(client_state, previous_weight, local_weights, previous_loss, local_losses):
    client_state_all = [item for sublist in client_state for item in sublist]
    active_list = find(client_state_all, 1)
    print('ooooooooooooooooooooo', active_list)
    new_local_loss = []
    new_local_weight = []
    ir = 0
    for i in range(0, len(client_state_all)):
        if client_state_all[i] == 0:
            client_weight_pre = previous_weight[i]
            client_loss_pre = previous_loss[i]
            new_local_weight.append(copy.deepcopy(client_weight_pre))
            new_local_loss.append(copy.deepcopy(client_loss_pre))
        else:
            client_weight = local_weights[ir]
            client_loss = local_losses[ir]
            ir = ir + 1
            new_local_weight.append(copy.deepcopy(client_weight))
            new_local_loss.append(copy.deepcopy(client_loss))

    return new_local_weight, new_local_loss