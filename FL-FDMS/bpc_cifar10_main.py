import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import pandas as pd
import math

import torch
from tensorboardX import SummaryWriter

from options import args_parser
import random
from update import LocalUpdate, test_inference
from models import CNNCifar, CNNCifar_v2
from utils import get_dataset, average_weights, exp_details
from torch.utils.data import ConcatDataset
from bpc_per import Client_State_Matrix, cosine_similarity_matrix, CSM_Update, communication_bet, CSM_Execution

# This is for BPC cifar10 datasets

def randNums(n,a,b,s):
    #finds n random ints in [a,b] with sum of s
    hit = False
    while not hit:
        total, count = 0, 0
        nums = []
        while total < s and count < n:
            r = random.randint(a, b)
            total += r
            count += 1
            nums.append(r)
        if total == s and count == n: hit = True
    return nums

numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
lam = 1
for num in numbers:
    print(f'\n############################The Iteration {num} ############################ \n')
    if not os.path.exists('cifar_bpcp_%d' % lam):
        os.mkdir('cifar_bpcp_%d' % lam)
        # define paths
    path_project = os.path.abspath('../../../..')
    logger = SummaryWriter('../logs')
    np.random.seed(num)
    args = args_parser()
    # print the detail of the parameters
    exp_details(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load dataset and user groups train_dataset = 60000  user_groups = 200, each is a client with 300 point
    train_dataset, test_dataset, user_groups = get_dataset(args)
    #print('zzzzzzzzzzzzzzzzzzzzz',  len(user_groups))

    # BUILD MODEL base on the structure
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'cifar':
            global_model = CNNCifar(args=args)

    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    #print('A1 global model:', global_model)

    # copy weights
    global_weights = global_model.state_dict()
    #print('A2 global_weights:', global_weights)

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 1
    val_loss_pre, counter = 0, 0

    # The following two select the clinet indxs with total m
    m = max(int(args.frac * args.num_users), 1)
    print('tttttttttttttttttttttt', m)
    # 1) Fedavg-a, This is for all clients for this iterations
    idxs_users = sorted(np.random.choice(range(args.num_users), m, replace=False))
    print('List of Clients', idxs_users)

    communication_matrix = []
    round_data = []
    training_loss_data = []
    test_acc_data = []
    M_ij_all = []
    global_model_all = []
    for epoch in tqdm(range(args.epochs)):
        power = epoch // 10
        #lr_int = 0.1
        #lr = lr_int * math.pow(0.95, power)
        lr = 0.1
        local_weights, local_losses = [], []
        print(f'\n | Global Communication Round : {epoch+1} |\n')
        dropout_rate = 0.5
        global_model.train()
        if epoch > 0:
            num_non_dropout = int((1 - dropout_rate) * m)
            client_schedule_list = randNums(5, 1, 4, num_non_dropout)
            print('client-sechduel-per-distribution', client_schedule_list)
            per_data_distribution = [idxs_users[x:x + 4] for x in range(0, len(idxs_users), 4)]
            #print('per_data_distribution', per_data_distribution)
            active_client = []
            for i in range(len(per_data_distribution)):
                active_client_per = sorted(np.random.choice(per_data_distribution[i], client_schedule_list[i], replace=False))
                active_client.append(copy.deepcopy(active_client_per))

            #aaaabreak the []
            active_client_all = [item for sublist in active_client for item in sublist]
            print('active_client', active_client_all)
        else:
            active_client_all = idxs_users
            # aaaaadd the []
            active_client = [active_client_all[x:x + 4] for x in range(0, len(active_client_all), 4)]

        idxs_users_p = [idxs_users[x:x + 4] for x in range(0, len(idxs_users), 4)]
        #print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX', idxs_users_p)

        # get the client state at each round
        client_state = Client_State_Matrix(idxs_users_p, active_client)
        #print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX', client_state)
        # (1) local update for 0 degree
        global_model_all.append(copy.deepcopy(global_model))
        for idx in active_client_all:
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], logger=logger, lr=lr)
            w, loss = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

            # the new global weight is calculated by adjusted local weight (10 + 10)
        if epoch > 0:
            M_ij_pre = M_ij_all[-1]
        # calculate global weights
            #print('YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY', M_ij_pre)
            new_local_weight, new_local_losses = CSM_Execution(local_weights, local_losses, client_state, M_ij_pre)
        else:
            new_local_weight = local_weights
            new_local_losses = local_losses


        # calculate global weights
        global_weights = average_weights(new_local_weight)

        # XXXXXXXXXXXXXFunction get the cosine similarity at this round R_ij
        begin_weight = global_model_all[-1].state_dict()
        R_ij = cosine_similarity_matrix(local_weights, begin_weight, client_state)
        #print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX', R_ij)
        if epoch > 0:
            # This is com(T-1)
            T_e_pre = communication_matrix[-1]
            # update of com(T)
            T_e = communication_bet(client_state, T_e_pre)
            communication_matrix.append(copy.deepcopy(T_e))
            M_ij = CSM_Update(M_ij_pre, R_ij, T_e_pre, client_state)
            M_ij_all.append(copy.deepcopy(M_ij))

        else:
            T_e = [[1] * m] * m
            communication_matrix.append(copy.deepcopy(T_e))
            M_ij_all.append(copy.deepcopy(R_ij))


        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(new_local_losses) / len(new_local_weight)


        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()

        # Test inference after completion of training
        test_acc, test_loss = test_inference(args, global_model, test_dataset)

        round_data.append(epoch)
        training_loss_data.append(loss_avg)
        test_acc_data.append(test_acc)
        print(f' \n Results after {epoch + 1} global rounds of training:')
        print('|-------------------------------- Training Loss', loss_avg)
        print('|-------------------------------- Test Accuracy', test_acc)

        df = pd.DataFrame({'Round': round_data, 'Loss': training_loss_data, 'Accuracy': test_acc_data})
        # cs = pd.DataFrame(M_ij_client_1)
        df.to_csv('cifar_bpcp_%d/iteration_%d.csv' % (lam, num), sep=',', index=False)

    #M_final_get = M_ij_all[-1]
    #print('cccccccccccccccccccccccccccccc', M_final_get)
    #mf = pd.DataFrame(M_final_get)
    #mf.to_csv('55cifar_bpcp_%d/CS_%d.csv' % (lam, num), sep=',', index=False)