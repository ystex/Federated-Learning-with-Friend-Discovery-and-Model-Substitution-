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

#1/26/2022 This is for AI cifar10 datasets
#This is for Nclass data distibution
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
lam = 5225
for num in numbers:
    print(f'\n############################The Iteration {num} ############################ \n')
    if not os.path.exists('ai_cifar_%d' % lam):
        os.mkdir('ai_cifar_%d' % lam)
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
    # 1) Fedavg-a, This is for all clients for this iterations
    idxs_users = sorted(np.random.choice(range(args.num_users), m, replace=False))
    print('List of Clients', idxs_users)

    round_data = []
    training_loss_data = []
    test_acc_data = []
    for epoch in tqdm(range(args.epochs)):
        power = epoch // 10
        #lr_int = 0.1
        #lr = lr_int * math.pow(0.95, power)
        lr = 0.1
        local_weights, local_losses = [], []
        print(f'\n | Global Communication Round : {epoch+1} |\n')

        global_model.train()

        # (1) local update for 0 degree
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], logger=logger, lr=lr)
            w, loss = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # calculate global weights
        global_weights = average_weights(local_weights)
        #print('ccccccccccccccccccccccccccccccglobal_weights',  global_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)

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
        df.to_csv('ai_cifar_%d/iteration_%d.csv' % (lam, num), sep=',', index=False)


