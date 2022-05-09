#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import numpy as np
from torchvision import datasets, transforms

def get_indices(dataset,class_name):
    indices =  []
    for i in range(len(dataset.targets)):
        if dataset.targets[i] == class_name:
            indices.append(copy.deepcopy(i))
    return indices

#1/23/2022 clear+

########################Yes
def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

########################Yes
def cifar_noniid(dataset, num_users):

    num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    # labels = dataset.train_labels.numpy()
    #labels = np.array(dataset.train_labels)
    labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def cifar_extr_noniid(train_dataset, num_users):
    #num_shards_train total number of shards
    n_class = 2
    num_samples = 500
    num_classes = 10
    id_all_class = []
    for i in range(num_classes):
        idx = get_indices(train_dataset, i)
        id_all_class.append(copy.deepcopy(idx))
    idxs = np.arange(5000)
    #print('1111111111111111111', id_all_class)
    # labels = dataset.train_labels.numpy()
    labels = np.array(train_dataset.targets)
    #idxs_labels = np.vstack((idxs, labels))
    # sort the image by label, the first one is index the second one is label
    #idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    dict_users_train = {i: np.array([]) for i in range(num_users)}
    #print('1111111111111111111', idxs_labels)
    #idxs = idxs_labels[0, :]
    #print('22222222222222', len(idxs))
    #labels = idxs_labels[1, :]
    idx_class = [i for i in range(num_classes)]
    #print('33333333333333333', labels)

    class_list = []
    sample_class = int(num_classes / n_class)
    for i in range(sample_class):
        rand_class = np.random.choice(idx_class, n_class, replace=False)
        for j in range(int(num_users / sample_class)):
            class_list.append(copy.deepcopy(rand_class))
        rand_class_set = set(rand_class)
        idx_class = list(set(idx_class) - rand_class_set)
    #print('xxxxxxxxxxxxxxxxxxxx', class_list)
    # divide and assign

    for i in range(num_users):
        user_labels = np.array([])
        #rand_set randomly pick two shard
        for rand in class_list[i]:
            # connect itself to make longer
            current_label = id_all_class[rand]
            #print('yyyyyyyyyyyyyyyyyyyyyyyy', len(current_label))
            #print('yyyyyyyyyyyyyyyyyyyyyyyy', type(current_label))
            choose_example = np.random.choice(current_label, num_samples, replace=False)
            id_all_class[rand] = [x for x in id_all_class[rand] if x not in choose_example.tolist()]
            dict_users_train[i] = np.concatenate((dict_users_train[i], choose_example), axis=0)
            user_labels = np.concatenate((user_labels, np.full((num_samples), rand)), axis=0)

        #print('yyyyyyyyyyyyyyyyyyyyyyyy', len(dict_users_train))
        #user_labels_set = set(user_labels)
        #print('zzzzzzzzzzzzzzzzzzzzzzz', type(dict_users_train))
    return dict_users_train

def mnist_extr_noniid_freedom(train_dataset, test_dataset, num_users):
    n_class = 3
    num_samples = 500
    num_classes = 10
    num_shards_train, num_imgs_train = int(50000/num_samples), num_samples
    num_imgs_perc_test, num_imgs_test_total = 1000, 10000
    assert(n_class * num_users <= num_shards_train)
    assert(n_class <= num_classes)
    idx_class = [i for i in range(num_classes)]
    idx_shard = [i for i in range(num_shards_train)]
    dict_users_train = {i: np.array([]) for i in range(num_users)}
    dict_users_test = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards_train*num_imgs_train)
    # labels = dataset.train_labels.numpy()
    labels = np.array(train_dataset.targets)
    idxs_test = np.arange(num_imgs_test_total)
    labels_test = np.array(test_dataset.targets)
    #labels_test_raw = np.array(test_dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    labels = idxs_labels[1, :]

    idxs_labels_test = np.vstack((idxs_test, labels_test))
    idxs_labels_test = idxs_labels_test[:, idxs_labels_test[1, :].argsort()]
    idxs_test = idxs_labels_test[0, :]
    #print(idxs_labels_test[1, :])

    # divide and assign
    for i in range(num_users):
        user_labels = np.array([])
        rand_set = set(np.random.choice(idx_shard, n_class, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        unbalance_flag = 0
        for rand in rand_set:
            if unbalance_flag == 0:
                dict_users_train[i] = np.concatenate(
                    (dict_users_train[i], idxs[rand*num_imgs_train:(rand+1)*num_imgs_train]), axis=0)
                user_labels = np.concatenate((user_labels, labels[rand*num_imgs_train:(rand+1)*num_imgs_train]), axis=0)

        user_labels_set = set(user_labels)
        #print(user_labels_set)
        #print(user_labels)
        for label in user_labels_set:
            dict_users_test[i] = np.concatenate((dict_users_test[i], idxs_test[int(label)*num_imgs_perc_test:int(label+1)*num_imgs_perc_test]), axis=0)
        #print(set(labels_test_raw[dict_users_test[i].astype(int)]))
    return dict_users_train, dict_users_test
