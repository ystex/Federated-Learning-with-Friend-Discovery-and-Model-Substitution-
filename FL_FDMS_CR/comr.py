import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import re
import glob
import copy

m = 20
delta = 0.01
B = 1
T = 500
f_max = 3
beta = 0.5
friend_cand_list_int = [[1] * m] * m
#print(n)
eta_all = []

def threshold_all(T):
    n = np.linspace(1, T, T)
    for each in n:
        alpha = (2 * np.log( 2 * m * m * T * f_max ) - 2 * np.log(delta))
        alpha_1 = alpha / (2 * beta * each)
        eta = 0.5 * np.sqrt(alpha_1) + 0.01
    #print('000000000000000000000', eta)
        eta_all.append(eta)
    return eta_all

#print(eta_all)

def friend_cal(M_ij, threshold_ite):
    friend_cand_list = []
    for u in range(20):
        clinet_i = M_ij[u]
        max_per = sorted(set( clinet_i))[-2]
        scope_gap = [max_per - x for x in clinet_i]
        #print('ppppppppppppppppppppppp',  clinet_i)
        #threshold_test = 0.3
        elimate_index = [i for i, v in enumerate(scope_gap) if v > threshold_ite]
        #print('eeeeeeeeeeeeeeeeeeeeeee', elimate_index)
        friend_cand = copy.deepcopy(friend_cand_list_int[u])
        for indexx in range(len(elimate_index)):
            friend_cand[elimate_index[indexx]] = 0
            #print('uuuuuuuuuuuuuuuuuuuu', friend_cand)
        friend_cand_list.append(copy.deepcopy(friend_cand))
    #print('xxxxxxxxxxxxxxxxxxxxxx', friend_cand_list)
    return friend_cand_list

def communication_r_ij(f_i_j, client_state, active_client_all):
    every_round_per_client = copy.deepcopy(0)
    client_state_all = [item for sublist in client_state for item in sublist]
    for idx in active_client_all:
        client_communication_times = sum([a * b for a, b in zip(f_i_j[idx], client_state_all)]) - 1
        every_round_per_client += client_communication_times
    print('ppppppppppppppppppppppp', f_i_j)
    print('xxxxxxxxxxxxxxxxxx', client_state_all)
    print('yyyyyyyyyyyy', active_client_all)
    com_round = 1
    return every_round_per_client