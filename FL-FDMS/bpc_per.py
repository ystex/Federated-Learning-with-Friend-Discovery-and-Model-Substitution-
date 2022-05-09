import numpy as np
import torch
import math
import copy
import torch
from options import args_parser
from torch import nn
import torch.nn.functional as F
import operator
#1/20/2021 BPC algorithm per-round
#(A)The CSM Execution Module: which use the Mij to find the best partner for dp client and return the client selection matrix to next step
#(B)The CSM Update Module: which update the Mij matrix (Clear)

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

def cosine_similarity_matrix(weights, begin_weight, client_state):
    client_state_all = [item for sublist in client_state for item in sublist]
    weights_first = copy.deepcopy(weights[0])
    weight_all = []
    for i in range(0, len(weights)):
        weight_flattened_row = []
        for key in weights_first.keys():
            weight_direct = weights[i][key] - begin_weight[key]
            weight_flattened_row.append(copy.deepcopy(torch.flatten(weight_direct)))
        weight_all.append(copy.deepcopy(torch.cat(weight_flattened_row)))

    all_client_cs = []
    for n in range(0, len(weight_all)):
        per_client_cs = []
        #print('1111111111111111111111', n)
        for m in range(0, len(weight_all)):
            cs_bet_n_m = F.cosine_similarity(weight_all[n], weight_all[m], dim=-1) #Dim = -1 is larger, so just select
            cs_bet_nob = cs_bet_n_m.item()
            #change the rij represent
            cs_bet_ob = 0.5 * (cs_bet_nob + 1)
            # Dim = -1 is larger, so just select
            per_client_cs.append(copy.deepcopy(cs_bet_ob))
        all_client_cs.append(copy.deepcopy(per_client_cs))
    return all_client_cs

def communication_bet(client_state, T_e_pre):
    client_state_all = [item for sublist in client_state for item in sublist]
    com_all = []
    #print('xxxxxxxxxxxxxxxxxxxxxxxxxxxx', len(client_state_all))
    for n in range(0, len(client_state_all)):
        if client_state_all[n] == 0:
            com_per_client = [0] * len(client_state_all)
            com_all.append(copy.deepcopy(com_per_client))
        else:
            com_per_client = client_state_all[n] * client_state_all
            com_all.append(copy.deepcopy(com_per_client))
    #print('xxxxxxxxxxxxxxxxxxxxxxxxxxxx', com_all)
    T_e_all = []
    for m in range(0, len(client_state_all)):
        T_e_per = [a + b for a, b in zip(com_all[m], T_e_pre[m])]
        T_e_all.append(copy.deepcopy(T_e_per))
    #print('yyyyyyyyyyyyyyyyyyyyyyyyyyyy', T_e_all)
    return T_e_all

#all set
def CSM_Update(M_ij_pre, R_ij, T_e_pre, client_state):
    client_state_all = [item for sublist in client_state for item in sublist]
    #(1) M_ij_pre is the previous M matrix (2) R_ij is current cosine similarity (3) T_e is rge Com(T-1) (4)client_state is current client status.
    #print('xxxxx', M_ij_pre)
    #print('yyyyy', R_ij)
    #print('zzzzz', T_e_pre)
    #print('ooooo', client_state_all)
    M_ij_all = []
    ir = 0
    for i in range(0, len(T_e_pre)):
        jr = 0
        M_ij_client = []
        for j in range(0, len(T_e_pre[0])):
            Judg = client_state_all[i] * client_state_all[j]
            if Judg == 0:
                M_ij_per = M_ij_pre[i][j]
                M_ij_client.append(copy.deepcopy(M_ij_per))
                #print('xxxxxxxxxxxxxxxxxxx', i, j)
            else:
                M_ij_per = (1 / (T_e_pre[i][j] + 1)) * (T_e_pre[i][j] * M_ij_pre[i][j] + R_ij[ir][jr])
                jr = jr + 1
                #print('yyyyyyyyyyyyyyyyyy', i, j)
                M_ij_client.append(copy.deepcopy(M_ij_per))
            #jr = jr + 1
        M_ij_all.append(copy.deepcopy(M_ij_client))
        if client_state_all[i] == 0:
            ir = ir
        else:
            ir = ir + 1
    return M_ij_all


def find(lst, a):
    return [i for i, x in enumerate(lst) if x== a]

def CSM_Execution(local_weights, local_losses, client_state, M_ij_pre):
    client_state_all = [item for sublist in client_state for item in sublist]
    #print('ooooooooooooooooooooo', local_losses)
    active_list = find(client_state_all, 1)
    #print('ooooooooooooooooooooo', active_list)
    best_partner_all = []
    new_local_loss = []
    new_local_weight = []
    ir = 0
    for i in range(0, len(client_state_all)):
        if client_state_all[i] == 0:
            M_best_partner = M_ij_pre[i]
            M_list = [a * b for a, b in zip(M_best_partner, client_state_all)]
            #print('pppppppppppppppppppppppppp', M_list)
            index, value = max(enumerate(M_list), key=operator.itemgetter(1))
            best_partner = find(active_list, index)[0]
            best_partner_all.append(copy.deepcopy(active_list[best_partner]))
            client_weight = local_weights[best_partner]
            client_loss = local_losses[best_partner]
            new_local_weight.append(copy.deepcopy(client_weight))
            new_local_loss.append(copy.deepcopy(client_loss))
        else:
            client_weight = local_weights[ir]
            client_loss = local_losses[ir]
            best_partner_all.append(copy.deepcopy(100))
            ir = ir + 1
            new_local_weight.append(copy.deepcopy(client_weight))
            new_local_loss.append(copy.deepcopy(client_loss))
    print('uuuuuuuuuuuuuuuuuuuuuuuuuuu', best_partner_all)
    return new_local_weight, new_local_loss

