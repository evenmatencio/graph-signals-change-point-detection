import time
import os

import numpy as np
import networkx as nx
import numba_cost_functions as my_numb
import result_related as my_res
import utils as my_ut
import rpy2_related as my_rpy2

from typing import List, Literal
from math import floor
from tqdm import tqdm

cpd_methods_name = Literal["statio", "standard_mle", "glasso", "r_covcp"]

### UTILS 
#########

def init_pred_saving(pred_dir, name):
    results = {"0": "INIT"}
    my_ut.create_parent_and_dump_json(pred_dir, name, results, indent=4)
    return os.path.join(pred_dir, name)


### CPD DYNAMIC PROGRAMMING SOLVER
##################################

def rglasso_cpd_dynprog(n_bkps:int, min_size:int, signal, pen_mult_coef, buffer_path):
    # path_mat[n, K] avec n --> [y_0, ... y_{n-1}] (very important to understand indexing) , K : n_bkps
    # sum_of_cost_mat[n, K]: best cost for signal until sample n with K bkps

    # initialization 
    n_samples = signal.shape[0]
    path_mat = np.empty((n_samples+1, n_bkps+1), dtype=np.int32)
    path_mat[:, 0] = 0
    path_mat[0, :] = -1
    sum_of_cost_mat = np.full((n_samples+1, n_bkps+1),  fill_value=np.inf, dtype=np.float64)
    sum_of_cost_mat[0, :] = 0

    # pre-computation, to optimize jit processing
    statio_segment_cost = np.full((n_samples+1, n_samples+1), fill_value=np.inf, dtype=np.float64)
    for start in tqdm(range(0, n_samples-min_size+1), desc='Looping over the segments start in glasso'):
        for end in range(start+min_size, n_samples+1):
            statio_segment_cost[start, end] = my_rpy2.glasso_cost_func(start, end, signal, pen_mult_coef, buffer_path=buffer_path)

    # forward computation
    for end in range(min_size, n_samples+1):
        sum_of_cost_mat[end, 0] = statio_segment_cost[0, end]
        # consistent because our cost functions compute the costs over [start, end[
        max_admissible_n_bkp = floor(end/min_size) - 1
        for k_bkps in range(1, min(max_admissible_n_bkp+1, n_bkps+1)):
            soc_optim = np.inf
            soc_argmin = -1
            for mid in range(min_size*k_bkps, end - min_size + 1):
                soc = sum_of_cost_mat[mid, k_bkps-1] + statio_segment_cost[mid, end]
                if soc < soc_optim:
                    soc_argmin = mid
                    soc_optim = soc
            sum_of_cost_mat[end, k_bkps] = soc_optim
            path_mat[end, k_bkps] = soc_argmin

    # backtracking
    bkps = np.full((n_bkps+1), fill_value=n_samples)
    for k_bkps in range(n_bkps, 0, -1):
        bkps[k_bkps-1] = path_mat[bkps[k_bkps], k_bkps]
    
    return bkps


### CPD TRIGGER
###############

def run_numba_statio_normal_cost_and_store_res(G: nx.Graph, signal: np.ndarray, gt_bkps: List[int], min_size: int, statio_json_path: str, exp_id: int):
    # running CPD algorithm
    t1 = time.perf_counter()
    graph_lapl_mat = nx.laplacian_matrix(G).toarray().astype(np.float64)
    ###############################################################
    # graph_lapl_mat = np.eye(signal.shape[1])
    ###############################################################
    gft_square_cumsum = my_numb.init_station_normal_cost(signal, graph_lapl_mat)
    statio_bkps = my_numb.numba_cpd_dynprog_statio_cost_2_optim(len(gt_bkps)-1, min_size, gft_square_cumsum)
    statio_bkps = [int(bkp) for bkp in statio_bkps]
    t2 = time.perf_counter()
    # logging
    res = my_res.update_pred_dic_with_one_exp(t1, t2, statio_bkps, gt_bkps, exp_id)
    my_ut.load_and_write_json(statio_json_path, exp_id, my_ut.turn_all_list_of_dict_into_str(res), indent=4)

def run_numba_standard_mle_normal_cost_and_store_res(signal: np.ndarray, gt_bkps: List[int], min_size:int, normal_json_path: str, exp_id:int):
    # running CPD algorithm
    t1 = time.perf_counter()
    normal_bkps = my_numb.numba_cpd_dynprog_mle_standard_cost_2_optim(len(gt_bkps) - 1, min_size, signal)
    normal_bkps = [int(bkp) for bkp in normal_bkps]
    t2 = time.perf_counter()
    # logging
    res = my_res.update_pred_dic_with_one_exp(t1, t2, normal_bkps, gt_bkps, exp_id)
    my_ut.load_and_write_json(normal_json_path, exp_id, my_ut.turn_all_list_of_dict_into_str(res), indent=4)

def run_r_glasso_cpd_algo_and_store(signal, gt_bkps: List[int], glasso_json_path: str, exp_id: str, pen_mult_coef: float, buffer_path: str):
    # running CPD algorithm
    t1 = time.perf_counter()
    glasso_bkps = rglasso_cpd_dynprog(n_bkps=len(gt_bkps)-1, min_size=signal.shape[1], signal=signal, pen_mult_coef=pen_mult_coef, buffer_path=buffer_path)
    t2 = time.perf_counter()
    glasso_bkps.sort()
    glasso_bkps = [int(bkp) for bkp in glasso_bkps]
    # logging
    res = my_res.update_pred_dic_with_one_exp(t1, t2, glasso_bkps, gt_bkps, exp_id)
    my_ut.load_and_write_json(glasso_json_path, exp_id, my_ut.turn_all_list_of_dict_into_str(res), indent=4)

def run_r_covcp_cpd_algo_and_store(signal_path, gt_bkps: List[int], covcp_json_path: dict, stable_set_length:int, min_seg_length:int, window_sizes, alpha, exp_id, buffer_path, nb_cores, r_covcp_seed):
    # running CPD algorithm
    t1 = time.perf_counter()
    print('Running R covcp')
    my_rpy2.init_r_core_management(nb_cores, r_covcp_seed)
    r_signal = my_rpy2.load_r_signal(f"{signal_path}/{exp_id}_signal.npy")
    covcp_bkps = my_rpy2.detect_multiple_covcp_bkps(n_bkps=len(gt_bkps)-1, signal=r_signal, stable_set_length=stable_set_length, min_seg_length=min_seg_length, window_sizes=window_sizes, alpha=alpha, bkps=[], buffer_path=buffer_path)
    t2 = time.perf_counter()
    covcp_bkps.sort()
    covcp_bkps = [int(bkp) for bkp in covcp_bkps] + [gt_bkps[-1]]
    # logging
    res = my_res.update_pred_dic_with_one_exp(t1, t2, covcp_bkps, gt_bkps, exp_id)
    my_ut.load_and_write_json(covcp_json_path, exp_id, my_ut.turn_all_list_of_dict_into_str(res), indent=4)