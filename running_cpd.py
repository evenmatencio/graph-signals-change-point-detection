import time

import numpy as np
import networkx as nx
import numba_cost_functions as my_numb
import result_related as my_res
import utils as my_ut

from typing import List


def run_numba_statio_normal_cost_and_store_res(G: nx.Graph, signal: np.ndarray, gt_bkps: List[int], statio_json_path: str, exp_id: int):
    # running CPD algorithm
    t1 = time.perf_counter()
    graph_lapl_mat = nx.laplacian_matrix(G).toarray().astype(np.float64)
    ###############################################################
    # graph_lapl_mat = np.eye(signal.shape[1])
    ###############################################################
    gft_square_cumsum = my_numb.init_station_normal_cost(signal, graph_lapl_mat)
    statio_bkps = my_numb.numba_cpd_dynprog_statio_cost_2_optim(len(gt_bkps)-1, signal.shape[1], gft_square_cumsum)
    statio_bkps = [int(bkp) for bkp in statio_bkps]
    t2 = time.perf_counter()
    # logging
    res = my_res.update_pred_dic_with_one_exp(t1, t2, statio_bkps, gt_bkps, exp_id)
    my_ut.load_and_write_json(statio_json_path, exp_id, my_ut.turn_all_list_of_dict_into_str(res), indent=4)

def run_numba_standard_mle_normal_cost_and_store_res(signal: np.ndarray, gt_bkps: List[int], normal_json_path: str, exp_id:int):
    # running CPD algorithm
    t1 = time.perf_counter()
    normal_bkps = my_numb.numba_cpd_dynprog_mle_standard_cost_2_optim(len(gt_bkps) - 1, signal.shape[1], signal)
    normal_bkps = [int(bkp) for bkp in normal_bkps]
    t2 = time.perf_counter()
    # logging
    res = my_res.update_pred_dic_with_one_exp(t1, t2, normal_bkps, gt_bkps, exp_id)
    my_ut.load_and_write_json(normal_json_path, exp_id, my_ut.turn_all_list_of_dict_into_str(res), indent=4)

