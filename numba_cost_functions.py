import numpy as np

from numba import njit
from math import floor
from scipy.linalg import eigh
from numpy.linalg import slogdet


### STATIONARY NORMAL COST
#####################################

def init_station_normal_cost(signal, graph_laplacian_mat):
    '''signal (array): of shape [n_samples, n_dim]'''
    # computation of the graph fourier transform
    _, eigvects = eigh(graph_laplacian_mat)
    gft =  signal @ eigvects # equals signal.dot(eigvects) = eigvects.T.dot(signal.T).T
    gft_mean = np.mean(gft, axis=0)
    # computation of the per-segment cost utils
    gft_square_cumsum = np.concatenate([np.zeros((1, signal.shape[1])), np.cumsum((gft - gft_mean[None, :])**2, axis=0)], axis=0)
    return gft_square_cumsum.astype(np.float64)

@njit
def numba_statio_cost_func(start, end, gft_square_cumsum):
    '''
    Computes the cost over signal[start:end, :] where end is excluded
    gft_square_cumsum (array): of shape [n_samples + 1, n_dim] 
    '''
    sub_square_sum = gft_square_cumsum[end, :] - gft_square_cumsum[start, :]
    return np.float64(end  - start) * np.sum(np.log(sub_square_sum / (end - start)), dtype=np.float64)

@njit
def numba_cpd_dynprog_statio_cost(n_bkps:int, min_size:int, data: np.ndarray):
    n_samples = data.shape[0]
    # if no bkp to find
    if n_bkps == 0:
        return np.array([1000], dtype=np.int64)
    # full partitions costs
    full_part_cost = np.inf * np.ones((n_bkps, n_samples, n_samples), dtype=np.float64)
    # compute the segment cost with no bpk, for admissible segment only
    for start in range(0, n_samples-min_size):
        # until n_samples + 1 because the call to cost_function(start, end, data) computes over [y_0, ... y_{n-1}] (remember data.shape[0] = n_samples + 1)
        for end in range(start+min_size, n_samples):  
            full_part_cost[0, start, end] = numba_statio_cost_func(start, end, data)
    # compute the cost of the possible higher order partitions 
    for bkp_order in range(1, n_bkps):
        min_multi_seg_length = (bkp_order + 1) * min_size
        for start in range(0, n_samples-min_multi_seg_length):
            for end in range(start + min_multi_seg_length, n_samples):
                min_size_left_seg = min_multi_seg_length - min_size
                full_part_cost[bkp_order, start, end] = np.min(full_part_cost[bkp_order-1, start, start+min_size_left_seg:end-min_size+1] + full_part_cost[0, start+min_size_left_seg:end-min_size+1, end])
    # successively pick the bkps from the right-end of the whole signal
    bkps = np.int64(n_samples-1) * np.ones(n_bkps+1, dtype=np.int64)
    for bkp_id in range(n_bkps, 0, -1):
        min_multi_seg_length = np.int64(bkp_id * min_size) 
        bkp_right = bkps[bkp_id]
        bkp_left = min_multi_seg_length + np.argmin(full_part_cost[bkp_id-1, 0, min_multi_seg_length:bkp_right-min_size+1] + full_part_cost[0, min_multi_seg_length:bkp_right-min_size+1, bkp_right])
        bkps[bkp_id-1] = bkp_left
    return bkps

@njit
def numba_cpd_dynprog_statio_cost_2_optim(n_bkps:int, min_size:int, data: np.ndarray):
    # path_mat[n, K] avec n --> [y_0, ... y_{n-1}], K : n_bkps

    # initialization 
    n_samples = data.shape[0] - 1
    path_mat = np.empty((n_samples+1, n_bkps+1), dtype=np.int32)
    path_mat[:, 0] = 0
    path_mat[0, :] = -1
    sum_of_cost_mat = np.full((n_samples+1, n_bkps+1),  fill_value=np.inf, dtype=np.float64)
    sum_of_cost_mat[0, :] = 0

    # forward computation
    for end in range(min_size, n_samples+1):
        sum_of_cost_mat[end, 0] = numba_statio_cost_func(0, end, data)
        # consistent because our cost functions compute the costs over [start, end[
        max_admissible_n_bkp = floor(end/min_size) - 1
        for k_bkps in range(1, min(max_admissible_n_bkp+1, n_bkps+1)):
            soc_optim = np.inf
            soc_argmin = -1
            for mid in range(min_size*k_bkps, end - min_size + 1): 
                soc = sum_of_cost_mat[mid, k_bkps-1] + numba_statio_cost_func(mid, end, data)
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



### STANDARD NORMAL COST
#####################################

@njit
def standard_normal_cost_func(start, end, signal):
    '''signal (array): of shape [n_samples, n_dim]'''
    sub = signal[start:end, :]
    cov = np.cov(sub.T)
    cov += 1e-6 * np.eye(signal.shape[1])
    _, val = slogdet(cov)
    return np.float64(val * (end - start))

@njit
def numba_cpd_dynprog_mle_standard_cost(n_bkps:int, min_size:int, signal: np.ndarray):
    n_samples = signal.shape[0]
    # if no bkp to find
    if n_bkps == 0:
        return np.array([1000], dtype=np.int64)
    # full partitions costs
    full_part_cost = np.inf * np.ones((n_bkps, n_samples, n_samples), dtype=np.float64)
    # compute the segment cost with no bpk, for admissible segment only
    for start in range(0, n_samples-min_size+1):
        # until n_samples + 1 because the call to cost_function(start, end, data) computes over [y_0, ... y_{n-1}] 
        for end in range(start+min_size, n_samples+1):  
            full_part_cost[0, start, end] = standard_normal_cost_func(start, end, signal)
    # compute the cost of the possible higher order partitions 
    for bkp_order in range(1, n_bkps):
        min_multi_seg_length = (bkp_order + 1) * min_size
        for start in range(0, n_samples-min_multi_seg_length):
            for end in range(start + min_multi_seg_length, n_samples):
                min_size_left_seg = min_multi_seg_length - min_size
                full_part_cost[bkp_order, start, end] = np.min(full_part_cost[bkp_order-1, start, start+min_size_left_seg:end-min_size+1] + full_part_cost[0, start+min_size_left_seg:end-min_size+1, end])
    # successively pick the bkps from the right-end of the whole signal
    bkps = np.int64(n_samples) * np.ones(n_bkps+1, dtype=np.int64)
    for bkp_id in range(n_bkps, 0, -1):
        min_multi_seg_length = np.int64(bkp_id * min_size) 
        bkp_right = bkps[bkp_id]
        bkp_left = min_multi_seg_length + np.argmin(full_part_cost[bkp_id-1, 0, min_multi_seg_length:bkp_right-min_size+1] + full_part_cost[0, min_multi_seg_length:bkp_right-min_size+1, bkp_right])
        bkps[bkp_id-1] = bkp_left
    return bkps

@njit
def numba_cpd_dynprog_mle_standard_cost_2_optim(n_bkps:int, min_size:int, signal):
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
    for start in range(0, n_samples-min_size+1):
        for end in range(start+min_size, n_samples+1):  
            statio_segment_cost[start, end] = standard_normal_cost_func(start, end, signal)

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