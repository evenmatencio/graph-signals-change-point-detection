import numpy as np

from numba import njit
from math import floor
from numpy.linalg import slogdet


### STATIONARY NORMAL COST
#####################################


@njit
def numba_statio_cost_func(start, end, gft_square_cumsum):
    '''
    Computes the cost over signal[start:end, :] where end is excluded
    gft_square_cumsum (array): of shape [n_samples + 1, n_dim] 
    '''
    sub_square_sum = gft_square_cumsum[end, :] - gft_square_cumsum[start, :]
    return np.float64(end  - start) * np.sum(np.log(sub_square_sum / (end - start)), dtype=np.float64)


@njit
def numba_cpd_dynprog_statio_cost_optim(n_bkps:int, min_size:int, data: np.ndarray):
    # path_mat[n, K] avec n --> [y_0, ... y_{n-1}] (very important to understand indexing) , K : n_bkps
    # sum_of_cost_mat[n, K]: best cost for signal until sample n with K bkps
    # we fill-in the sum_of_cost_mat row by row, i.e for longer and longer signals (top to down), for higher and 
    # highr number of bkp (left to right), because in dynamic programming
    # we need the results of the previous sub-problems before solving the current one

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
            # the following step corresponds to the last line of eq (21) P.19 Truong, Oudre, Vayatis
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
def numba_cpd_dynprog_mle_standard_cost_optim(n_bkps:int, min_size:int, signal):
    # path_mat[n, K] avec n --> [y_0, ... y_{n-1}] (very important to understand indexing) , K : n_bkps
    # sum_of_cost_mat[n, K]: best cost for signal until sample n with K bkps
    # we fill-in the sum_of_cost_mat row by row, i.e for longer and longer signals (top to down), for higher and 
    # highr number of bkp (left to right), because in dynamic programming
    # we need the results of the previous sub-problems before solving the current one

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
            # the following step corresponds to the last line of eq (21) P.19 Truong, Oudre, Vayatis
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