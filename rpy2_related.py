import numpy as np

import rpy2

from rpy2.robjects.packages import importr
from sklearn.covariance import log_likelihood



#### UTILS
###################

r_base = importr('base')
r_RcppCNPy = importr('RcppCNPy')
r_covcp = importr('covcp')
r_glasso = importr('glasso')
r_domc = importr('doMC')

def turn_r_bool_in_py_bool(r_bool):
    str_r_bool = str(r_bool.rx2(1))[4:9]
    return not str_r_bool == 'FALSE'

def get_nested_named_element_from_R_list(obj, named_elem:str):
    nested_names_list = named_elem.split('$')
    target_obj = obj
    for name in nested_names_list:
        if name.isdigit():
            target_obj = target_obj.rx2(int(name))
        else:
            target_obj = target_obj.rx2(name)
    return target_obj

def load_r_signal(signal_path):
    return r_RcppCNPy.npyLoad(signal_path)

def get_r_left_subsignal(signal: np.ndarray, split_id: int, temp_path: str):
    left_sub_signal_arr = np.copy(signal)[:split_id, :]
    np.save(f"{temp_path}_left_sub_signal.npy", left_sub_signal_arr)
    r_left_subsignal = r_RcppCNPy.npyLoad(f"{temp_path}_left_sub_signal.npy")
    return r_left_subsignal

def get_r_right_subsignal(signal: np.ndarray, split_id: int, temp_path: str):
    right_sub_signal_arr = np.copy(signal)[split_id:, :]
    np.save(f"{temp_path}_right_sub_signal.npy", right_sub_signal_arr)
    r_right_subsignal = r_RcppCNPy.npyLoad(f"{temp_path}_right_sub_signal.npy")
    return r_right_subsignal

def init_r_core_management(nb_cores, seed):
    r_domc.registerDoMC(cores = nb_cores)
    r_base.set_seed(seed)


### COST FUNCTIONS 
######################

def glasso_cost_func(start, end, signal, pen_mult_coef, temp_path):
    # extracting the target subsignal in R
    sub_signal = signal[start:end, :]
    np.save(f"{temp_path}.npy", sub_signal)
    r_subsignal = r_RcppCNPy.npyLoad(f"{temp_path}.npy")
    # applying the R implementation of Graphical Lasso
    raw_pen_coef = r_covcp.chooseRho(r_subsignal)
    pen_coef = r_base.c(pen_mult_coef * raw_pen_coef[0])
    r_emp_cov = r_covcp.Cov(r_subsignal)
    r_glasso_obj = r_glasso.glasso(s=r_emp_cov, rho=pen_coef, maxit=10)
    r_glasso_preci_mat = r_glasso_obj.rx2('wi')
    # computing the cost function
    glasso_preci_mat = np.asarray(r_glasso_preci_mat)
    emp_cov = np.asarray(r_emp_cov)
    cost_func_val = - log_likelihood(emp_cov, glasso_preci_mat)
    return cost_func_val

def get_covcp_localization(covcp_results, window_sizes):
    # check if there is actually a cp to localize
    is_there_cp = r_covcp.isRejected(covcp_results)
    if not is_there_cp:
        return None
    # if so, return the smallest window in which the cp can be localized
    ciritical_values = covcp_results.rx2('criticalValue')
    for i, window_size in enumerate(window_sizes):
        window_stat_results = get_nested_named_element_from_R_list(covcp_results, f'statistics$window2statistics${i+1}')
        max_stat = window_stat_results.rx2('statistics')[0]
        if max_stat > ciritical_values[i]:
            central_points = window_stat_results.rx2('centralPoints')
            stat_values_arr = np.asarray(window_stat_results.rx2('distances'))
            stats_argmx = int(np.argmax(stat_values_arr))
            central_point_argmax = central_points[stats_argmx]
            return central_point_argmax, (central_point_argmax - window_size, central_point_argmax + window_size)
        
def detect_multiple_covcp_bkps(n_bkps, signal, stable_set_length, min_seg_length, window_sizes, alpha, bkps, intermediate_temp_path, left_offset=0):
    bootstrap_stableSet = r_base.seq(1, stable_set_length)
    cov_cp_stat_test = r_covcp.covTest(window_sizes, alpha, signal, r_covcp.noPattern, r_covcp.infNorm, bootstrap_stableSet)
    cov_cp_loc = get_covcp_localization(cov_cp_stat_test, window_sizes)
    if cov_cp_loc is None:
        return bkps
    else:
        # add current bkp
        cp, _ = cov_cp_loc
        bkps.append(left_offset + cp)
        if len(bkps) < n_bkps:
            # apply to left and right subsignal
            signal_arr = np.copy(np.asarray(signal))
            if cp - 1 > 2*min_seg_length:
                r_left_subsignal = get_r_left_subsignal(signal_arr, cp-1, intermediate_temp_path)
                # adapting window size
                left_window_sizes = window_sizes
                if r_base.dim(r_left_subsignal)[0] < 2*window_sizes[0]:
                    left_window_sizes = [(r_base.dim(r_left_subsignal)[0])//2 - 1]
                # adapting stable set length
                left_stable_set_length = stable_set_length
                if r_base.dim(r_left_subsignal)[0] < stable_set_length:
                    left_stable_set_length = r_base.dim(r_left_subsignal)[0] - 1
                detect_multiple_covcp_bkps(n_bkps, r_left_subsignal, left_stable_set_length, min_seg_length, left_window_sizes, alpha, bkps, intermediate_temp_path, left_offset=left_offset)
            if signal_arr.shape[0] - cp + 1 > 2*min_seg_length:
                r_right_subsignal = get_r_right_subsignal(signal_arr, cp-1, intermediate_temp_path)
                # adapting window size
                right_window_sizes = window_sizes
                if r_base.dim(r_right_subsignal)[0] < 2*window_sizes[0]:
                    right_window_sizes = [(r_base.dim(r_right_subsignal)[0])//2 - 1]
                # adapting stable set length
                right_stable_set_length = stable_set_length
                if r_base.dim(r_right_subsignal)[0] < stable_set_length:
                    right_stable_set_length = r_base.dim(r_right_subsignal)[0] - 1
                detect_multiple_covcp_bkps(n_bkps, r_right_subsignal, right_stable_set_length, min_seg_length, right_window_sizes, alpha, bkps, intermediate_temp_path, left_offset=left_offset+cp)
        return bkps