import numpy as np

import rpy2
import rpy2.robjects as robjects

from rpy2.robjects.packages import importr
from sklearn.covariance import log_likelihood

r_base = importr('base')
r_RcppCNPy = importr('RcppCNPy')
r_covcp = importr('covcp')
r_glasso = importr('glasso')

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