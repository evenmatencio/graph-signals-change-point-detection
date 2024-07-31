import numpy as np
import networkx as nx

from scipy.linalg import eigh
from utils import get_min_size_for_hyp, seg_length

### SIGNAL GENERATION
#####################################

def generate_gaus_signal_with_cov_diag_in_basis(n_dims, n_samples, basis, signal_rng, diag_cov_max=1):
    # randomly draw diagonal coef (in the fourier space)
    diag_coefs = diag_cov_max * signal_rng.random(n_dims)
    diag_mat = np.diag(diag_coefs)
    # compute the corresponding covariance matrix and signal 
    cov_mat = basis @ diag_mat @ basis.T
    signal = signal_rng.multivariate_normal(np.zeros(n_dims), cov_mat, size=n_samples)
    return signal

def draw_bkps_with_gap_constraint(n_samples, bkps_gap, bkps_rng, n_bkps_max, max_tries=10000):
    # randomly pick an admissible number of bkps
    n_bkps = bkps_rng.integers(low=1, high=min(n_bkps_max, n_samples // bkps_gap))
    bkps = []
    n_tries = 0
    # select admissible randomly drawn bkps
    while n_tries < max_tries and len(bkps) < n_bkps:
        new_bkp = bkps_rng.integers(low=bkps_gap, high=n_samples-bkps_gap)
        to_keep = True
        for bkp in bkps:
            if abs(new_bkp - bkp) < bkps_gap:
                to_keep = False
                break
        if to_keep:
            bkps.append(new_bkp)
        n_tries+=1
    bkps.sort()
    return bkps + [n_samples]

def draw_fixed_nb_bkps_with_gap_constraint(n_samples, bkps_gap, bkps_rng, max_tries=10000, n_bkps=-1):
    # randomly pick an admissible number of bkps
    if n_bkps < 0:
        n_bkps = bkps_rng.integers(low=n_samples // bkps_gap - 1, high= n_samples // bkps_gap)
    bkps = []
    n_tries = 0
    # select admissible randomly drawn bkps
    while n_tries < max_tries and len(bkps) < n_bkps:
        new_bkp = bkps_rng.integers(low=bkps_gap, high=n_samples-bkps_gap)
        to_keep = True
        for bkp in bkps:
            if abs(new_bkp - bkp) < bkps_gap:
                to_keep = False
                break
        if to_keep:
            bkps.append(new_bkp)
        n_tries+=1
    bkps.sort()
    return bkps + [n_samples]

def generate_rd_signal_in_hyp(G:nx.Graph, signal_rng:np.random.Generator, hyp:seg_length, n_samples:int, diag_cov_max):
    # randomly draw a set of admissible change points
    n_dims = G.number_of_nodes()
    min_size = get_min_size_for_hyp(n_dims=n_dims, hyp=hyp)
    bkps = draw_bkps_with_gap_constraint(n_samples=n_samples, bkps_gap=min_size, bkps_rng=signal_rng, n_bkps_max=n_samples)
    # generate the signal
    _, eigvects = eigh(nx.laplacian_matrix(G).toarray())
    signal_gen_func = lambda size: generate_gaus_signal_with_cov_diag_in_basis(n_dims, size, eigvects, signal_rng, diag_cov_max)
    signal = signal_gen_func(bkps[0])
    # add each sub-segment
    for i in range(1, len(bkps)):
        sub_signal = signal_gen_func(bkps[i] - bkps[i-1])
        signal = np.concatenate([signal, sub_signal], axis=0)
    return bkps, signal.astype(np.float64)

def generate_rd_signal_in_hyp_with_fixed_min_size(G:nx.Graph, signal_rng:np.random.Generator, hyp:seg_length, n_samples:int, min_size_coef:int, diag_cov_max, n_bkps=-1):
    # randomly draw a set of admissible change points
    n_dims = G.number_of_nodes()
    min_size = int(min_size_coef * get_min_size_for_hyp(n_dims=n_dims, hyp=hyp))
    bkps = draw_fixed_nb_bkps_with_gap_constraint(n_samples=n_samples, bkps_gap=min_size, bkps_rng=signal_rng, n_bkps=n_bkps)
    # generate the signal
    _, eigvects = eigh(nx.laplacian_matrix(G).toarray())
    signal_gen_func = lambda size: generate_gaus_signal_with_cov_diag_in_basis(n_dims, size, eigvects, signal_rng, diag_cov_max)
    signal = signal_gen_func(bkps[0])
    # add each sub-segment
    for i in range(1, len(bkps)):
        sub_signal = signal_gen_func(bkps[i] - bkps[i-1])
        signal = np.concatenate([signal, sub_signal], axis=0)
    return bkps, signal.astype(np.float64)

def generate_rd_signal_in_hyp_with_max_tries(G:nx.Graph, signal_rng:np.random.Generator, n_bkps_max, hyp:seg_length, n_samples:int, diag_cov_max=1):
    # randomly draw a set of admissible change points
    n_dims = G.number_of_nodes()
    min_size = get_min_size_for_hyp(n_dims=n_dims, hyp=hyp)
    bkps = draw_bkps_with_gap_constraint(n_samples, min_size, signal_rng, n_bkps_max)
    # generate the signal
    _, eigvects = eigh(nx.laplacian_matrix(G).toarray())
    signal_gen_func = lambda size: generate_gaus_signal_with_cov_diag_in_basis(n_dims, size, eigvects, signal_rng, diag_cov_max)
    signal = signal_gen_func(bkps[0])
    # add each sub-segment
    for i in range(1, len(bkps)):
        sub_signal = signal_gen_func(bkps[i] - bkps[i-1])
        signal = np.concatenate([signal, sub_signal], axis=0)
    return bkps, signal.astype(np.float64)


### SIGNAL MODIFICATION
#####################################

def add_diagonal_white_noise(signal_rng:np.random.Generator, signal, sigma):
    n_samples , n_dims = signal.shape
    cov_mat = sigma * np.eye(n_dims)
    white_noise = signal_rng.multivariate_normal(np.zeros(n_dims), cov_mat, size=n_samples)
    return signal + white_noise

def modify_signal_to_simulate_breakdown(signal, signal_rng, n_breakdown, breakdown_length):
    # initialization
    n_samples = signal.shape[0]
    # randomly pick the location of the breakdowns
    breakdowns = {}
    broken_node_ids = signal_rng.choice(signal.shape[1], size=n_breakdown, replace=False)
    for node_id in broken_node_ids:
        start = int(signal_rng.integers(0, n_samples-breakdown_length))
        end = start + breakdown_length
        signal[start:end, node_id] = 0
        breakdowns[int(node_id)] = (start, end)
    return signal, breakdowns