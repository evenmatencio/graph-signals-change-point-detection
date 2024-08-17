import argparse

import numpy as np
import networkx as nx
import running_cpd as my_cpd
import utils as my_ut

from datetime import datetime

def run_cpd_algorithms(kwargs_namespace: argparse.Namespace):

    ### OVERALL INITIALIZATION
    # logging
    pred_dir = kwargs_namespace.pred_dir
    verbose = kwargs_namespace.verbose
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    my_ut.create_parent_and_dump_json(pred_dir, f"experiment_metadata_{now}.json", my_ut.turn_all_list_of_dict_into_str(kwargs_namespace.__dict__), indent=4)
    # parsing
    signal_path = kwargs_namespace.signal_path
    graph_path = kwargs_namespace.graph_path
    seg_length_hyp = kwargs_namespace.min_size_hypothesis
    exp_id_list = [str(id) for id in range(kwargs_namespace.exp_id_min, kwargs_namespace.exp_id_max)]
    if kwargs_namespace.eeg_movement:
        exp_id_list = ['0' + str(exp_id) for exp_id in exp_id_list if len(str(exp_id)) == 1] + [str(exp_id) for exp_id in exp_id_list if len(str(exp_id)) == 2]

    # COST FUNCTION INITIALIZATION
    # graph stationarity-based
    if kwargs_namespace.statio:
        statio_name = "statio_pred.json"
        statio_json_path = my_cpd.init_pred_saving(pred_dir, statio_name)
    # standard_mle
    if kwargs_namespace.standard_mle:
        mle_name = "standard_mle_pred.json"
        mle_json_path = my_cpd.init_pred_saving(pred_dir, mle_name)
    # r_glasso
    if kwargs_namespace.glasso:
        lasso_buffer_path = f"data_1/.temp/glasso_subsignal_{kwargs_namespace.glasso_buffer_id}"
        lasso_pen_mult_coef = kwargs_namespace.glasso_pen_multcoef
        lasso_name = f"lasso_penmultcoef_{lasso_pen_mult_coef}_pred.json"
        lasso_json_path = my_cpd.init_pred_saving(pred_dir, lasso_name)
    # r_covcp
    if kwargs_namespace.r_covcp:
        r_covcp_stable_set_length = kwargs_namespace.r_covcp_stable_set_length
        r_covcp_window_size = [kwargs_namespace.r_covcp_window_size]
        r_covcp_level = kwargs_namespace.r_covcp_level
        r_covcp_nb_cores = kwargs_namespace.r_covcp_nb_cores
        r_covcp_seed = kwargs_namespace.r_covcp_seed
        covcp_name = f"covcp_windsize_{r_covcp_window_size[0]}_stableset_{r_covcp_stable_set_length}.json"
        covcp_json_path = my_cpd.init_pred_saving(pred_dir, covcp_name)

    # RUNNING CPD ALGORTIHMS
    for exp_id in exp_id_list:
        print(f"\n\tOver exp_id={exp_id}...")
        # loading data
        if kwargs_namespace.eeg_movement:
            G = nx.from_numpy_array(np.load(graph_path, allow_pickle=False))
            volunteer_id = (signal_path).split('/')[-1]
            signal_file_path = f'{signal_path}/volunteer{volunteer_id}_exp{exp_id}'
            gt_bkps = my_ut.open_json(f"{signal_file_path}_bkps.json")
            signal = np.load(f"{signal_file_path}_signal.npy", allow_pickle=False)
            min_size = my_ut.get_min_size_for_hyp(G.number_of_nodes(), hyp=seg_length_hyp)
        else:
            signal_file_path = f"{signal_path}/{exp_id}"
            G, signal, gt_bkps, min_size = my_ut.load_data(graph_path, signal_path, exp_id, seg_length_hyp)
        if kwargs_namespace.statio:
            my_cpd.run_numba_statio_normal_cost_and_store_res(G, signal, gt_bkps, min_size, statio_json_path, exp_id)
        if kwargs_namespace.standard_mle:
            my_cpd.run_numba_standard_mle_normal_cost_and_store_res(signal, gt_bkps, min_size, mle_json_path, exp_id)
        if kwargs_namespace.r_covcp:
            my_cpd.run_r_covcp_cpd_algo_and_store(signal_file_path, gt_bkps, covcp_json_path, r_covcp_stable_set_length, min_size, r_covcp_window_size, r_covcp_level, exp_id, r_covcp_nb_cores, r_covcp_seed)
        if kwargs_namespace.glasso:
            my_cpd.run_r_glasso_cpd_algo_and_store(signal, gt_bkps, lasso_json_path, exp_id, lasso_pen_mult_coef, lasso_buffer_path)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # data to run over: signal, graphs, exp_ids...
    parser.add_argument("signal_path", help="The relative path to the folder containing the signal and bkps files")
    parser.add_argument("exp_id_min", type=int, help='The lower bound of the target experiments ids')
    parser.add_argument("exp_id_max", type=int, help='The upper bound of the target experiments ids (excluded)')
    parser.add_argument("min_size_hypothesis", choices=['minimal', 'large'], default="minimal", help='the minimal size of the stationarity segment for the cpd algorithms. large --> min_size=n_dim*(n_dim-1)/2, minimal --> min_size=n_dim')
    parser.add_argument("--graph-path", help="The relative path to the folder containing the graph files")
    parser.add_argument("--eeg-movement", action="store_true", help="Whether to run over the real dataset EEG imagerymovement")
    # cost functions to run
    parser.add_argument("--statio", action="store_true", help="Whether to run the graph stationarity based cost function")
    parser.add_argument("--standard-mle", action="store_true", help="Whether to run the mle cost function")
    parser.add_argument("--glasso", action="store_true", help="Whether to run the Graph Lasso cost function")
    parser.add_argument("--r-covcp", action="store_true", help="Whether to run the r_covcp cost function")
    # logging and output
    parser.add_argument("pred_dir", help="The relative path to the folder for result saving")
    parser.add_argument("-v", "--verbose", type=int, choices=[0, 1], default=0, help="Verbosity level")
    # cost functions hyper-parameters
    parser.add_argument("--glasso-pen-multcoef", nargs='?', const=4., type=float, default=4., help="The multiplicative factor applied to the penalty coefficient used for graph lasso, equal to glasso-pen-multcoef * sqrt(log(n_dim) / n_samples)")
    parser.add_argument("--glasso-buffer-id", nargs='?', const=1, type=int, default=1, help="Buffer file id for graph lasso intermediate file saving, must be chosen carefully so that each cpd thread has its own buffer id")
    parser.add_argument("--r-covcp-stable-set-length", nargs='?', const=80, type=int, default=80, help='The number of samples (from the first one) contained in the bootstrap set for the r_covcp method, must not contain bkp and should be chosen on the knowledge of the signal')
    parser.add_argument("--r-covcp-window-size", nargs='?', const=80, type=int, default=80, help='The window size used to compute the covariance matrix estimator and then the statistic')
    parser.add_argument("--r-covcp-level", nargs='?', const=0.3, type=float, default=0.3, help='The level of the test used to compute the threshold for cpd')
    parser.add_argument("--r-covcp-nb-cores", nargs='?', const=1, type=int, default=1, choices=[1, 2, 3], help='The number of cores allocated to the R library covcp for cpd solving')
    parser.add_argument("--r-covcp-seed", nargs='?', const=42, type=int, default=42, help='Random seed used in the algorithm from the R library covcp')

    args = parser.parse_args()
    run_cpd_algorithms(args)
