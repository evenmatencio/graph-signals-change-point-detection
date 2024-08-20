import argparse
import os
import json

import numpy as np

import utils as my_ut
import result_related as my_res


def compute_and_store_results(kwargs_namespace: argparse.Namespace):
    
    margin_val_list = kwargs_namespace.margins
    verbose = kwargs_namespace.verbose

    # collecting the folder paths corresponding to each experiment
    res_folder_root = kwargs_namespace.pred_fold_root
    folder_name_list = os.listdir(kwargs_namespace.pred_fold_root)
    pred_dir_list = [os.path.join(res_folder_root, file_name) for file_name in folder_name_list]

    # initialization: for each experiment, create the json file for metric storage
    for pred_dir in pred_dir_list:
        with open(os.path.join(pred_dir, 'metrics.json'), 'w+') as f:
            json.dump({}, f, indent=0)

    # for each method 
    for file_name in kwargs_namespace.method_pred_file_names:
        if verbose:
            print(f"Working over {file_name}...")

        # for each experiment 
        for pred_dir in pred_dir_list:

            # formatting the metrics storage
            method_metrics = {"randindex":  {'raw': []}, "hausdorff": {'raw': []}, "assignement_cost": {'raw': []}, "time": {"raw": []}}
            for margin in margin_val_list:
                method_metrics[f"margin_{margin}"] = {"recall": {'raw': []}, "precision": {'raw': []}, "f1_score": {'raw': []}}

            # retrieving the different iterations of the target experiment,
            # with specific pre-processing for the real dataset eeg-movement
            method_pred = my_ut.open_json(f"{pred_dir}/{file_name}.json")
            method_keys = list(method_pred.keys())
            if kwargs_namespace.eeg_movement:
                method_keys = ['0' + i for i in list(method_pred.keys()) if len(i)==1] + [i for i in list(method_pred.keys()) if len(i)==2]
                method_keys.sort()
                if verbose:
                    print(method_keys)
            # compute metrics for each iteration of the current experiment
            for exp_id in method_keys:
                method_pred_bkps = my_ut.turn_str_of_list_into_list_of_int(method_pred[exp_id]["pred"])
                gt_bkps = my_ut.turn_str_of_list_into_list_of_int(method_pred[exp_id]["gt"])
                my_res.compute_and_update_metrics(gt_bkps, method_pred_bkps, method_metrics, margin_val_list)
                my_res.compute_assignement_cost(gt_bkps, method_pred_bkps, method_metrics)
                method_metrics["time"]["raw"].append(method_pred[exp_id]["time"])

            # compute statistics on the metrics
            for metric_name, res in method_metrics.items():
                # for f1, precsion and recall
                if 'margin' in metric_name:
                    for metric_name, res2 in res.items():
                        res[metric_name]['mean'] = round(np.mean(res2['raw']), ndigits=4)
                        res[metric_name]['std'] = round(np.std(res2['raw']), ndigits=4)
                # for randindex, haussdorf and assignement_cost
                else:
                    method_metrics[metric_name]['mean'] = round(np.mean(res['raw']), ndigits=4)
                    method_metrics[metric_name]['std'] = round(np.std(res['raw']), ndigits=4)
       
            # storing the result of the current metric for the current experiment
            formatted_metrics = my_ut.turn_all_list_of_dict_into_str(method_metrics)
            my_ut.load_and_write_json(os.path.join(pred_dir, 'metrics.json'), new_key=file_name, new_data=formatted_metrics, indent=4)
    

        
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # required arguments
    parser.add_argument("pred_fold_root", help="The relative path to the directory containing the single experiment folders. These folders should all contain the file referenced in file_names.")
    parser.add_argument("--method-pred-file-names", nargs='+', help="The (several) file names containing the prediction, without the extension (must be json files located in the folders in exp_fold_path). Each of them correspond to a method. Example:  statio_pred-normal_pred covcp_windsize_80_stable_set_80")
    parser.add_argument("--margins", type=int, nargs='+', help="The (several) margins values to compute the precision, recall and F1 score")

    # optional arguments
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbosity level")
    parser.add_argument("--eeg-movement", action="store_true", help="Whether to run over the real dataset EEG imagerymovement.")

    args = parser.parse_args()
    compute_and_store_results(args)