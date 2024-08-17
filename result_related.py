import scipy.optimize

import numpy as np

from datetime import datetime
from ruptures.metrics import precision_recall, hausdorff, randindex
from utils import turn_all_list_of_dict_into_str, create_parent_and_dump_json


def update_pred_dic_with_one_exp(t1, t2, pred_bkps, gt_bkps, exp_id):
    res = {}
    res["time"] = round(t2 - t1, ndigits=3)
    res["pred"] = pred_bkps
    res["gt"] = gt_bkps
    res["n_bkps"] = len(gt_bkps)-1
    return res

def compute_f1_score(preci, recall):
    if preci + recall > 0:
        return 2 * (preci * recall) / (preci + recall)
    return 0

def compute_assignement_cost(true_bkps, pred_bkps, metrics_dic):
    # initialize cost matrix
    cost_matrix = -1 * np.ones((len(true_bkps)-1, len(pred_bkps)-1))
    for i, gt_bkp in enumerate(true_bkps[:-1]):
        for j, pred_bkp in enumerate(pred_bkps[:-1]):
            cost_matrix[i, j] = abs(gt_bkp - pred_bkp)
    # find solution and compute resulting cost
    solution_row_ind, solution_col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
    assignement_cost = cost_matrix[solution_row_ind, solution_col_ind].mean()
    metrics_dic['assignement_cost']['raw'].append(int(assignement_cost))

def compute_and_update_metrics(true_bkps, pred_bkps, metrics_dic, prec_rec_margin_list):
    hsdrf = hausdorff(true_bkps, pred_bkps)
    randind = randindex(true_bkps, pred_bkps)
    for margin in prec_rec_margin_list:
        preci, recall = precision_recall(true_bkps, pred_bkps, margin)
        if hsdrf <= margin:
            preci=1
            if len(pred_bkps) >= len(true_bkps):
                recall = 1
        f1_score = compute_f1_score(preci, recall)
        metrics_dic[f"margin_{margin}"]["precision"]['raw'].append(round(preci, 4))
        metrics_dic[f"margin_{margin}"]["recall"]['raw'].append(round(recall, 4))
        metrics_dic[f"margin_{margin}"]["f1_score"]['raw'].append(round(f1_score, 4))
    metrics_dic["hausdorff"]['raw'].append(int(hsdrf))
    metrics_dic["randindex"]['raw'].append(round(randind, 4))

def compute_and_add_stat_on_metrics(model_metrics:dict):
    for model_res in model_metrics.values():
        for metric_name, res in model_res.items():
            model_res[metric_name]['mean'] = round(np.mean(res['raw']), ndigits=4)
            model_res[metric_name]['std'] = round(np.std(res['raw']), ndigits=4)
    return model_metrics

def save_metrics(metrics_per_models, stats, dir, comment=''):
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    to_save = {"date_time": now, 'comment': comment}
    to_save["hyper-parameters"] = stats
    to_save["results"] = metrics_per_models
    to_save = turn_all_list_of_dict_into_str(to_save)
    create_parent_and_dump_json(dir, now + '.json', to_save, indent=4)
