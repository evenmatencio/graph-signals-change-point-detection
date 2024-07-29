import numpy as np

from datetime import datetime
from ruptures.metrics import precision_recall
from ruptures.metrics import hausdorff
from utils import turn_all_list_of_dict_into_str, create_parent_and_dump_json

def compute_f1_score(preci, recall):
    if preci + recall > 0:
        return 2 * (preci * recall) / (preci + recall)
    return 0

def compute_and_update_metrics(true_bkps, pred_bkps, metrics_dic, prec_rec_margin):
    preci, recall = precision_recall(true_bkps, pred_bkps, prec_rec_margin)
    hsdrf = hausdorff(true_bkps, pred_bkps)
    if hsdrf <= prec_rec_margin:
        preci=1
        if len(pred_bkps) >= len(true_bkps):
            recall = 1
    f1_score = compute_f1_score(preci, recall)
    metrics_dic["precision"]['raw'].append(round(preci, 4))
    metrics_dic["recall"]['raw'].append(round(recall, 4))
    metrics_dic["f1_score"]['raw'].append(round(f1_score, 4))
    metrics_dic["hausdorff"]['raw'].append(int(hsdrf))

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