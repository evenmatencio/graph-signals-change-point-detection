import os
import json
import subprocess

import numpy as np
import networkx as nx

from pathlib import Path
from typing import Literal, List

seg_length = Literal["large", "minimal"]

def get_min_size_for_hyp(n_dims, hyp:seg_length):
    # the minimal segment length for admissible computations
    min_size = n_dims
    if hyp == "large":
        #for segment long enough for good estimates
        min_size = n_dims * (n_dims-1) / 2
    return min_size

def open_json(path):
    with open(path, 'r+') as f:
        content = json.load(f)
    return content

def create_parent_and_dump_json(parent_dir, name, data, indent=None):
    if os.path.exists(os.path.join(parent_dir, name)):
        raise FileExistsError
    if not os.path.exists(parent_dir):
        Path(parent_dir).mkdir(parents=True, exist_ok=False)
    with open(os.path.join(parent_dir, name), 'w+') as f:
        json.dump(data, f, indent=indent)

def load_and_write_json(json_path, new_key, new_data, indent):
    stored_data = open_json(json_path)
    stored_data[new_key] = new_data
    with open(json_path, 'a+') as f:
        json.dump(stored_data, f, indent=indent)

def read_data_1(graph_path: str, signal_path: str, exp_id: int):
    adj_mat = np.load(f"{graph_path}/{exp_id}_mat_adj.npy", allow_pickle=False)
    G = nx.from_numpy_array(adj_mat)
    signal = np.load(f"{signal_path}/{exp_id}_signal.npy", allow_pickle=False)
    bkps = open_json(f"{signal_path}/{exp_id}_bkps.json")
    return G, signal, bkps

def load_data(graph_path: str, signal_path: str, exp_id: int, hyp:seg_length):
    G, signal, bkps = read_data_1(graph_path, signal_path, exp_id)
    min_size = get_min_size_for_hyp(G.number_of_nodes(), hyp=hyp)
    return G, signal, bkps, min_size

def save_signal_and_bkps(signal:np.ndarray, bkps:List[int], dir, name):
    path = os.path.join(dir, name)
    if os.path.exists(f"{path}_signal.npy"):
        raise FileExistsError
    # save signal
    with open(f"{path}_signal.npy", 'wb+') as np_f:
        np.save(np_f, signal, allow_pickle=False)
    # save bkps
    bkps_str = [int(bkp) for bkp in bkps]
    with open(f"{path}_bkps.json", 'w+') as f:
        json.dump(bkps_str, f, indent=None)

def save_graph(G:nx.Graph, path):
    '''note: nx.to_numpy_array(G) returns the same object as nx.adjacency_matrix(G).toarray() '''
    if os.path.exists(path):
        raise FileExistsError
    # save graph
    adj_mat = nx.to_numpy_array(G)
    with open(path, 'wb+') as nx_f:
        np.save(nx_f, adj_mat, allow_pickle=False)

def turn_str_of_list_into_list_of_int(list_str):
    assert list_str[0] == '[' and list_str[-1] == ']'
    list_of_str = list_str[1:-1].split(',')
    return [int(val_str) for val_str in list_of_str]

def turn_str_of_list_into_list_of_float(list_str):
    assert list_str[0] == '[' and list_str[-1] == ']'
    list_of_str = list_str[1:-1].split(',')
    return [float(val_str) for val_str in list_of_str]

def turn_all_list_of_dict_into_str(data:dict):
    new_dict = {}
    for key, val in data.items():
        if isinstance(val, list):
            new_dict[key] = str(val)
        elif isinstance(val, dict):
            new_dict[key] = turn_all_list_of_dict_into_str(val)
        else:
            new_dict[key] = val
    return new_dict

def get_git_head_short_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
