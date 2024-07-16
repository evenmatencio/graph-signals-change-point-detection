import os

import numpy as np
import networkx as nx

from libpysal.weights import KNN
from scipy.spatial.distance import pdist, squareform
from utils import save_graph

### GRAPH GENERATION
#####################################

def generate_random_er_graphs_fixed_nodes_nb(params_rng, nx_graph_seed, n_nodes, target_deg, bandwidth_coef):
    min_edge_p = (1- bandwidth_coef) * target_deg / (n_nodes -1)
    max_edge_p = (1+ bandwidth_coef) * target_deg / (n_nodes -1)
    edge_p = min_edge_p + (max_edge_p-min_edge_p) * params_rng.random()
    G = nx.erdos_renyi_graph(n=n_nodes, p=edge_p, seed=nx_graph_seed)
    params = {"edge_prob": edge_p}
    return G, params

def generate_random_geographic_graph_with_gauss_kernel(params_rng, n_nodes, target_degree):
    # generation of the nodes coordinates
    nodes_coord = params_rng.random(size=(n_nodes, 2))
    # computation of the threshold for the exponential adjacency matrix
    dist_mat_condensed = pdist(nodes_coord, metric='euclidean')
    sigma = np.median(dist_mat_condensed)  # median heuristic
    expsim_condensed = np.exp(-(dist_mat_condensed**2) / (sigma**2))
    # ordering the values to find the right threshold
    ordered_exp_sim = np.sort(expsim_condensed)
    n_edge_for_target_deg = target_degree*n_nodes//2
    threshold = ordered_exp_sim[-n_edge_for_target_deg+1]
    # creating the corresponding adjacency matrix and graph
    adj_mat_condensed = np.where(expsim_condensed > threshold, expsim_condensed, 0.0)
    adj_mat = squareform(adj_mat_condensed)
    G = nx.Graph(adj_mat)
    return G, nodes_coord

def generate_KNN_random_geographic_graph(params_rng, n_nodes, K):
    # generation of the nodes coordinates 
    nodes_coord = params_rng.random(size=(n_nodes, 2))
    # computation of the KNN adjacency matrix
    knn_weights = KNN.from_array(nodes_coord, K)
    # build the graph
    G_directed = knn_weights.to_networkx()
    G = G_directed.to_undirected()
    return G, nodes_coord


### GRAPH MODIFICATION
#####################################

def modify_graph_connectivity_from_binary_adj_mat_2(G:nx.Graph, edge_prop, graph_rng:np.random.Generator):
    # initialization
    n_edges = G.number_of_edges()
    n_nodes = G.number_of_nodes()
    n_modif_edges = int(n_edges * edge_prop)
    adj_mat = nx.adjacency_matrix(G).toarray()
    # retrieving the actual graph edges
    directed_edges = [(i, j) for i, j in zip(np.nonzero(adj_mat)[0], np.nonzero(adj_mat)[1])]
    non_directed_edges = set([(min(e), max(e)) for e in directed_edges])
    # listing all possible edges to select those to be add
    all_possible_edges_list = []
    for i in range(n_nodes-1):
        for j in range(i+1, n_nodes):
            all_possible_edges_list.append((i, j))
    all_possible_edges_set = set(all_possible_edges_list)
    # removing some random edges
    n_edges_to_remove = n_modif_edges // 2
    edges_id_to_remove = graph_rng.choice(len(non_directed_edges), min(n_edges_to_remove, len(non_directed_edges)), replace=False)
    for edge_id in edges_id_to_remove:
        edge_to_remove = list(non_directed_edges)[edge_id]
        adj_mat[edge_to_remove[0], edge_to_remove[1]] = 0
        adj_mat[edge_to_remove[1], edge_to_remove[0]] = 0
    # adding some edges
    possibles_edges_to_add = all_possible_edges_set.difference(non_directed_edges)
    possibles_edges_to_add_list = list(possibles_edges_to_add)
    n_edges_to_add = n_edges_to_remove
    new_edges_ids = graph_rng.choice(len(possibles_edges_to_add), min(n_edges_to_add, len(possibles_edges_to_add)), replace=False)
    new_edges_list = [possibles_edges_to_add_list[edge_id] for edge_id in new_edges_ids]
    for new_edge in new_edges_list:
        adj_mat[new_edge[1], new_edge[0]] = 1
        adj_mat[new_edge[0], new_edge[1]] = 1
    new_G = nx.from_numpy_array(adj_mat)
    return new_G

def load_modify_connec_and_store_graph(original_graph_path:int, exp_id:int, edge_prop:float, rng:np.random.Generator, target_dir:str):
    adj_mat = np.load(f"{original_graph_path}/{exp_id}_mat_adj.npy", allow_pickle=False)
    G = nx.from_numpy_array(adj_mat)
    G_modif = modify_graph_connectivity_from_binary_adj_mat_2(G, edge_prop, rng)
    save_graph(G_modif, f"{target_dir}/{exp_id}_mat_adj.npy")

def pick_another_graph(graph_path: str, previous_exp_id: int, max_id: int, graph_rng: np.random.Generator):
    new_exp_id = previous_exp_id
    while new_exp_id == previous_exp_id:
        new_exp_id = graph_rng.integers(low=0, high=max_id)
    new_exp_id = str(new_exp_id)
    new_adj_mat = np.load(f"{graph_path}/{new_exp_id}_mat_adj.npy", allow_pickle=False)
    new_G = nx.from_numpy_array(new_adj_mat)
    return new_G, new_exp_id