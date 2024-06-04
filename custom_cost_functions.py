import numpy as np

from scipy.linalg import eigh
from ruptures.base import BaseCost


class MyL2Cost(BaseCost):

    """The standard L2 cost for fixed variance gaussian hypothesis."""

    model = "my_l2_cost"
    min_size = 1

    def __init__(self) -> None:
        self.signal_square_cumsum = None
        self.signal_cumsum = None
        self.signal = None
        super().__init__()
    
    def fit(self, signal):
        """
        Performs pre-computations for per-segment approximation cost.

        args:
            signal: (array) of shape [n_samples, n_dim]
        """
        self.signal = signal
        self.signal_square_cumsum =  np.concatenate([np.zeros((1, signal.shape[1])), np.cumsum(signal**2, axis=0)], axis=0)
        self.signal_cumsum =  np.concatenate([np.zeros((1, signal.shape[1])), np.cumsum(signal, axis=0)], axis=0)
        return self

    def error(self, start, end):
        """Return the L2 approximation cost on the segment [start:end] 
        where end is excluded.

        Args:
            start (int): start of the segment
            end (int): end of the segment

        Returns:
            float: segment cost
        """
        if end - start < self.min_size:
            raise ValueError(f'end - start shoud be higher than {self.min_size}')
        
        sub_square_sum = self.signal_square_cumsum[end] - self.signal_square_cumsum[start]
        sub_sum = self.signal_cumsum[end] - self.signal_cumsum[start]
        return np.sum(sub_square_sum - (sub_sum**2) / (end - start))  
    

class CostGFSSL2(BaseCost):

    """
    Applies the GFSS rotation to the whole signal before computing
    the standard L2 cost for fixed variance gaussian hypothesis.
    """

    model = "gfss_l2_cost"
    min_size = 1

    def __init__(self, laplacian_mat, cut_sparsity) -> None:
        """
        Args:
            laplacian_mat (array): the discrete Laplacian matrix of the graph: D - W
            where D is the diagonal matrix diag(d_i) of the node degrees and W the adjacency matrix
            cut_sparsity (float): frequency threshold of the GFSS spectral filter
        """
        self.cut_sparsity = cut_sparsity
        self.graph_laplacian_mat = laplacian_mat
        self.signal = None
        self.gft_square_cumsum = None
        self.gft_cumsum = None
        super().__init__()

    def filter(self, freqs, eps=0.00001):
        """Applies the GFSS filter to the input (spatial) frequences.
        NOTE: the frequences must be in increasing order.

        Args:
            freqs (array): ordered frequences to filter.
            eps (float, optional): threshold for non zero values. Defaults to 0.00001.

        Returns:
            filtered_freqs (array): the output of the filter.
        """
        nb_zeros = np.sum(freqs < eps)
        filtered_freqs = np.minimum(1, np.sqrt(self.cut_sparsity / freqs[nb_zeros:]))
        return np.concatenate([np.zeros(nb_zeros), filtered_freqs])
    
    def fit(self, signal):
        """Performs pre-computations for per-segment approximation cost.

        NOTE: the number of dimensions of the signal and their ordering
        must match those of the nodes of the graph.
        The function eigh used below returns the eigenvector corresponding to 
        the ith eigenvalue in the ith column eigvect[:, i]

        Args:
            signal (array): of shape [n_samples, n_dim].
        """
        self.signal = signal
        # Computation of the GFSS
        eigvals, eigvects = eigh(self.graph_laplacian_mat)
        filter_matrix = np.diag(self.filter(eigvals), k=0)
        gft = filter_matrix.dot(eigvects.T.dot(signal.T)).T
        # Computation of the per-segment cost utils
        self.gft_square_cumsum =  np.concatenate([np.zeros((1, signal.shape[1])), np.cumsum(gft**2, axis=0)], axis=0)
        self.gft_cumsum =  np.concatenate([np.zeros((1, signal.shape[1])), np.cumsum(gft, axis=0)], axis=0)
        return self

    def error(self, start, end):
        """Return the L2 approximation cost on the segment [start:end] 
        where end is excluded, over the filtered signal.

        Args:
            start (int): start of the segment
            end (int): end of the segment

        Returns:
            float: segment cost
        """
        if end - start < self.min_size:
            raise ValueError(f'end - start shoud be higher than {self.min_size}')
        
        sub_square_sum = self.gft_square_cumsum[end] - self.gft_square_cumsum[start]
        sub_sum = self.gft_cumsum[end] - self.gft_cumsum[start]
        return np.sum(sub_square_sum - (sub_sum**2) / (end  - start))  

