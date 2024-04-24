import numpy as np

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
            signal: (np.ndarray) of shape [n_samples, n_dim]
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