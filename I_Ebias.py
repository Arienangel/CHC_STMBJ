from typing import Literal, Union

import numpy as np
import scipy.signal
import yaml

from base import *


def extract_data(raw_data: Union[np.ndarray, str, list], height: float = 1.45, length: int = 1200, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    '''
    Extract traces from raw_data

    Args:
        raw_data (ndarray | str): 1D array contains raw data, directory of files, zip file, or txt file
        height (float, optional): peak height of E bias
        length (int, optional): length of trace

    Returns:
        I (ndarray): current (A) in 2D array (#traces, length)
        V (ndarray): E bias (V) in 2D array (#traces, length)
    '''
    if not isinstance(raw_data, np.ndarray): raw_data = load_data(raw_data, **kwargs)
    V, I = raw_data * np.array([[1], [1e-6]])
    peaks, *_ = scipy.signal.find_peaks(abs(V), height=height)
    start, end = peaks[:-1].ravel(), peaks[1:].ravel()
    filter = (end - start) == length
    if filter.size >= 2: return np.stack([[I[i:j], V[i:j]] for i, j in zip(start[filter], end[filter])], axis=1)
    else: return np.empty((2, 0, length))


def noise_remove(I: np.ndarray, V: np.ndarray, V_range: float = None, I_max: float = None, **kwargs):
    '''
    Remove noise traces

    Args:
        I (ndarray): current (A) in 2D array (#traces, length)
        V (ndarray): E bias (V) in 2D array (#traces, length)
        V_range (float, optional): Remove traces that I.min() is not between ±V_range
        I_max (float, optional): Remove traces that I.max() is greater than I_max limit

    Returns:
        I (ndarray): current (A) in 2D array (#traces, length)
        V (ndarray): E bias (V) in 2D array (#traces, length)
    '''
    if V_range:
        zero_point = np.diagonal(V[:, np.abs(I).argmin(axis=1)])
        filter = np.abs(zero_point) < V_range
        I, V = I[filter], V[filter]
    if I_max:
        filter = np.where(np.abs(I).max(axis=1) < 1e-5, True, False)
        I, V = I[filter], V[filter]
    return I, V


def zeroing(I: np.ndarray, V: np.ndarray):
    '''
    Set minimum current value at V=0

    Args:
        I (ndarray): current (A) in 2D array (#traces, length)
        V (ndarray): E bias (V) in 2D array (#traces, length)

    Returns:
        I (ndarray): current (A) in 2D array (#traces, length)
        V (ndarray): E bias (V) in 2D array (#traces, length)
    '''
    zero_point = np.diagonal(V[:, np.abs(I).argmin(axis=1)])
    return I, V - np.expand_dims(zero_point, axis=1)


def split_scan_direction(I: np.ndarray, V: np.ndarray, **kwargs) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    '''
    Split E bias scanning direction

    Args:
        I (ndarray): current (A) in 2D array (#traces, length)
        V (ndarray): E bias (V) in 2D array (#traces, length)

    Returns:
        ascending (tuple):  tuple of current and ascending E bias
        descending (tuple):  tuple of current and descending E bias
    '''
    filter = np.where((V[:, -1] - V[:, 0]) > 0, True, False)
    ascending = I[filter], V[filter]
    descending = I[~filter], V[~filter]
    return ascending, descending


class Hist_GV(Hist2D):

    def __init__(self, xlim: tuple[float, float] = (-1.5, 1.5), ylim: tuple[float, float] = (1e-5, 1e-1), num_x_bin: float = 300, num_y_bin: float = 300, xscale: Literal['linear', 'log'] = 'linear', yscale: Literal['linear', 'log'] = 'log', **kwargs) -> None:
        super().__init__(xlim, ylim, num_x_bin, num_y_bin, xscale, yscale, **kwargs)
        self.ax.set_xlabel('$E_{bias}\/(V)$')
        self.ax.set_ylabel('$Conductance\/(G/G_0)$')

    def add_data(self, I: np.ndarray, V: np.ndarray, **kwargs) -> None:
        """
        Add data into 2D histogram (GV)

        Args:
            I (ndarray): 2D I array with shape (trace, length)
            V (ndarray): 2D Ebias array with shape (trace, length)
        """
        G = conductance(I, V)
        super().add_data(V, np.abs(G), **kwargs)


class Hist_IV(Hist2D):

    def __init__(self, xlim: tuple[float, float] = (-1.5, 1.5), ylim: tuple[float, float] = (1e-11, 1e-5), num_x_bin: float = 300, num_y_bin: float = 300, xscale: Literal['linear', 'log'] = 'linear', yscale: Literal['linear', 'log'] = 'log', **kwargs) -> None:
        super().__init__(xlim, ylim, num_x_bin, num_y_bin, xscale, yscale, **kwargs)
        self.ax.set_xlabel('$E_{bias}\/(V)$')
        self.ax.set_ylabel('$Current\/(A)$')

    def add_data(self, I: np.ndarray, V: np.ndarray, **kwargs) -> None:
        """
        Add data into 2D histogram (GV)

        Args:
            I (ndarray): 2D I array with shape (trace, length)
            V (ndarray): 2D Ebias array with shape (trace, length)
        """
        super().add_data(V, np.abs(I), **kwargs)


class Run(Base_Runner):
    """
    Load data and plot

    Args:
        path (str): directory of files, or txt file
        segment (int): number of segments in one cycle
        num_file (int): number of files to finish one cycle

    Attributes:
        hist_GV (Hist_GV)
        hist_IV (Hist_IV)
    """

    def __init__(self, path: str, segment: int = 4, num_file: int=10,  **kwargs) -> None:
        self.hist_GV = Hist_GV(**conf['hist_GV'])
        self.hist_IV = Hist_IV(**conf['hist_IV'])
        self.segment = segment
        self.num_file=num_file
        self.pending = []
        super().__init__(path, **kwargs)

    def add_data(self, path: str, **kwargs) -> None:
        if os.path.isdir(path):
            if not os.listdir(path): return  # empty directory
        self.pending.append(path)
        I, V = extract_data(self.pending[-self.num_file:], **conf['extract_data'])
        if I.shape[0] < self.segment:
            return
        else:
            I, V = noise_remove(I, V, **conf['noise_remove'])
            if I.size:
                self.hist_GV.add_data(I, V)
                self.hist_IV.add_data(I, V)
            self.pending.clear()


if __name__ == '__main__':
    if os.path.exists('config.yaml'):
        with open('config.yaml', mode='r', encoding='utf-8') as f:
            conf = yaml.load(f.read().replace('\\', '/'), yaml.SafeLoader)['I_Ebias']

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path")
    args = parser.parse_args()
    if args.path:
        conf['path'] = args.path

    runner = Run(**conf)
    if conf['realtime']: runner.plot_realtime()
    else: runner.plot_once()