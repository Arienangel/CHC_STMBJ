from typing import Literal, Union

import numpy as np
import scipy.signal

from baseclass import *


def extract_data(raw_data: Union[np.ndarray, str, list], height: float = 1.45, length: int = 1200, offset: list = [0, 0], units: list = [1e-6, 1], **kwargs) -> tuple[np.ndarray, np.ndarray]:
    '''
    Extract traces from raw_data

    Args:
        raw_data (ndarray | str): 2D array (I, V) contains raw data, directory of files, zip file, or txt file
        height (float, optional): peak height of E bias
        length (int, optional): length of trace
        offset (list, optional): offset length to first peak and final peak
        units (list, optional): default: (μA, V)

    Returns:
        I (ndarray): current (A) in 2D array (#traces, length)
        V (ndarray): E bias (V) in 2D array (#traces, length)
    '''
    if not isinstance(raw_data, np.ndarray): raw_data = load_data(raw_data, **kwargs)[::-1]
    I, V = raw_data * np.expand_dims(units, 1)
    peaks, *_ = scipy.signal.find_peaks(abs(V), height=height)
    start, end = peaks[:-1].ravel(), peaks[1:].ravel()
    f = list(map(lambda x: x + length - sum(offset) in end, start))
    if len(f) >= 1: return np.stack([[I[i:j], V[i:j]] for i, j in zip(start[f] - offset[0], start[f] - offset[0] + length)], axis=1)
    else: return np.empty((2, 0, length))


def noise_remove(I: np.ndarray, V: np.ndarray, V_range: float = None, I_max: float = None, **kwargs) -> tuple[np.ndarray, np.ndarray]:
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
        filter = np.where(np.abs(I).max(axis=1) < I_max, True, False)
        I, V = I[filter], V[filter]
    return I, V


def zeroing(I: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
        self.ax.set_ylabel('Conductance ($G/G_0$)')
        self.colorbar.set_label('Count/trace')

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
        self.ax.set_ylabel('Current (A)')
        self.colorbar.set_label('Count/trace')

    def add_data(self, I: np.ndarray, V: np.ndarray, **kwargs) -> None:
        """
        Add data into 2D histogram (GV)

        Args:
            I (ndarray): 2D I array with shape (trace, length)
            V (ndarray): 2D Ebias array with shape (trace, length)
        """
        super().add_data(V, np.abs(I), **kwargs)
