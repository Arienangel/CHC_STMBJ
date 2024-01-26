from typing import Literal, Union

import numpy as np
import scipy.signal

from .common import *


def extract_data(raw_data: Union[np.ndarray, str, list], height: float = 1.45, length_segment: int = 1200, num_segment: int = 1, offset: list = [0, 0], units: list = [1e-6, 1], **kwargs):
    '''
    Extract traces from raw_data

    Args:
        raw_data (ndarray | str): 2D array (I, V) contains raw data, directory of files, zip file, or txt file
        height (float, optional): peak height of E bias
        length_segment (int, optional): length of one segment
        num_segment (int, oprional):numbber of segments
        offset (list, optional): length from first point to first peak and last point to last peak
        units (list, optional): default: (μA, V)

    Returns:
        I (ndarray): current (A) in 2D array (#traces, length)
        V (ndarray): E bias (V) in 2D array (#traces, length)
    '''
    if not isinstance(raw_data, np.ndarray): raw_data = load_data(raw_data, **kwargs)[::-1]
    I, V = raw_data * np.expand_dims(units, 1)
    peaks, *_ = scipy.signal.find_peaks(abs(V), height=height)
    start = peaks[np.isin(peaks + num_segment * length_segment, peaks)] - offset[0]
    start = start[(start >= 0) & (start + num_segment * length_segment + sum(offset) < I.size)]
    end = start + num_segment * length_segment + sum(offset)
    return np.stack([[I[i:j], V[i:j]] for i, j in zip(start, end) if np.where((peaks >= i) & (peaks <= j), 1, 0).sum() == num_segment + 1], axis=1)


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


class Hist_IVt(Hist2D):

    def __init__(self,
                 xlim: tuple[float, float] = (0, 0.2),
                 y1lim: tuple[float, float] = (1e-11, 1e-5),
                 y2lim: tuple[float, float] = (1e-11, 1e-5),
                 num_x_bin: float = 1000,
                 num_y1_bin: float = 300,
                 xscale: Literal['linear', 'log'] = 'linear',
                 y1scale: Literal['linear', 'log'] = 'log',
                 y2scale: Literal['linear', 'log'] = 'linear',
                 x_conversion: float = 40000,
                 **kwargs) -> None:
        super().__init__(xlim, y1lim, num_x_bin, num_y1_bin, xscale, y1scale, **kwargs)
        self.colorbar.remove()
        self.ax2 = self.ax.twinx()
        self.ax2.axhline(0, color='black', linestyle='dashed')
        self.plot2 = pd.Series()
        self.ax2.set_ylim(*sorted(y2lim))
        self.ax2.set_yscale(y2scale)
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Current (A)')
        self.ax2.set_ylabel('$E_{bias}\/(V)$')
        self.x_conversion = x_conversion

    def add_data(self, I: np.ndarray, V: np.ndarray, **kwargs) -> None:
        """
        Add data into 2D histogram (GV)

        Args:
            I (ndarray): 2D I array with shape (trace, length)
            V (ndarray): 2D Ebias array with shape (trace, length)
        """
        t = np.mgrid[0:I.shape[0]:1, 0:I.shape[1]:1][1] / self.x_conversion
        super().add_data(t, np.abs(I), **kwargs)
        V = pd.Series(list(V)).drop_duplicates()
        V_new = V[~V.isin(self.plot2)]
        if V_new.size > 0:
            self.ax2.plot(t[0], np.stack(V_new).T, color='black', linewidth=0.5)
            self.plot2 = pd.concat([self.plot2, V_new])
