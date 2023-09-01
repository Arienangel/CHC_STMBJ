import os
from typing import Literal, Union

import numpy as np
import scipy.interpolate
import scipy.optimize
import scipy.signal
import yaml

from base import *


def extract_data(raw_data: Union[np.ndarray, str, list], length: int = 1000, upper: float = 3.2, lower: float = 1e-6, method: Literal['pull', 'crash', 'both'] = 'pull', **kwargs) -> np.ndarray:
    '''
    Extract useful data from raw_data

    Args:
        raw_data (ndarray | str): 1D array contains raw data, directory of files, zip file, or txt file
        length (int): length of extracted data per trace
        upper (float): extract data greater than upper limit
        lower (float): extract data less than lower limit
        method (str): 'pull', 'crash' or 'both'

    Returns:
        extracted_data (ndarray): 2D array with shape (trace, length)
    '''
    if not isinstance(raw_data, np.ndarray): raw_data = load_data(raw_data, **kwargs)
    if raw_data.size:
        step, *_ = scipy.signal.find_peaks(np.abs(np.gradient(np.where(raw_data > (upper + lower) / 2, 1, 0))), distance=length)
        if len(step):
            split = np.stack([raw_data[:length] if (i - length // 2) < 0 else raw_data[-length:] if (i + length // 2) > raw_data.size else raw_data[i - length // 2:i + length // 2] for i in step])
            res = split[np.where((split.max(axis=1) > upper) & (split.min(axis=1) < lower), True, False)]
            if method == 'pull': res = res[np.where(res[:, 0] > res[:, -1], True, False)]
            elif method == 'crash': res = res[np.where(res[:, 0] < res[:, -1], True, False)]
            elif method == 'both': pass
            return res
    return np.empty((0, length))


def get_distance(G: np.ndarray, zero_point: float = 0.5, x_conversion: float = 1, **kwargs) -> np.ndarray:
    """
    Get distance from conductance array

    Args:
        G (ndarray): 2D G array with shape (trace, length)
        zero_point (float): set x=0 at G=zero_point
        x_conversion (float, optional): point per distance

    Returns:
        x (ndarray): 2D x array with shape (trace, length)
    """
    _, x = np.mgrid[:G.shape[0], :G.shape[-1]]
    n, x1 = np.where((G[:, :-1] > 0.5) & (G[:, 1:] < 0.5))
    _, start, count = np.unique(n, return_counts=True, return_index=True)
    x1 = x1[start + count - 1]
    x2 = x1 + 1
    y1, y2 = G[:, x1].diagonal(), G[:, x2].diagonal()
    zero_point = x1 + (0.5 - y1) / (y2 - y1) * (x2 - x1)
    return (x - np.expand_dims(zero_point, axis=-1)) / x_conversion


class Hist_G(Hist1D):

    def __init__(self, xlim: tuple[float, float] = (1e-5, 10**0.5), num_bin: float = 550, x_scale: Literal['linear', 'log'] = 'log', **kwargs) -> None:
        super().__init__(xlim, num_bin, x_scale, **kwargs)
        self.ax.set_xlabel('$Conductance\/(G/G_0)$')
        self.ax.set_ylabel('$Count/trace$')

    def add_data(self, G: np.ndarray, **kwargs) -> None:
        """
        Add data into 1D histogram (G)

        Args:
            G (ndarray): 2D array with shape (trace, length)
        """
        super().add_data(G, **kwargs)
        '''peak, *_ = scipy.signal.find_peaks(height_per_trace, prominence=0.1)
        peak_position = self.bins[peak], height_per_trace[peak]
        self.ax.plot(*peak_position, 'xr')
        for i, j in zip(*peak_position):
            self.ax.annotate(f'{i:1.2E}', xy=(i, j+0.02), ha='center', fontsize=8)'''

    def get_peak(self, *, window_length=25, polyorder=5, prominence=0.05):
        """
        Get peak position and width by fitting Gaussian function

        Args:
            window_length (int)
            polyorder (int)
            prominence (float)

        Returns:
            avg (ndarray): average
            stdev (ndarray): standard derivative
            height (ndarray): peak height
        """
        x = np.sqrt(self.x_bins[:-1] * self.x_bins[1:])
        y = (self.height - self.height.min()) / (self.height.max() - self.height.min())
        y = scipy.signal.savgol_filter(y, window_length, polyorder)
        peak, *_ = scipy.signal.find_peaks(y, prominence=prominence)
        _, _, left, right = scipy.signal.peak_widths(y, peak, rel_height=1)
        left, right = np.ceil(left).astype(int), np.ceil(right).astype(int)
        avg, stdev, height = [], [], []
        for i in range(left.size):
            (a, u, s), *_ = scipy.optimize.curve_fit(gaussian, x[left[i]:right[i]], self.height[left[i]:right[i]], bounds=(0, np.inf))
            avg.append(u)
            stdev.append(s)
            height.append(a)
        return np.array(avg), np.array(stdev), np.array(height)


class Hist_GS(Hist2D):

    def __init__(self, xlim: tuple[float, float] = (-0.3, 0.5), ylim: tuple[float, float] = (1e-5, 10**0.5), num_x_bin: float = 800, num_y_bin: float = 550, xscale: Literal['linear', 'log'] = 'linear', yscale: Literal['linear', 'log'] = 'log', x_conversion: float = 800, **kwargs) -> None:
        super().__init__(xlim, ylim, num_x_bin, num_y_bin, xscale, yscale, **kwargs)
        self.ax.set_xlabel('$Distance\/(nm)$')
        self.ax.set_ylabel('$Conductance\/(G/G_0)$')
        self.colorbar.set_label('$Count/trace$')
        self.x_conversion = x_conversion

    def add_data(self, G: np.ndarray, **kwargs) -> None:
        """
        Add data into 2D histogram (GS)

        Args:
            G (ndarray): 2D G array with shape (trace, length)
        """
        x = get_distance(G, zero_point=0.5, x_conversion=self.x_conversion)
        super().add_data(x, G, **kwargs)


class Run(Base_Runner):
    """
    Load data and plot

    Args:
        path (str): directory of files, or txt file

    Attributes:
        hist_G (Hist_G)
        hist_GS (Hist_GS)
    """

    def __init__(self, path: str, **kwargs) -> None:
        self.hist_G = Hist_G(**conf['hist_G'])
        self.hist_GS = Hist_GS(**conf['hist_GS'])
        super().__init__(path, **kwargs)

    def add_data(self, path: str, **kwargs) -> None:
        print(f'File detected: {path}')
        if os.path.isdir(path):
            if not os.listdir(path): return  # empty directory
        extracted = extract_data(path, **conf['extract_data'])
        self.hist_G.add_data(extracted)
        self.hist_GS.add_data(extracted)
        print(f'Traces: {self.hist_G.trace}')


if __name__ == '__main__':
    if os.path.exists('config.yaml'):
        with open('config.yaml', mode='r', encoding='utf-8') as f:
            conf = yaml.load(f.read().replace('\\', '/'), yaml.SafeLoader)['STM_bj']

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path")
    args = parser.parse_args()
    if args.path: conf['path'] = args.path

    runner = Run(**conf)
    if conf['realtime']: runner.plot_realtime()
    else: runner.plot_once()