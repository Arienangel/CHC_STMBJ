import concurrent.futures
import glob
import os
from typing import Literal, Union
from zipfile import ZipFile

import datatable as dt
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate
import scipy.optimize
import scipy.signal
from scipy.constants import physical_constants

cmap = matplotlib.colors.LinearSegmentedColormap('Cmap',
                                                 segmentdata={
                                                     'red': [[0, 1, 1], [0.05, 0, 0], [0.1, 0, 0], [0.15, 1, 1], [0.3, 1, 1], [1, 1, 1]],
                                                     'green': [[0, 1, 1], [0.05, 0, 0], [0.1, 1, 1], [0.15, 1, 1], [0.3, 0, 0], [1, 0, 0]],
                                                     'blue': [[0, 1, 1], [0.05, 1, 1], [0.1, 0, 0], [0.15, 0, 0], [0.3, 0, 0], [1, 1, 1]]
                                                 },
                                                 N=256)

G0, *_ = physical_constants['conductance quantum']


def conductance(I: np.ndarray, V: np.ndarray, **kwargs) -> np.ndarray:
    """
    Calculate conductance

    Args:
        I (ndarray): current (A)
        V (ndarray): E bias (V)

    Returns:
        G/G0 (ndarray): conductance
    """
    with np.errstate(divide='ignore'):
        return I / V / G0


def gaussian(x: np.ndarray, a, u, s) -> np.ndarray:
    """
    Gaussian distribution curve

    Args:
        x (ndarray): input value
        a (float | np.ndarray): peak height
        u (float | np.ndarray): average
        s (float | np.ndarray): standard derivative

    Returns:
        x (ndarray): output value
    """
    if isinstance(a, np.ndarray):
        return np.expand_dims(a, axis=1) * np.exp(-((x - np.expand_dims(u, axis=1)) / np.expand_dims(s, axis=1))**2 / 2)
    else:
        return a * np.exp(-((x - u) / s)**2 / 2)


def multi_gaussian(x: np.ndarray, *args: float):
    """
    Sum of multiple gaussian distribution curve

    Args:
        x (ndarray): input value
        args (float): a1, a2, a3..., u1, u2, u3..., s1, s2, s3...

    Returns:
        x (ndarray): output value
    """
    return np.sum([gaussian(x, *i) for i in np.array(args).reshape(3, len(args) // 3).T], axis=0)


def get_peak(X: np.ndarray, Y: np.ndarray, *, window_length=25, polyorder=5, prominence=0.05):
    """
    Get peak position and width by fitting Gaussian function

    Args:
        X (np.ndarray)
        Y (np.ndarray)
        window_length (int)
        polyorder (int)
        prominence (float)

    Returns:
        height (ndarray): peak height
        avg (ndarray): average
        stdev (ndarray): standard derivative
    """
    y = (Y - Y.min()) / (Y.max() - Y.min())
    y = scipy.signal.savgol_filter(y, window_length, polyorder)
    peak, *_ = scipy.signal.find_peaks(y, prominence=prominence)
    _, _, left, right = scipy.signal.peak_widths(y, peak, rel_height=1)
    left, right = np.ceil(left).astype(int), np.ceil(right).astype(int)
    return np.stack([scipy.optimize.curve_fit(gaussian, X[left[i]:right[i]], Y[left[i]:right[i]], bounds=(0, np.inf))[0] for i in range(left.size)]).T


def load_data(path: Union[str, list], recursive: bool = False, max_workers=None, **kwargs) -> np.ndarray:
    """
    Load data from text files.

    Args:
        path (str): directory of files, zip file, or txt file
        recursive (bool, optional): read txt files in folder recursively

    Returns:
        out (ndarray): Data read from the text files.
    """
    if isinstance(path, str):
        if path.endswith('.txt'):
            return np.loadtxt(path, unpack=True)
        elif path.endswith('.npy'):
            return np.load(path)
        elif path.endswith('zip'):
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                return executor.submit(_load_data, zipfile=path, recursive=recursive, **kwargs).result()
        elif os.path.isdir(path):
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                return executor.submit(_load_data, folder=path, recursive=recursive, **kwargs).result()
    elif isinstance(path, list):
        return np.concatenate(list(map(lambda path: load_data(path, recursive, max_workers, **kwargs), path)), axis=-1)


def _load_data(folder: str = None, zipfile: str = None, recursive: bool = False, **kwargs) -> np.ndarray:
    # use subprocess to save memory usage
    if folder:
        path = folder
        files = list(map(lambda f: os.path.join(path, f), glob.glob(os.path.join(path, '**/*.txt'), recursive=True) if recursive else glob.glob(os.path.join(path, '*.txt'), recursive=False)))
        return dt.rbind(dt.iread(files)).to_numpy().T.squeeze()
    elif zipfile:
        path = zipfile
        with ZipFile(path) as zf:
            files = list(map(zf.read, filter(lambda file: file.endswith('.txt') and ('/' not in file or recursive), zf.namelist())))
            return dt.rbind(dt.iread(files)).to_numpy().T.squeeze()


def load_data_with_metadata(path: Union[str, bytes, list], recursive: bool = False, max_workers=None, **kwargs) -> pd.DataFrame:
    """
    Load data and metadata from text files.

    Args:
        path (str): directory of files, zip file, or txt file
        recursive (bool, optional): read txt files in folder recursively

    Returns:
        out (DataFrame): File path, data read from the text files and unix time.
    """
    if isinstance(path, str):
        if path.endswith('.txt'):
            return pd.DataFrame([[path, np.loadtxt(path, unpack=True), os.path.getmtime(path)]], columns=['path', 'data', 'time'])
        elif path.endswith('zip'):
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                return executor.submit(_load_data_with_metadata, zipfile=path, recursive=recursive, **kwargs).result()
        elif os.path.isdir(path):
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                return executor.submit(_load_data_with_metadata, folder=path, recursive=recursive, **kwargs).result()
    elif isinstance(path, list):
        return pd.concat(map(lambda path: load_data_with_metadata(path, recursive, max_workers, **kwargs), path), axis=0)


def _load_data_with_metadata(folder: str = None, zipfile: str = None, recursive: bool = False, **kwargs) -> pd.DataFrame:
    # use subprocess to save memory usage
    if folder:
        path = folder
        df = pd.DataFrame()
        df['path'] = list(map(lambda f: os.path.join(path, f), glob.glob(os.path.join(path, '**/*.txt'), recursive=True) if recursive else glob.glob(os.path.join(path, '*.txt'), recursive=False)))
        df['data'] = df['path'].apply(lambda f: dt.rbind(dt.iread(f)).to_numpy().T.squeeze())
        df['time'] = df['path'].apply(os.path.getmtime)
        return df[['path', 'data', 'time']]
    elif zipfile:
        path = zipfile
        df = pd.DataFrame()
        with ZipFile(path) as zf:
            df['path'] = list(filter(lambda file: file.endswith('.txt') and ('/' not in file or recursive), zf.namelist()))
            df['data'] = df['path'].apply(lambda f: dt.rbind(dt.iread(zf.read(f))).to_numpy().T.squeeze())
            df['time'] = df['path'].apply(lambda f: pd.Timestamp(*zf.getinfo(f).date_time).timestamp())
        return df[['path', 'data', 'time']]


class Hist1D:
    """
    1D histogram

    Args:
        xlim (tuple): max and min value of x
        num_bin (float): number of bins
        x_scale (str): linear or log scale of x axis

    Attributes:
        trace (int): number of traces
        bins (ndarray): 1D array of bin edges
        height (ndarray): height of the histogram
        fig (Figure): plt.Figure object
        ax (Axes): plt.Axes object
        plot (StepPatch): 1D histogram container
    """

    def __init__(self, xlim: tuple, num_x_bin: int, xscale: Literal['linear', 'log'] = 'linear', **kwargs) -> None:
        self.x_min, self.x_max = sorted(xlim)
        self.x_bins = np.linspace(self.x_min, self.x_max, num_x_bin + 1) if xscale == 'linear' else np.logspace(np.log10(self.x_min), np.log10(self.x_max), num_x_bin + 1) if xscale == 'log' else None
        self.height, *_ = np.histogram([], self.x_bins)
        self.trace = 0
        self.xscale = xscale
        self.fig, self.ax = plt.subplots()
        self.plot = self.ax.stairs(np.zeros(self.x_bins.size - 1), self.x_bins, fill=True)
        self.ax.set_xlim(self.x_min, self.x_max)
        self.ax.set_xscale(xscale)
        self.ax.grid(visible=True, which='major')
        self.x = np.sqrt(self.x_bins[1:] * self.x_bins[:-1]) if xscale == 'log' else (self.x_bins[1:] + self.x_bins[:-1]) / 2

    @property
    def height_per_trace(self):
        """ndarray: histogram height divided by number of traces"""
        return self.height / self.trace

    def add_data(self, x: np.ndarray, set_ylim: bool = True, **kwargs) -> None:
        """
        Add data into histogram

        Args:
            x (ndarray): 2D array with shape (trace, length)
            set_ylim (bool, optional): set largest y as y max
        """
        self.trace = self.trace + x.shape[0]
        self.height = self.height + np.histogram(x, self.x_bins)[0]
        height_per_trace = self.height_per_trace
        self.plot.set_data(height_per_trace)
        if set_ylim:
            self.ax.set_ylim(0, height_per_trace.max())

    def clear_data(self):
        self.trace = 0
        self.height.fill(0)
        self.plot.set_data(self.height)


class Hist2D:
    """
    2D histogram

    Args:
        xlim (tuple): max and min value of x
        ylim (tuple): max and min value of y
        num_x_bin (float): number of x bins
        num_y_bin (float): number of y bins
        xscale (str): linear or log scale of x axis
        yscale (str): linear or log scale of y axis

    Attributes:
        trace (int): number of traces
        x_bins (ndarray): 1D array of x bin edges
        y_bins (ndarray): 1D array of y bin edges
        height (ndarray): height of the histogram
        fig (Figure): plt.Figure object
        ax (Axes): plt.Axes object
        plot (StepPatch): 1D histogram container
    """

    def __init__(self, xlim: tuple, ylim: tuple, num_x_bin: int, num_y_bin: int, xscale: Literal['linear', 'log'] = 'linear', yscale: Literal['linear', 'log'] = 'linear', **kwargs) -> None:
        (self.x_min, self.x_max), (self.y_min, self.y_max) = sorted(xlim), sorted(ylim)
        self.x_bins = np.linspace(self.x_min, self.x_max, num_x_bin + 1) if xscale == 'linear' else np.logspace(np.log10(self.x_min), np.log10(self.x_max), num_x_bin + 1) if xscale == 'log' else None
        self.y_bins = np.linspace(self.y_min, self.y_max, num_y_bin + 1) if yscale == 'linear' else np.logspace(np.log10(self.y_min), np.log10(self.y_max), num_y_bin + 1) if yscale == 'log' else None
        self.height = np.histogram2d([], [], (self.x_bins, self.y_bins))[0]
        self.trace = 0
        self.xscale = xscale
        self.yscale = yscale
        self.fig, self.ax = plt.subplots()
        self.plot = self.ax.pcolormesh(self.x_bins, self.y_bins, np.zeros((self.y_bins.size - 1, self.x_bins.size - 1)), cmap=cmap, vmin=0)
        self.ax.set_xlim(self.x_min, self.x_max)
        self.ax.set_ylim(self.y_min, self.y_max)
        self.ax.set_xscale(xscale)
        self.ax.set_yscale(yscale)
        self.colorbar = self.fig.colorbar(self.plot, ax=self.ax, shrink=0.5)
        self.x = np.sqrt(self.x_bins[1:] * self.x_bins[:-1]) if xscale == 'log' else (self.x_bins[1:] + self.x_bins[:-1]) / 2
        self.y = np.sqrt(self.y_bins[1:] * self.y_bins[:-1]) if yscale == 'log' else (self.y_bins[1:] + self.y_bins[:-1]) / 2

    @property
    def height_per_trace(self):
        """ndarray: histogram height divided by trace"""
        return self.height / self.trace

    def add_data(self, x: np.ndarray, y: np.ndarray, set_clim: bool = True, **kwargs) -> None:
        """
        Add data into histogram

        Args:
            x (ndarray): 2D x array with shape (trace, length)
            y (ndarray): 2D y array with shape (trace, length)
            set_clim (bool, optional): set largest z as z max
        """
        self.trace = self.trace + x.shape[0]
        self.height = self.height + np.histogram2d(x.ravel(), y.ravel(), (self.x_bins, self.y_bins))[0]
        height_per_trace = self.height_per_trace
        self.plot.set_array(height_per_trace.T)
        if set_clim:
            self.plot.set_clim(0, height_per_trace.max())

    def clear_data(self):
        self.trace = 0
        self.height.fill(0)
        self.plot.set_array(self.height.T)

    def set_cmap(self, cmap):
        """
        Args:
            cmap (LinearSegmentedColormap | dict): 
                {
                    'red':   [[0, 1, 1], [0.05, 0, 0], [0.1, 0, 0], [0.15, 1, 1], [0.3, 1, 1], [1, 1, 1]],
                    'green': [[0, 1, 1], [0.05, 0, 0], [0.1, 1, 1], [0.15, 1, 1], [0.3, 0, 0], [1, 0, 0]],
                    'blue':  [[0, 1, 1], [0.05, 1, 1], [0.1, 0, 0], [0.15, 0, 0], [0.3, 0, 0], [1, 1, 1]]
                }
        """
        if isinstance(cmap, dict):
            cmap = matplotlib.colors.LinearSegmentedColormap('Cmap', segmentdata=cmap, N=256)
        self.plot.set_cmap(cmap)
