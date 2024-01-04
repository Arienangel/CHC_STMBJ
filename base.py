import atexit
import io
import multiprocessing
import os
import time
from abc import abstractmethod
from typing import Literal, Union
from zipfile import ZipFile

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.signal
from matplotlib.colors import LinearSegmentedColormap
from scipy.constants import physical_constants
from watchdog.events import FileCreatedEvent, FileSystemEventHandler
from watchdog.observers import Observer

matplotlib.rc('font', size=16)
matplotlib.rc("figure", autolayout=True)
cmap = LinearSegmentedColormap('Cmap',
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


def gaussian(x: np.ndarray, a: float, u: float, s: float) -> np.ndarray:
    """
    Gaussian distribution curve

    Args:
        x (ndarray): input value
        a (float): peak height
        u (float): average
        s (float): standard derivative

    Returns:
        x (ndarray): output value
    """
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


def __read_text(byte: bytes, delimiter='\t'):
    return pd.read_csv(io.BytesIO(byte), delimiter=delimiter, dtype=np.float64, header=None).values.T.squeeze()


def load_data(path: Union[str, bytes, list], threads: int = multiprocessing.cpu_count(), **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """
    Load data from text files.

    Args:
        path (str): directory of files, zip file, or txt file
        threads (int, optional): number of CPU threads to use, default use all

    Returns:
        out (ndarray): Data read from the text files.
    """
    if isinstance(path, list):
        return np.concatenate(list(map(load_data, path)), axis=-1)
    if path.endswith('.npy'):
        return np.load(path)
    if path.endswith('.txt'):
        return np.loadtxt(path, unpack=True)
    else:
        if os.path.isdir(path):
            files = filter(lambda file: file.endswith('.txt'), os.listdir(path))
            files = [open(os.path.join(path, file), 'rb').read() for file in files]
            if files:
                with multiprocessing.Pool(threads) as pool:
                    return np.concatenate(pool.map(__read_text, files), axis=-1)
            else:
                return None
        elif path.endswith('zip'):
            with multiprocessing.Pool(threads) as pool, ZipFile(path) as zf:
                files = filter(lambda file: file.endswith('.txt') and '/' not in file, zf.namelist())
                files = map(zf.read, files)
                return np.concatenate(pool.map(__read_text, files), axis=-1)


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

    def __init__(self, xlim: tuple[float, float], num_x_bin: float, x_scale: Literal['linear', 'log'] = 'linear', **kwargs) -> None:
        self.x_min, self.x_max = sorted(xlim)
        self.x_bins = np.linspace(self.x_min, self.x_max, num_x_bin + 1) if x_scale == 'linear' else np.logspace(np.log10(self.x_min), np.log10(self.x_max), num_x_bin + 1) if x_scale == 'log' else None
        self.height, *_ = np.histogram([], self.x_bins)
        self.trace = 0
        self.fig, self.ax = plt.subplots()
        self.plot = self.ax.stairs(np.zeros(self.x_bins.size - 1), self.x_bins, fill=True)
        self.ax.set_xlim(self.x_min, self.x_max)
        self.ax.set_xscale(x_scale)
        self.ax.grid(visible=True, which='major')

    @property
    def height_per_trace(self):
        """ndarray: histogram height divided by number of traces"""
        return self.height / self.trace

    def add_data(self, x: np.ndarray, **kwargs) -> None:
        """
        Add data into histogram

        Args:
            x (ndarray): 2D array with shape (trace, length)
        """
        self.trace = self.trace + x.shape[0]
        self.height = self.height + np.histogram(x, self.x_bins)[0]
        height_per_trace = self.height_per_trace
        self.plot.set_data(height_per_trace)
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

    def __init__(self, xlim: tuple[float, float], ylim: tuple[float, float], num_x_bin: float, num_y_bin: float, xscale: Literal['linear', 'log'] = 'linear', yscale: Literal['linear', 'log'] = 'linear', **kwargs) -> None:
        (self.x_min, self.x_max), (self.y_min, self.y_max) = sorted(xlim), sorted(ylim)
        self.x_bins = np.linspace(self.x_min, self.x_max, num_x_bin + 1) if xscale == 'linear' else np.logspace(np.log10(self.x_min), np.log10(self.x_max), num_x_bin + 1) if xscale == 'log' else None
        self.y_bins = np.linspace(self.y_min, self.y_max, num_y_bin + 1) if yscale == 'linear' else np.logspace(np.log10(self.y_min), np.log10(self.y_max), num_y_bin + 1) if yscale == 'log' else None
        self.height, *_ = np.histogram2d([], [], (self.x_bins, self.y_bins))
        self.trace = 0
        self.fig, self.ax = plt.subplots()
        self.plot = self.ax.pcolormesh(self.x_bins, self.y_bins, np.zeros((self.y_bins.size - 1, self.x_bins.size - 1)), cmap=cmap, vmin=0)
        self.ax.set_xlim(self.x_min, self.x_max)
        self.ax.set_ylim(self.y_min, self.y_max)
        self.ax.set_xscale(xscale)
        self.ax.set_yscale(yscale)
        self.colorbar = self.fig.colorbar(self.plot, ax=self.ax, shrink=0.5)

    @property
    def height_per_trace(self):
        """ndarray: histogram height devided by trace"""
        return self.height / self.trace

    def add_data(self, x: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        Add data into histogram

        Args:
            x (ndarray): 2D x array with shape (trace, length)
            y (ndarray): 2D y array with shape (trace, length)
        """
        self.trace = self.trace + x.shape[0]
        self.height = self.height + np.histogram2d(x.ravel(), y.ravel(), (self.x_bins, self.y_bins))[0]
        height_per_trace = self.height_per_trace
        self.plot.set_array(height_per_trace.T)
        self.plot.set_clim(0, height_per_trace.max())

    def clear_data(self):
        self.trace = 0
        self.height.fill(0)
        self.plot.set_array(self.height.T)


class Base_Runner(FileSystemEventHandler):
    """
    Load data and plot

    Args:
        path (str): directory of files, or txt file
    """

    def __init__(self, path: str, **kwargs) -> None:
        self.path = path.strip('"')
        self.add_data(self.path)

    def plot_realtime(self, pause=0.5, recursive: bool = False) -> None:
        """
        Plot data and updatee in realtime

        Args:
            pause (float, optional): plt.pause() interval
            recursive (bool): detect new file in subdirectory or not
        """
        if not os.path.isdir(self.path): raise ValueError(f'Path is not a directory: {self.path}')
        observer = Observer()
        observer.schedule(self, path=self.path, recursive=recursive)
        observer.start()
        plt.show(block=False)
        atexit.register(plt.close)
        atexit.register(observer.stop)
        print(f'Monitoring directory: {self.path}')
        while True:
            try:
                plt.pause(pause)
            except KeyboardInterrupt:
                return

    def plot_once(self) -> None:
        """
        Plot data once.
        """
        try:
            plt.show(block=True)
        except KeyboardInterrupt:
            return

    def on_created(self, event):
        if isinstance(event, FileCreatedEvent):
            if (event.src_path.endswith('.txt')):
                try:
                    if os.path.getsize(event.src_path) == 0: time.sleep(0.5)
                    self.add_data(event.src_path)
                except Exception as E:
                    print(f'ERROR: {type(E).__name__}: {E.args}')

    @abstractmethod
    def add_data(self, path: str, **kwargs) -> None:
        """
        Called when on_created() detected new file

        Args:
            path (str): directory of files, zip file, or txt file
        """

    @abstractmethod
    def clear_data(self, **kwargs) -> None:
        """
        Clear all data
        """
