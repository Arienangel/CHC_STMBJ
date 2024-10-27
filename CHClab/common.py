import concurrent.futures
import glob
import logging
import os
from typing import Iterable, Literal, Union
from zipfile import ZipFile

import matplotlib.axes
import matplotlib.colors
import matplotlib.lines
import matplotlib.ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

G0 = physical_constants['conductance quantum'][0]


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


def gaussian(x: np.ndarray, a: float | np.ndarray, u: float | np.ndarray, s: float | np.ndarray) -> np.ndarray:
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


def load_data(path: Union[str, Iterable], recursive: bool = False, max_workers: int = None, delimiter: str = '\t', transpose: bool = True, squeeze: bool = True, **kwargs) -> np.ndarray:
    """
    Load data from text files.

    Args:
        path (str): directory of files, zip file, or txt file
        recursive (bool, optional): read txt files in folder recursively
        max_workers (int, optional): maximum number of processes that can be used
        delimiter (str, optional):  character used to separate the values
        transpose (bool, optional): transpose row and column
        squeeze (bool, optional): remove dimensions of length 1

    Returns:
        out (ndarray): Data read from the text files.
    """

    def _load(path) -> list:
        if isinstance(path, str):
            # read all txt files in directory
            if os.path.isdir(path):
                files = map(lambda f: os.path.join(path, f), glob.glob('**/*.txt', root_dir=path, recursive=True) if recursive else glob.glob('*.txt', root_dir=path, recursive=False))
                try:
                    import datatable as dt
                    return list(dt.iread(list(files)))
                except ImportError:
                    logging.info('Module datatable was not found. Use pandas instead.')
                    return list(thread_executor.map(lambda f: pd.read_csv(f, sep=delimiter, header=None, engine='c'), files))
            # read all txt files in zip file
            elif path.endswith('zip'):
                with ZipFile(path) as zf:
                    files = list(thread_executor.map(zf.read, filter(lambda file: file.endswith('.txt') and ('/' not in file or recursive), zf.namelist())))
                    try:
                        import datatable as dt
                        return list(dt.iread(files))
                    except ImportError:
                        logging.info('Module datatable was not found. Use pandas instead.')
                        import io
                        return list(thread_executor.map(lambda f: pd.read_csv(io.BytesIO(f), sep=delimiter, header=None, engine='c'), files))
            # read txt file
            elif path.endswith('.txt'):
                return [pd.read_csv(path, sep=delimiter, header=None, engine='c')]
            # read npy file
            elif path.endswith('.npy'):
                return [np.load(path)]
            # read all groups in tdms file
            elif path.endswith('tdms'):
                try:
                    from nptdms import TdmsFile
                    with TdmsFile.read(path) as f:
                        return list(thread_executor.map(lambda g: g.as_dataframe(), f.groups()))
                except ImportError:
                    raise ImportError('Module nptdms was not found. TDMS files can not be read.')
        elif isinstance(path, Iterable):
            return sum(thread_executor.map(_load, path), start=[])

    res = None
    if isinstance(path, list):
        if len(path) == 1: path = path[0]
    if isinstance(path, str):
        if path.endswith('.txt'): res = np.array(pd.read_csv(path, sep=delimiter, header=None, engine='c'))
        elif path.endswith('.npy'): res = np.load(path)
    if res is None:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as thread_executor:
            logging.debug(f'Use {thread_executor._max_workers} workers in load_data')
            res = np.concatenate(_load(path), axis=0)
    if transpose: res = res.T
    if squeeze: res = res.squeeze()
    return res


def load_data_with_metadata(path: Union[str, Iterable], recursive: bool = False, max_workers=None, delimiter: str = '\t', transpose: bool = True, squeeze: bool = True, **kwargs) -> pd.DataFrame:
    """
    Load data and metadata from text files.

    Args:
        path (str): directory of files, zip file, or txt file
        recursive (bool, optional): read txt files in folder recursively
        max_workers (int, optional): maximum number of workers can be used
        delimiter (str, optional):  character used to separate the values
        transpose (bool, optional): transpose row and column
        squeeze (bool, optional): remove dimensions of length 1 

    Returns:
        out (DataFrame): File path, data read from the text files and unix time.
    """

    def _load(path) -> list:
        if isinstance(path, str):
            # read all txt files in directory
            if os.path.isdir(path):
                df = pd.DataFrame()
                df['path'] = list(map(lambda f: os.path.abspath(os.path.join(path, f)), glob.glob('**/*.txt', root_dir=path, recursive=True) if recursive else glob.glob('*.txt', root_dir=path, recursive=False)))
                try:
                    import datatable as dt
                    df['data'] = pd.Series(thread_executor.map(lambda f: np.array(f), dt.iread(df['path'].to_list())))
                except ImportError:
                    logging.info('Module datatable was not found. Use pandas instead.')
                    df['data'] = pd.Series(thread_executor.map(lambda f: np.array(pd.read_csv(f, sep=delimiter, header=None, engine='c')), df['path']))
                df['time'] = df['path'].apply(os.path.getmtime)
                return [df[['path', 'data', 'time']]]
            # read all txt files in zip file
            elif path.endswith('zip'):
                with ZipFile(path) as zf:
                    df = pd.DataFrame()
                    df['path'] = list(filter(lambda f: f.endswith('.txt') and ('/' not in f or recursive), zf.namelist()))
                    files = thread_executor.map(zf.read, df['path'])
                    try:
                        import datatable as dt
                        df['data'] = pd.Series(thread_executor.map(lambda f: np.array(f), dt.iread(list(files))))
                    except ImportError:
                        logging.info('Module datatable was not found. Use pandas instead.')
                        import io
                        df['data'] = pd.Series(thread_executor.map(lambda f: np.array(pd.read_csv(io.BytesIO(f), sep=delimiter, header=None, engine='c')), files))
                    df['time'] = df['path'].apply(lambda f: pd.Timestamp(*zf.getinfo(f).date_time).timestamp())
                return [df[['path', 'data', 'time']]]
            # read txt file
            elif path.endswith('.txt'):
                return [pd.DataFrame([[os.path.abspath(path), np.array(pd.read_csv(path, sep=delimiter, header=None, engine='c')), os.path.getmtime(path)]], columns=['path', 'data', 'time'])]
            # read all groups in tdms file
            elif path.endswith('tdms'):
                try:
                    from nptdms import TdmsFile
                    with TdmsFile.read(path) as f:
                        return list(thread_executor.map(lambda g: pd.DataFrame([[g.name, np.array(g.as_dataframe()), g.channels()[0].properties['wf_start_time']]], columns=['path', 'data', 'time']), f.groups()))
                except ImportError:
                    raise ImportError('Module nptdms was not found. TDMS files can not be read.')
        elif isinstance(path, Iterable):
            return sum(thread_executor.map(_load, path), start=[])

    res = None
    if isinstance(path, list):
        if len(path) == 1: path = path[0]
    if isinstance(path, str):
        if path.endswith('.txt'): res = pd.DataFrame([[os.path.abspath(path), np.array(pd.read_csv(path, sep=delimiter, header=None, engine='c')), os.path.getmtime(path)]], columns=['path', 'data', 'time'])
    if res is None:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as thread_executor:
            logging.debug(f'Use {thread_executor._max_workers} workers in load_data')
            res = pd.concat(_load(path), axis=0)
    if transpose: res['data'] = res['data'].apply(np.transpose)
    if squeeze: res['data'] = res['data'].apply(np.squeeze)
    res.reset_index(drop=True, inplace=True)
    return res


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

    def __init__(self, xlim: tuple[float, float], num_x_bin: int, xscale: Literal['linear', 'log'] = 'linear', *, fig: plt.Figure = None, ax: matplotlib.axes.Axes = None, figsize: tuple = None, **kwargs) -> None:
        self.x_min, self.x_max = sorted(xlim)
        self.x_bins = np.linspace(self.x_min, self.x_max, num_x_bin + 1) if xscale == 'linear' else np.logspace(np.log10(self.x_min), np.log10(self.x_max), num_x_bin + 1) if xscale == 'log' else None
        self.height, *_ = np.histogram([], self.x_bins)
        self.trace = 0
        self.xscale = xscale
        if fig: self.fig, self.ax = fig, ax
        else: self.fig, self.ax = plt.subplots(figsize=figsize) if figsize else plt.subplots()
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
        self.trace = self.trace + x.shape[0] if x.ndim == 2 else self.trace + 1
        self.height = self.height + np.histogram(x, self.x_bins)[0]
        height_per_trace = self.height_per_trace
        self.plot.set_data(height_per_trace)
        if set_ylim:
            self.ax.set_ylim(0, height_per_trace.max())

    def clear_data(self):
        self.trace = 0
        self.height.fill(0)
        self.plot.set_data(self.height)

    def fitting(self, x_range: list[float, float] = [-np.inf, np.inf], p0: list = None, bounds: list = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Args:
            x_range (list, optional): x range used to fit with height_per_trace, always in linear scale
            p0 (list, optional): Initial guess for the parameters in multi_gaussian function, use log scale if xscale=='log'
            bounds (list, optional): Lower and upper bounds on parameters

        Returns:
            ndarray: fitting results
            ndarray: fitting parameters (A, U, S)
        """
        f = np.where((self.x > min(x_range)) & (self.x < max(x_range)))
        x = self.x if self.xscale == 'linear' else np.log10(self.x)
        param = scipy.optimize.curve_fit(multi_gaussian, x[f], self.height_per_trace[f], p0=p0 or [1, 0, 1], bounds=bounds or [[0, -np.inf, 0], [np.inf, np.inf, np.inf]])[0]
        return multi_gaussian(x, *param), param

    def plot_fitting(self, x_range: list[float, float] = [-np.inf, np.inf], p0: list = None, bounds: list = None, *args, **kwargs) -> tuple[np.ndarray, np.ndarray, list[matplotlib.lines.Line2D]]:
        """
        Args:
            x_range (list, optional): x range used to fit with height_per_trace, always in linear scale
            p0 (list, optional): Initial guess for the parameters in multi_gaussian function, use log scale if xscale=='log'
            bounds (list, optional): Lower and upper bounds on parameters

        Returns:
            ndarray: fitting results
            ndarray: fitting parameters (A, U, S)
            list: matplotlib Line2D objects
        """
        fit, params = self.fitting(x_range, p0, bounds)
        return fit, params, self.ax.plot(self.x, fit, *args, **kwargs)


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

    def __init__(self, xlim: tuple[float, float], ylim: tuple[float, float], num_x_bin: int, num_y_bin: int, xscale: Literal['linear', 'log'] = 'linear', yscale: Literal['linear', 'log'] = 'linear', *, fig: plt.Figure = None, ax: matplotlib.axes.Axes = None, figsize: tuple = None, **kwargs) -> None:
        (self.x_min, self.x_max), (self.y_min, self.y_max) = sorted(xlim), sorted(ylim)
        self.x_bins = np.linspace(self.x_min, self.x_max, num_x_bin + 1) if xscale == 'linear' else np.logspace(np.log10(self.x_min), np.log10(self.x_max), num_x_bin + 1) if xscale == 'log' else None
        self.y_bins = np.linspace(self.y_min, self.y_max, num_y_bin + 1) if yscale == 'linear' else np.logspace(np.log10(self.y_min), np.log10(self.y_max), num_y_bin + 1) if yscale == 'log' else None
        self.height = np.histogram2d([], [], (self.x_bins, self.y_bins))[0]
        self.trace = 0
        self.xscale = xscale
        self.yscale = yscale
        if fig: self.fig, self.ax = fig, ax
        else: self.fig, self.ax = plt.subplots(figsize=figsize) if figsize else plt.subplots()
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

    def set_cmap(self, cmap: matplotlib.colors.LinearSegmentedColormap | dict):
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

    def fitting(self, axis: Literal['x', 'y'] = 'x', p0: list = None, bounds: list = None, sigma: float | list = 0, default_values: list[float] = [np.nan, np.nan, np.nan]) -> np.ndarray:
        """
        Args:
            axis (str): Select the slice of axis, use y and height_per_trace to fit if axis=='x'
            p0 (list, optional): Initial guess for the parameters in gaussian function, use log scale if xscale=='log'
            bounds (list, optional): Lower and upper bounds on parameters
            sigma (float|list, optional): standard derivative
            default_values (list, optional): values used when least-square minimization failed

        Returns:
            ndarray: fitting results with shape (sigma.size or None, x.size or y.size)
        """

        def gaussian_fit(label, xdata, ydata, p0, bounds):
            try:
                return scipy.optimize.curve_fit(gaussian, xdata, ydata, p0=p0, bounds=bounds)[0]
            except Exception as E:
                logging.warning(f'Least-square minimization failed at {axis}={label}')
                return default_values

        match axis:
            case 'x':
                A, U, S = np.array([gaussian_fit(self.x[ind], self.y if self.yscale == 'linear' else np.log10(self.y), z, p0=p0 or 0, bounds=bounds or [[0, -np.inf, 0], [np.inf, np.inf, np.inf]]) for ind, z in enumerate(self.height_per_trace)]).T
                return U + np.expand_dims(sigma, -1) * S if self.yscale == 'linear' else 10**(U + np.expand_dims(sigma, -1) * S)
            case 'y':
                A, U, S = np.array([gaussian_fit(self.y[ind], self.x if self.xscale == 'linear' else np.log10(self.x), z, p0=p0 or 0, bounds=bounds or [[0, -np.inf, 0], [np.inf, np.inf, np.inf]]) for ind, z in enumerate(self.height_per_trace.T)]).T
                return U + np.expand_dims(sigma, -1) * S if self.xscale == 'linear' else 10**(U + np.expand_dims(sigma, -1) * S)

    def plot_fitting(self, axis: Literal['x', 'y'] = 'x', p0: list = None, bounds: list = None, sigma: float = 0, default_values: list[float] = [np.nan, np.nan, np.nan], *args, **kwargs) -> tuple[np.ndarray, list[matplotlib.lines.Line2D]]:
        """
        Args:
            axis (str): Select the slice of axis, use y and height_per_trace to fit if axis=='x'
            p0 (list, optional): Initial guess for the parameters in gaussian function, use log scale if xscale=='log'
            bounds (list, optional): Lower and upper bounds on parameters
            sigma (float|list, optional): standard derivative
            default_values (list, optional): values used when least-square minimization failed

        Returns:
            ndarray: fitting results with shape (sigma.size or None, x.size or y.size)
            list: matplotlib Line2D objects
        """
        fit = self.fitting(axis, p0, bounds, sigma, default_values)
        match axis:
            case 'x':
                return fit, self.ax.plot(self.x, fit.T, *args, **kwargs)
            case 'y':
                return fit, self.ax.plot(self.y, fit.T, *args, **kwargs)
