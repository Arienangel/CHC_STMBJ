import atexit
import multiprocessing
import os
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
# import scipy.signal
import yaml
from watchdog.events import FileCreatedEvent, FileSystemEventHandler
from watchdog.observers import Observer

if os.path.exists('config.yaml'):
    with open('config.yaml', mode='r', encoding='utf-8') as f:
        conf = yaml.load(f, yaml.SafeLoader)
else:
    print('Config file not found.')
    exit()


class DAQ_Histogram:

    def __init__(self, G_max: float = 3.2, G_min: float = 1e-5, x_max: float = 1, x_min: float = -1, G_bins1: int = 1000, G_bins2: int = 1000, x_bins2: int = 1000, x_conversion=800) -> None:
        self.G_max, self.G_min, self.x_max, self.x_min = G_max, G_min, x_max, x_min
        self.G_bins1 = np.logspace(np.log10(self.G_min), np.log10(self.G_max), G_bins1 + 1)
        self.G_bins2 = np.logspace(np.log10(self.G_min), np.log10(self.G_max), G_bins2 + 1)
        self.x_conversion = x_conversion
        self.x_bins2 = np.linspace(self.x_min, self.x_max, x_bins2 + 1)
        self.fig, (self.ax1, self.ax2) = plt.subplots(ncols=2)
        self.ax1.set_xscale('log')
        self.ax1.set_xlim(left=self.G_min, right=self.G_max)
        self.ax1.set_xlabel('Conductance (G/G0)')
        self.ax1.set_ylabel('Count/trace')
        self.ax1.grid(visible=True, which='major')
        self.ax2.set_yscale('log')
        self.ax2.set_xlim(left=self.x_min, right=self.x_max)
        self.ax2.set_ylim(bottom=self.G_min, top=self.G_max)
        self.ax2.set_xlabel('Distance (nm)')
        self.ax2.set_ylabel('Conductance (G/G0)')

    @staticmethod
    def load_multiple_data(dir: str, threads: int = None) -> np.ndarray:
        '''
        Parameters
        ----------
        dir: data directory
        threads: #CPU threads
       
        Returns
        ----------
        raw_data: 1D array
        '''
        files = filter(lambda file: file.endswith('.txt'), os.listdir(dir))
        files = [os.path.join(dir, file) for file in files]
        with multiprocessing.Pool(threads or multiprocessing.cpu_count()) as pool:
            return np.concatenate(pool.map(np.loadtxt, files))

    @staticmethod
    def extract_data(raw_data: np.ndarray, length: int = 1000, upper: float = 2, lower: float = 1e-4, method: Literal['pull', 'crash', 'both'] = 'pull') -> tuple[np.ndarray, int]:
        '''
        Parameters
        ----------
        raw_data: 1D array

        Returns
        ----------
        extracted_data: 2D array (trace, length)
        trace: int
        '''
        split: np.ndarray = np.stack(np.split(raw_data, len(raw_data) / length), axis=0)
        res = split[np.where((split.max(axis=1) > upper) & (split.min(axis=1) < lower), True, False)]
        if method == 'pull': res = res[np.where(res[:, 0] > res[:, -1], True, False)]
        elif method == 'crash': res = res[np.where(res[:, 0] < res[:, -1], True, False)]
        elif method == 'both': pass
        return res, res.shape[0]

    def plot_init(self, extracted_data: np.ndarray = None, trace: int = 0):
        '''
        Parameters
        ----------
        extracted_data: 2D array
        trace: int
        '''
        self.trace = trace
        self.hist1 = self.ax1.stairs(np.zeros(self.G_bins1.size - 1), self.G_bins1, fill=True)
        self.hist2 = self.ax2.pcolormesh(self.x_bins2, self.G_bins2, np.zeros((self.G_bins2.size - 1, self.x_bins2.size - 1)), cmap='rainbow', vmin=0)
        self.fig.colorbar(self.hist2, ax=self.ax2, shrink=0.5)
        if trace > 0:
            self.plot(extracted_data, trace)

    def plot(self, add_data: np.ndarray, add_trace: int):
        '''
        Parameters
        ----------
        add_data: 2D array
        add_trace: int
        '''
        # ax1
        add_hist1, self.G_bins1 = np.histogram(add_data, self.G_bins1)
        new_hist1 = (self.hist1.get_data().values * self.trace + add_hist1) / (self.trace + add_trace)
        self.hist1.set_data(new_hist1)
        self.ax1.set_ylim(0, max(new_hist1) * 1.1)
        '''peak, *_ = scipy.signal.find_peaks(new_hist1, prominence=0.1)
        peak_position = self.G_bins1[peak], new_hist1[peak]
        self.ax1.plot(*peak_position, 'xr')
        for i, j in zip(*peak_position):
            self.ax1.annotate(f'{i:1.2E}', xy=(i, j+0.02), ha='center', fontsize=8)'''

        # ax2
        zero_point = np.argmin(np.abs(add_data - 0.5), axis=1)
        _, x = np.mgrid[:zero_point.size, :add_data.shape[-1]]
        x = (x - np.expand_dims(zero_point, axis=-1)) / self.x_conversion
        add_hist2, self.x_bins2, self.G_bins2 = np.histogram2d(x.ravel(), add_data.ravel(), (self.x_bins2, self.G_bins2))
        new_hist2 = self.hist2.get_array().reshape((self.G_bins2.size - 1, self.x_bins2.size - 1)) + add_hist2.T
        self.hist2.set_array(new_hist2)
        self.hist2.set_clim(0, new_hist2.max())
        self.trace = self.trace + add_trace


class DAQ_Run(FileSystemEventHandler):

    def __init__(self, dir: str, watchdog: bool = False) -> None:
        '''
        Parameters
        ----------
        dir: data directory
        watchdog: auto load new data when new file create
        '''

        super().__init__()
        self.dir = dir
        if watchdog:
            plt.ion()
            atexit.register(plt.close)
            self.__first_run()
            self.__watchdog()
            self.__keep_alive()
        else:
            self.__first_run()
            plt.show()

    def __first_run(self):
        self.plt = DAQ_Histogram(**conf['histogram'])
        if os.listdir(self.dir):
            raw_data = self.plt.load_multiple_data(self.dir, **conf['load_data'])
            extracted_data, trace = self.plt.extract_data(raw_data, **conf['extract_data'])
            self.plt.plot_init(extracted_data, trace)
        else:
            self.plt.plot_init()

    def __watchdog(self):
        observer = Observer()
        observer.schedule(self, path=self.dir, recursive=False)
        observer.start()
        atexit.register(observer.stop)

    def on_created(self, event):
        if isinstance(event, FileCreatedEvent):
            if (event.src_path.endswith('.txt')):
                try:
                    raw_data = np.loadtxt(event.src_path)
                    extracted_data, trace = self.plt.extract_data(raw_data, **conf['extract_data'])
                    self.plt.plot(extracted_data, trace)
                except Exception as E:
                    print(f'ERROR: {type(E).__name__}: {E.args}')

    @staticmethod
    def __keep_alive():
        try:
            while True:
                plt.pause(0.5)
        except KeyboardInterrupt:
            exit()


if __name__ == '__main__':
    DAQ_Run(dir=conf['dir'], watchdog=conf['watchdog'])
