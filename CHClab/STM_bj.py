from .common import *


def extract_data(raw_data: Union[np.ndarray, str, list], length: int = 1000, upper: float = 3.2, lower: float = 1e-6, method: Literal['pull', 'crash', 'both'] = 'pull', offset: tuple[float, float] = (10, 10), **kwargs) -> np.ndarray:
    '''
    Extract useful data from raw_data

    Args:
        raw_data (ndarray | str): 1D array contains raw data, directory of files, zip file, or txt file
        length (int): length of extracted data per trace
        upper (float): extract data greater than upper limit
        lower (float): extract data less than lower limit
        method (str): 'pull', 'crash' or 'both'
        limit_offset (tuple): length of data in the front or end of the trace used to determine the upper/lower limit

    Returns:
        extracted_data (ndarray): 2D array with shape (trace, length)
    '''
    if not isinstance(raw_data, np.ndarray): raw_data = load_data(raw_data, **kwargs)
    if raw_data.size:
        index, *_ = scipy.signal.find_peaks(np.abs(np.gradient(np.where(raw_data > (upper * lower)**0.5, 1, 0))), distance=length)
        if len(index):
            split_trace = np.stack([raw_data[:length] if (i - length // 2) < 0 else raw_data[-length:] if (i + length // 2) > raw_data.size else raw_data[i - length // 2:i + length // 2] for i in index])
            match method:
                case 'pull':
                    return split_trace[(split_trace[:, :offset[0]] > upper).any(axis=1) & (split_trace[:, -offset[1]:] < lower).any(axis=1)]
                case 'crash':
                    return split_trace[(split_trace[:, :offset[0]] < lower).any(axis=1) & (split_trace[:, -offset[1]:] > upper).any(axis=1)]
                case 'both':
                    return split_trace[((split_trace[:, :offset[0]] > upper).any(axis=1) & (split_trace[:, -offset[1]:] < lower).any(axis=1)) | ((split_trace[:, :offset[0]] < lower).any(axis=1) & (split_trace[:, -offset[1]:] > upper).any(axis=1))]
    return np.empty((0, length))


def get_displacement(G: np.ndarray, zero_point: float = 0.5, x_conversion: float = 1, **kwargs) -> np.ndarray:
    """
    Get displacement from conductance array

    Args:
        G (ndarray): 2D G array with shape (trace, length)
        zero_point (float): set x=0 at G=zero_point
        x_conversion (float, optional): point per displacement

    Returns:
        x (ndarray): 2D x array with shape (trace, length)
    """
    is_pull = G[:, 0] > G[:, -1]
    _, X = np.mgrid[:G.shape[0], :G.shape[-1]]
    row, col = np.where(((G[:, :-1] > zero_point) & (G[:, 1:] < zero_point) & np.expand_dims(is_pull, axis=-1)) | ((G[:, :-1] < zero_point) & (G[:, 1:] > zero_point) & np.expand_dims(~is_pull, axis=-1)))
    x_min, x_max = pd.DataFrame([row, col]).transpose().groupby(0).agg(['min', 'max']).values.T
    x1 = np.where(is_pull, x_max, x_min)
    x2 = x1 + 1
    y1, y2 = G[:, x1].diagonal(), G[:, x2].diagonal()
    x = x1 + (zero_point - y1) / (y2 - y1) * (x2 - x1)
    return (X - np.expand_dims(x, axis=-1)) / x_conversion


def get_correlation_matrix(G: np.ndarray, bins: np.ndarray):
    N = np.apply_along_axis(lambda g: np.histogram(g, bins)[0], 1, G)
    return np.corrcoef(N, rowvar=False)


class Hist_G(Hist1D):

    def __init__(self, xlim: tuple[float, float] = (1e-5, 10**0.5), num_bin: float = 550, x_scale: Literal['linear', 'log'] = 'log', **kwargs) -> None:
        super().__init__(xlim, num_bin, x_scale, **kwargs)
        self.ax.set_xlabel('Conductance ($G/G_0$)')
        self.ax.set_ylabel('Count/trace')

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
            height (ndarray): peak height
            avg (ndarray): average
            stdev (ndarray): standard derivative
        """
        X = np.sqrt(self.x_bins[:-1] * self.x_bins[1:])
        Y = self.height
        return get_peak(X, Y, window_length=window_length, polyorder=polyorder, prominence=prominence)


class Hist_GS(Hist2D):

    def __init__(self,
                 xlim: tuple[float, float] = (-0.3, 0.5),
                 ylim: tuple[float, float] = (1e-5, 10**0.5),
                 num_x_bin: float = 800,
                 num_y_bin: float = 550,
                 xscale: Literal['linear', 'log'] = 'linear',
                 yscale: Literal['linear', 'log'] = 'log',
                 zero_point: float = 0.5,
                 x_conversion: float = 800,
                 **kwargs) -> None:
        super().__init__(xlim, ylim, num_x_bin, num_y_bin, xscale, yscale, **kwargs)
        self.ax.set_xlabel('Displacement (nm)')
        self.ax.set_ylabel('Conductance ($G/G_0$)')
        self.colorbar.set_label('Count/trace')
        self.zero_point = zero_point
        self.x_conversion = x_conversion

    def add_data(self, G: np.ndarray, **kwargs) -> None:
        """
        Add data into 2D histogram (GS)

        Args:
            G (ndarray): 2D G array with shape (trace, length)
        """
        x = get_displacement(G, zero_point=self.zero_point, x_conversion=self.x_conversion)
        super().add_data(x, G, **kwargs)


class Hist_Gt(Hist2D):

    def __init__(self, xlim: tuple[float, float] = (0, 3600), ylim: tuple[float, float] = (1e-5, 10**0.5), size_x_bin: float = 30, num_y_bin: float = 550, xscale: Literal['linear', 'log'] = 'linear', yscale: Literal['linear', 'log'] = 'log', **kwargs) -> None:
        super().__init__(xlim, ylim, int(np.abs(xlim[1] - xlim[0]) // size_x_bin), num_y_bin, xscale, yscale, **kwargs)
        self.ax.set_xlabel('Time (min)')
        self.ax.set_ylabel('Conductance ($G/G_0$)')
        self.colorbar.set_label('Count/trace')
        self.trace = np.zeros(self.x.size)
        self.ax.set_xticks(np.arange(0, self.x_max, 600), np.arange(0, self.x_max / 60, 10, dtype=int))
        self.ax.set_xticks(np.arange(0, self.x_max, 60), minor=True)
        self.ax.xaxis.grid(visible=True, which='major')

    @property
    def height_per_trace(self):
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.nan_to_num(np.divide(self.height, np.expand_dims(self.trace, axis=1)))

    def add_data(self, t: np.ndarray, G: np.ndarray, set_clim: bool = True, **kwargs) -> None:
        """
        Add data into 2D histogram

        Args:
            t (ndarray): 1D time array with shape (trace)
            G (ndarray): 2D G array with shape (trace, length)
        """
        self.trace = self.trace + np.histogram(t, self.x_bins)[0]
        self.height = self.height + np.histogram2d(np.tile(t, (G.shape[1], 1)).T.ravel(), G.ravel(), (self.x_bins, self.y_bins))[0]
        height_per_trace = self.height_per_trace
        self.plot.set_array(height_per_trace.T)
        if set_clim:
            self.plot.set_clim(0, height_per_trace.max())


class Hist_Correlation(Hist2D):

    def __init__(self, xlim: tuple[float, float] = (1e-5, 10**0.5), num_bin: float = 550, x_scale: Literal['linear', 'log'] = 'log', **kwargs) -> None:
        super().__init__(xlim, xlim, num_bin, num_bin, x_scale, x_scale, **kwargs)
        self.ax.set_xlabel('Conductance ($G/G_0$)')
        self.ax.set_ylabel('Conductance ($G/G_0$)')
        self.plot.set_cmap(cmap='seismic', )
        self.plot.set_norm(norm=matplotlib.colors.SymLogNorm(0.1, 1, -1, 1))

    def add_data(self, G: np.ndarray, **kwargs) -> None:
        N = np.apply_along_axis(lambda g: np.histogram(g, self.x_bins)[0], 1, G)
        if hasattr(self, 'N'):
            self.N = np.concatenate([self.N, N], axis=0)
        else:
            self.N = N
        with np.errstate(divide='ignore',invalid='ignore'): 
            self.height = np.corrcoef(self.N, rowvar=False)
        self.plot.set_array(self.height.T)
