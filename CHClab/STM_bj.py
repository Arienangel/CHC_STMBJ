from .common import *


def extract_data(raw_data: Union[np.ndarray, str, list],
                 length: int = 1000,
                 upper: float = 3.2,
                 lower: float = 1e-6,
                 method: Literal['pull', 'crash', 'both'] = 'pull',
                 check_size: tuple[int, int] = [10, 10],
                 cut_point: float = None,
                 **kwargs) -> np.ndarray:
    '''
    Extract useful data from raw_data

    Args:
        raw_data (ndarray | str): 1D array contains raw data, directory of files, zip file, or txt file
        length (int): length of extracted data per trace
        upper (float): extract data greater than upper limit
        lower (float): extract data less than lower limit
        method (str): 'pull', 'crash' or 'both'
        check_size (list, optional): length of data in the front or end of the trace used to determine the upper/lower limit
        cut_point (float, optional): used to find traces at G=cut_point. Default to (upper * lower)**0.5

    Returns:
        extracted_data (ndarray): 2D array with shape (trace, length)
    '''
    if not isinstance(raw_data, np.ndarray): raw_data = load_data(raw_data, **kwargs)
    if raw_data.size:
        if cut_point == None: cut_point = (upper * lower)**0.5
        index, *_ = scipy.signal.find_peaks(np.abs(np.gradient(np.where(raw_data > cut_point, 1, 0))), distance=length)
        if len(index):
            split_trace = np.stack(
                [raw_data[:length] if (i - length // 2) < 0 else raw_data[-length:] if (i + length // 2) > raw_data.size else raw_data[i - length // 2:i + length // 2] for i in index])
            match method:
                case 'pull':
                    return split_trace[(split_trace[:, :check_size[0]] > upper).any(axis=1) & (split_trace[:, -check_size[1]:] < lower).any(axis=1)]
                case 'crash':
                    return split_trace[(split_trace[:, :check_size[0]] < lower).any(axis=1) & (split_trace[:, -check_size[1]:] > upper).any(axis=1)]
                case 'both':
                    return split_trace[((split_trace[:, :check_size[0]] > upper).any(axis=1) & (split_trace[:, -check_size[1]:] < lower).any(axis=1)) |
                                       ((split_trace[:, :check_size[0]] < lower).any(axis=1) & (split_trace[:, -check_size[1]:] > upper).any(axis=1))]
    return np.empty((0, length))


def get_displacement(G: np.ndarray, zero_point: float = 0.5, x_conversion: float = 800, **kwargs) -> np.ndarray:
    """
    Get displacement from conductance array

    Args:
        G (ndarray): 2D G array with shape (trace, length)
        zero_point (float): set x=0 at G=zero_point
        x_conversion (float, optional): points per displacement

    Returns:
        X (ndarray): 2D X array with shape (trace, length)
    """
    if G.ndim == 1: G = np.expand_dims(G, 0)
    is_pull = G[:, 0] > G[:, -1]
    _, X = np.mgrid[:G.shape[0], :G.shape[-1]]
    row, col = np.where(((G[:, :-1] > zero_point) & (G[:, 1:] < zero_point) & np.expand_dims(is_pull, axis=-1))
                        | ((G[:, :-1] < zero_point) & (G[:, 1:] > zero_point) & np.expand_dims(~is_pull, axis=-1)))
    x_min, x_max = pd.DataFrame([row, col]).transpose().groupby(0).agg(['min', 'max']).values.T
    x1 = np.where(is_pull, x_max, x_min)
    x2 = x1 + 1
    y1, y2 = G[:, x1].diagonal(), G[:, x2].diagonal()
    x = x1 + (zero_point - y1) / (y2 - y1) * (x2 - x1)
    return (X - np.expand_dims(x, axis=-1)) / x_conversion


class Hist_G(Hist1D):
    """
    1D conductance histogram

    Args:
        Glim (tuple): max and min value of G
        num_bins (float): number of bins
        x_scale (str): linear or log scale of x axis
        kwargs (dict, optional): Hist1D kwargs

    Attributes:
        trace (int): number of traces
        bins (ndarray): 1D array of bin edges
        height (ndarray): height of the histogram
        fig (Figure): plt.Figure object
        ax (Axes): plt.Axes object
        plot (StepPatch): plt.stairs object
    """

    def __init__(self, Glim: tuple[float, float] = (1e-5, 10**0.5), num_bins: float = 550, x_scale: Literal['linear', 'log'] = 'log', **kwargs) -> None:
        super().__init__(Glim, num_bins, x_scale, **kwargs)
        self.ax.set_xlabel('Conductance ($G/G_0$)')
        self.ax.set_ylabel('Count/trace')

    def add_data(self, G: np.ndarray, **kwargs) -> None:
        """
        Add data into 1D histogram (G)

        Args:
            G (ndarray): 2D array with shape (trace, length)
        """
        super().add_data(G, **kwargs)


class Hist_GS(Hist2D):
    """
    2D conductance-displacement histogram

    Args:
        Xlim (tuple): max and min value of X
        Glim (tuple): max and min value of G
        num_X_bins (float): number of X bins
        num_G_bins (float): number of G bins
        xscale (str): linear or log scale of x axis
        yscale (str): linear or log scale of y axis
        zero_point (float): set x=0 at G=zero_point
        x_conversion (float, optional): points per nm
        kwargs (dict, optional): Hist2D kwargs

    Attributes:
        trace (int): number of traces
        x_bins (ndarray): 1D array of x bin edges
        y_bins (ndarray): 1D array of y bin edges
        height (ndarray): height of the histogram
        fig (Figure): plt.Figure object
        ax (Axes): plt.Axes object
        plot (QuadMesh): plt.pcolormesh object
    """

    def __init__(self,
                 Xlim: tuple[float, float] = (-0.3, 0.5),
                 Glim: tuple[float, float] = (1e-5, 10**0.5),
                 num_X_bins: float = 800,
                 num_G_bins: float = 550,
                 xscale: Literal['linear', 'log'] = 'linear',
                 yscale: Literal['linear', 'log'] = 'log',
                 zero_point: float = 0.5,
                 x_conversion: float = 800,
                 **kwargs) -> None:
        super().__init__(Xlim, Glim, num_X_bins, num_G_bins, xscale, yscale, **kwargs)
        self.ax.set_xlabel('Displacement (nm)')
        self.ax.set_ylabel('Conductance ($G/G_0$)')
        if hasattr(self, 'colorbar'): self.colorbar.set_label('Count/trace')
        self.zero_point = zero_point
        self.x_conversion = x_conversion

    def add_data(self, G: np.ndarray, X: np.ndarray = None, zero_point: float = None, set_clim: bool = True, **kwargs) -> None:
        """
        Add data into 2D histogram (GS)

        Args:
            G (ndarray): 2D G array with shape (trace, length)
            X (ndarray, optional): 2D X array with shape (trace, length)
            zero_point (float, optional): set x=0 at G=zero_point, overwrite attribute
            set_clim (bool, optional): set largest z as z max
            
        Returns:
            X (ndarray): 2D X array with shape (trace, length)
        """
        if X is None:
            X = get_displacement(G, zero_point=zero_point or self.zero_point, x_conversion=self.x_conversion)
        super().add_data(X, G, set_clim, **kwargs)
        return X


class Hist_Gt(Hist2D):
    """
    Conductance-time histogram

    Args:
        tlim (tuple): max and min value of t
        Glim (tuple): max and min value of G
        size_t_bins (float): t bin size in seconds
        num_G_bins (float): number of G bins
        xscale (str): linear or log scale of x axis
        yscale (str): linear or log scale of y axis
        kwargs (dict, optional): Hist2D kwargs

    Attributes:
        trace (int): number of traces
        x_bins (ndarray): 1D array of x bin edges
        y_bins (ndarray): 1D array of y bin edges
        height (ndarray): height of the histogram
        fig (Figure): plt.Figure object
        ax (Axes): plt.Axes object
        plot (QuadMesh): plt.pcolormesh object
    """

    def __init__(self,
                 tlim: tuple[float, float] = (0, 3600),
                 Glim: tuple[float, float] = (1e-5, 10**0.5),
                 size_t_bins: float = 30,
                 num_G_bins: float = 550,
                 xscale: Literal['linear', 'log'] = 'linear',
                 yscale: Literal['linear', 'log'] = 'log',
                 **kwargs) -> None:
        super().__init__(tlim, Glim, int(np.abs(tlim[1] - tlim[0]) // size_t_bins), num_G_bins, xscale, yscale, **kwargs)
        self.ax.set_xlabel('Time (min)')
        self.ax.set_ylabel('Conductance ($G/G_0$)')
        if hasattr(self, 'colorbar'): self.colorbar.set_label('Count/trace')
        self.trace = np.zeros(self.x.size)
        self.ax.set_xticks(np.arange(0, self.x_max, 600), np.arange(0, self.x_max / 60, 10, dtype=int))
        self.ax.set_xticks(np.arange(0, self.x_max, 60), minor=True)
        self.ax.xaxis.grid(visible=True, which='major')

    @property
    def height_per_trace(self):
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.nan_to_num(np.divide(self.height, np.expand_dims(self.trace, axis=1)))

    def add_data(self, G: np.ndarray, t: np.ndarray = None, set_clim: bool = True, *, interval: float = 0.5, **kwargs) -> None:
        """
        Add data into 2D histogram

        Args:
            G (ndarray): 2D G array with shape (trace, length)
            t (ndarray): 1D time array with shape (trace)
            interval (float, optional): time interval between each trace
        """
        if t is None: t = np.arange(G.shape[0]) * interval
        self.trace = self.trace + np.histogram(t, self.x_bins)[0]
        self.height = self.height + np.histogram2d(np.tile(t, (G.shape[1], 1)).T.ravel(), G.ravel(), (self.x_bins, self.y_bins))[0]
        height_per_trace = self.height_per_trace
        self.plot.set_array(height_per_trace.T)
        if set_clim:
            self.plot.set_clim(0, height_per_trace.max())


class Hist_Correlation(Hist2D):
    """
    Cross-correlation conductance histogram

    Args:
        Glim (tuple): max and min value of G
        num_bins (float): number of bins
        x_scale (str): linear or log scale of x axis
        cmap (optional): plt cmap
        norm (optional): plt norm
        kwargs (dict, optional): Hist2D kwargs

    Attributes:
        trace (int): number of traces
        bins (ndarray): 1D array of bin edges
        height (ndarray): height of the histogram
        fig (Figure): plt.Figure object
        ax (Axes): plt.Axes object
        plot (StepPatch): plt.stairs object
    """

    def __init__(self,
                 Glim: tuple[float, float] = (1e-5, 10**0.5),
                 num_bins: float = 550,
                 x_scale: Literal['linear', 'log'] = 'log',
                 cmap='seismic',
                 norm=matplotlib.colors.SymLogNorm(0.1, 1, -1, 1),
                 **kwargs) -> None:
        super().__init__(Glim, Glim, num_bins, num_bins, x_scale, x_scale, **kwargs)
        self.ax.set_xlabel('Conductance ($G/G_0$)')
        self.ax.set_ylabel('Conductance ($G/G_0$)')
        if isinstance(cmap, dict):
            cmap = matplotlib.colors.LinearSegmentedColormap('Cmap', segmentdata=cmap, N=256)
        self.plot.set_cmap(cmap=cmap)
        self.plot.set_norm(norm=norm)

    def add_data(self, G: np.ndarray, **kwargs) -> None:
        """
        Add data into 2D histogram

        Args:
            G (ndarray): 2D G array with shape (trace, length)
        """
        N = np.apply_along_axis(lambda g: np.histogram(g, self.x_bins)[0], 1, G)
        if hasattr(self, 'N'):
            self.N = np.concatenate([self.N, N], axis=0)
        else:
            self.N = N
        with np.errstate(divide='ignore', invalid='ignore'):
            self.height = np.corrcoef(self.N, rowvar=False)
        self.plot.set_array(self.height.T)
