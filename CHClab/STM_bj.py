from .common import *


def extract_data(raw_data: Union[np.ndarray, str, list] = None,
                 length: int = 1600,
                 upper: float = 3.2,
                 lower: float = 1e-6,
                 direction: Literal['pull', 'crash', 'both'] = 'pull',
                 check_size: tuple[int, int] = [10, 10],
                 cut_point: float = None,
                 offset: tuple[int, int] = [0, 0],*,
                 G_raw: np.ndarray = None,
                 X_raw: np.ndarray = None,
                 **kwargs) -> np.ndarray:
    """
    Extract traces from raw data

    Parameters
    ----------
    raw_data : Union[np.ndarray, str, list]
        1D array (G) contains raw data, directory of files, zip file, or txt file
    length : int, optional
        length of extracted data per trace, by default 1600
    upper : float, optional
        extract traces with data greater than upper limit, by default 3.2
    lower : float, optional
        extract traces with data less than lower limit, by default 1e-6
    direction : Literal[&#39;pull&#39;, &#39;crash&#39;, &#39;both&#39;], optional
        extract pull, crash or both direction, by default 'pull'
    check_size : tuple[int, int], optional
        length of data in the front or end of the trace used to determine the upper/lower limit, by default [10, 10]
    cut_point : float, optional
        used to find traces at G=cut_point, by default (upper * lower)**0.5
    offset : list, optional
        additional length before and after extracted traces, by default [0, 0]
    kwargs
        load_data kwargs
    G_raw : np.array, optional
        raw conductance in 1D array , by default None
    X_raw : np.array, optional
        raw dispolacement in 1D array, by default None
    Returns
    -------
    G : np.ndarray
        2D G array with shape (trace, length)
    X : np.ndarray, optional
        2D X array with shape (trace, length), return only when X_raw is given
    """
    if raw_data is not None:
        if not isinstance(raw_data, np.ndarray): raw_data = load_data(raw_data, **kwargs)
        G_raw = raw_data if raw_data.ndim == 1 else raw_data[0]
    if G_raw.size:
        if cut_point == None: cut_point = (upper * lower)**0.5
        index, *_ = scipy.signal.find_peaks(np.abs(np.gradient(np.where(G_raw > cut_point, 1, 0))), distance=length)
        if len(index):
            G_trace = np.stack([G_raw[i - length // 2 - offset[0]:i + length // 2 + offset[1]] for i in index if (i + length // 2 + offset[1] < G_raw.size) and (i - length // 2 + offset[0] >= 0)])
            total_length = length + sum(offset)
            match direction:
                case 'pull':
                    condition = (G_trace[:, offset[0]:offset[0] + check_size[0]] > upper).any(axis=1) & (G_trace[:, -check_size[1] - offset[1]:total_length - offset[1]] < lower).any(axis=1)
                case 'crash':
                    condition = (G_trace[:, offset[0]:offset[0] + check_size[0]] < lower).any(axis=1) & (G_trace[:, -check_size[1] - offset[1]:total_length - offset[1]] > upper).any(axis=1)
                case 'both':
                    condition = ((G_trace[:, offset[0]:offset[0] + check_size[0]] > upper).any(axis=1) & (G_trace[:, -check_size[1] - offset[1]:total_length - offset[1]] < lower).any(axis=1)) | (
                        (G_trace[:, offset[0]:offset[0] + check_size[0]] < lower).any(axis=1) & (G_trace[:, -check_size[1] - offset[1]:total_length - offset[1]] > upper).any(axis=1))
            if X_raw is not None:
                X_trace = np.stack([X_raw[i - length // 2 - offset[0]:i + length // 2 + offset[1]] for i in index if (i + length // 2 + offset[1] < G_raw.size) and (i - length // 2 + offset[0] >= 0)])
                return G_trace[condition], X_trace[condition]
            else:
                return G_trace[condition]
    return np.empty((0, length))


def get_displacement(G: np.ndarray, zero_point: float = 0.5, x_conversion: float = 800, **kwargs) -> np.ndarray:
    """
    Get displacement from conductance

    Parameters
    ----------
    G : np.ndarray
        2D G array with shape (trace, length)
    zero_point : float, optional
        set x=0 at G=zero_point, by default 0.5
    x_conversion : float, optional
        points per nm, calculated by sampling rate/ramp rate, by default 800

    Returns
    -------
    X : np.ndarray
        2D X array with shape (trace, length)
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

    Parameters
    ----------
    Glim : tuple[float, float], optional
        max and min value of G, by default (1e-5, 10**0.5)
    num_bins : float, optional
        number of bins, by default 550
    x_scale : Literal[&#39;linear&#39;, &#39;log&#39;], optional
        linear or log scale of x axis, by default 'log'
    xlabel : str, optional
        set xlabel, by default 'Conductance ($G/G_0$)'
    ylabel : str, optional
        set ylabel, by default 'Count/trace'
        
    kwargs
        Hist1D kwargs
    """

    def __init__(self,
                 Glim: tuple[float, float] = (1e-5, 10**0.5),
                 num_bins: float = 550,
                 x_scale: Literal['linear', 'log'] = 'log',
                 *,
                 xlabel: str = 'Conductance ($G/G_0$)',
                 ylabel: str = 'Count/trace',
                 **kwargs) -> None:
        super().__init__(Glim, num_bins, x_scale, xlabel=xlabel, ylabel=ylabel, **kwargs)

    def add_data(self, G: np.ndarray, set_ylim: bool = True, trace: int = None, **kwargs) -> None:
        """
        Add data into 1D conductance histogram

        Parameters
        ----------
        G : np.ndarray
            2D array with shape (trace, length), or 1D array
        set_ylim : bool, optional
            set largest y as y max, by default True
        trace : int, optional
            set custom #trace, by default None
        kwargs
            Hist1D.add_data kwargs
        """
        super().add_data(G, set_ylim, trace, **kwargs)


class Hist_GS(Hist2D):
    """
    2D conductance-displacement histogram

    Parameters
    ----------
    Xlim : tuple[float, float], optional
        max and min value of X, by default (-0.5, 0.5)
    Glim : tuple[float, float], optional
        max and min value of G, by default (1e-5, 10**0.5)
    num_X_bins : float, optional
        number of X bins, by default 500
    num_G_bins : float, optional
        number of G bins, by default 550
    xscale : Literal[&#39;linear&#39;, &#39;log&#39;], optional
        linear or log scale of x axis, by default 'linear'
    yscale : Literal[&#39;linear&#39;, &#39;log&#39;], optional
        linear or log scale of y axis, by default 'log'
    zero_point : float, optional
        set x=0 at G=zero_point, by default 0.5
    x_conversion : float, optional
        _description_, by default 800
    xlabel : str, optional
        set xlabel, by default 'Displacement (nm)'
    ylabel : str, optional
        set ylabel, by default 'Conductance ($G/G_0$)'
    colorbar_label : str, optional
        set colorbar label, by default 'Count/trace'
    kwargs
        Hist2D kwargs
    """

    def __init__(self,
                 Xlim: tuple[float, float] = (-0.5, 0.5),
                 Glim: tuple[float, float] = (1e-5, 10**0.5),
                 num_X_bins: float = 500,
                 num_G_bins: float = 550,
                 xscale: Literal['linear', 'log'] = 'linear',
                 yscale: Literal['linear', 'log'] = 'log',
                 zero_point: float = 0.5,
                 x_conversion: float = 800,
                 *,
                 xlabel: str = 'Displacement (nm)',
                 ylabel: str = 'Conductance ($G/G_0$)',
                 colorbar_label: str = 'Count/trace',
                 **kwargs) -> None:
        super().__init__(Xlim, Glim, num_X_bins, num_G_bins, xscale, yscale, xlabel=xlabel, ylabel=ylabel, colorbar_label=colorbar_label, **kwargs)
        self.zero_point = zero_point
        self.x_conversion = x_conversion

    def add_data(self, G: np.ndarray, X: np.ndarray = None, zero_point: float = None, set_clim: bool = True, **kwargs) -> np.ndarray:
        """
        Add data into 2D conductance-displacement histogram

        Parameters
        ----------
        G : np.ndarray
            2D G array with shape (trace, length)
        X : np.ndarray, optional
            2D X array with shape (trace, length), by default None
        zero_point : float, optional
            set x=0 at G=zero_point, by default self.x_conversion
        set_clim : bool, optional
            set largest z as z max, by default True
        kwargs
            Hist2D.add_data kwargs

        Returns
        -------
        X : np.ndarray
            2D X array with shape (trace, length)
        """
        if X is None:
            X = get_displacement(G, zero_point=zero_point or self.zero_point, x_conversion=self.x_conversion)
        super().add_data(X, G, set_clim, **kwargs)
        return X


class Hist_Gt(Hist2D):
    """
    2D conductance-time histogram

    Parameters
    ----------
    tlim : tuple[float, float], optional
        max and min value of t, by default (0, 3600)
    Glim : tuple[float, float], optional
        max and min value of G, by default (1e-5, 10**0.5)
    size_t_bins : float, optional
        t bin size in seconds, by default 30
    num_G_bins : float, optional
        number of G bins, by default 550
    xscale : Literal[&#39;linear&#39;, &#39;log&#39;], optional
        linear or log scale of x axis, by default 'linear'
    yscale : Literal[&#39;linear&#39;, &#39;log&#39;], optional
        linear or log scale of y axis, by default 'log'
    xlabel : str, optional
        set xlabel, by default 'Time (min)'
    ylabel : str, optional
        set ylabel, by default 'Conductance ($G/G_0$)'
    colorbar_label : str, optional
        set colorbar label, by default 'Count/trace'
    kwargs
        Hist2D kwargs
    """

    def __init__(self,
                 tlim: tuple[float, float] = (0, 3600),
                 Glim: tuple[float, float] = (1e-5, 10**0.5),
                 size_t_bins: float = 30,
                 num_G_bins: float = 550,
                 xscale: Literal['linear', 'log'] = 'linear',
                 yscale: Literal['linear', 'log'] = 'log',
                 *,
                 xlabel: str = 'Time (min)',
                 ylabel: str = 'Conductance ($G/G_0$)',
                 colorbar_label: str = 'Count/trace',
                 **kwargs) -> None:
        super().__init__(tlim, Glim, int(np.abs(tlim[1] - tlim[0]) // size_t_bins), num_G_bins, xscale, yscale, xlabel=xlabel, ylabel=ylabel, colorbar_label=colorbar_label, **kwargs)
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
        Add data into 2D conductance-time histogram

        Parameters
        ----------
        G : np.ndarray
            2D G array with shape (trace, length)
        t : np.ndarray, optional
            1D time array with shape (trace), or use interval to generate t, by default None
        set_clim : bool, optional
            set largest z as z max, by default True
        interval : float, optional
            time interval between each trace, by default 0.5
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

    Parameters
    ----------
    Glim : tuple[float, float], optional
        max and min value of G, by default (1e-5, 10**0.5)
    num_bins : float, optional
        number of G bins, by default 550
    x_scale : Literal[&#39;linear&#39;, &#39;log&#39;], optional
        linear or log scale of x axis, by default 'log'
    xlabel : str, optional
        set xlabel, by default 'Conductance ($G/G_0$)'
    colorbar_label : str, optional
        set colorbar label, by default None
    cmap : str, optional
        colormap, by default 'seismic'
    norm : _type_, optional
        colormap normalization, by default matplotlib.colors.SymLogNorm(0.1, 1, -1, 1)
    kwargs
        Hist2D kwargs
    """

    def __init__(self,
                 Glim: tuple[float, float] = (1e-5, 10**0.5),
                 num_bins: float = 550,
                 x_scale: Literal['linear', 'log'] = 'log',
                 *,
                 xlabel: str = 'Conductance ($G/G_0$)',
                 colorbar_label: str = None,
                 cmap='seismic',
                 norm=matplotlib.colors.SymLogNorm(0.1, 1, -1, 1),
                 **kwargs) -> None:
        super().__init__(Glim, Glim, num_bins, num_bins, x_scale, x_scale, xlabel=xlabel, ylabel=xlabel, colorbar_label=colorbar_label, **kwargs)
        if isinstance(cmap, dict):
            cmap = matplotlib.colors.LinearSegmentedColormap('Cmap', segmentdata=cmap, N=256)
        self.plot.set_cmap(cmap=cmap)
        self.plot.set_norm(norm=norm)

    def add_data(self, G: np.ndarray, **kwargs) -> None:
        """
        Add data into cross-correlation histogram

        Parameters
        ----------
        G : np.ndarray
            2D G array with shape (trace, length)
        """
        N = np.apply_along_axis(lambda g: np.histogram(g, self.x_bins)[0], 1, G)
        if hasattr(self, 'N'):
            self.N = np.concatenate([self.N, N], axis=0)
        else:
            self.N = N
        with np.errstate(divide='ignore', invalid='ignore'):
            self.height = np.corrcoef(self.N, rowvar=False)
        self.plot.set_array(self.height.T)
