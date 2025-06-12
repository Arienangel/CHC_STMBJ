import scipy.fft
import scipy.integrate

from .common import *


def extract_data(raw_data: Union[np.ndarray, str, list] = None,
                 length: int = 6000,
                 start_from: int = 4000,
                 zero_point: float = 0.5,
                 units: list = [1e-6, 1],
                 *,
                 I_raw: np.array = None,
                 V_raw: np.array = None,
                 **kwargs) -> np.ndarray:
    """
    Extract traces from raw_data

    Parameters
    ----------
    raw_data : Union[np.ndarray, str, list], optional
        1D G array or 2D array (I, V) contains raw data, directory of files, zip file, or txt file, by default None
    length : int, optional
        length of extracted data per trace, by default 6000
    start_from : int, optional
        length between zero_point and start of trace, by default 4000
    zero_point : float, optional
        set x=0 at G=zero_point, by default 0.5
    units : list, optional
        unit of I and V, by default [1e-6, 1]
    I_raw : np.array, optional
        raw current in 1D array, by default None
    V_raw : np.array, optional
        raw voltage in 1D array, by default None

    Returns
    -------
    G np.ndarray
        2D G array with shape (trace, length)
    """                 
    if raw_data is None: raw_data = conductance(I_raw * units[0], V_raw * units[1])
    elif not isinstance(raw_data, np.ndarray): raw_data = load_data(raw_data, **kwargs)
    if raw_data.shape[0] == 2: raw_data = conductance(*(raw_data[::-1] * np.expand_dims(units, 1)))
    if raw_data.size:
        index, *_ = scipy.signal.find_peaks(np.abs(np.gradient(np.where(raw_data > zero_point, 1, 0))), distance=start_from + length)
        index = index[raw_data[index] > raw_data[index + start_from + length]]
        return np.stack([raw_data[i + start_from:i + start_from + length] for i in filter(lambda i: i + start_from + length < raw_data.size, index)])
    return np.empty((0, length))


def PSD(G: np.ndarray, sampling_rate: float = 40000, *, return_freq: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate power spectral density by Fourier transform

    Parameters
    ----------
    G : np.ndarray
        2D G array with shape (trace, length)
    sampling_rate : float, optional
        points per second, by default 40000
    return_freq : bool, optional
        calculate frequency, by default True

    Returns
    -------
    PSD : np.array
        2D power spectral density
    freq : np.array, optional
        1D frequency of PSD
    """    
    dt = 1 / sampling_rate
    n = G.shape[1]
    t = n / sampling_rate
    if return_freq:
        return np.stack([(dt**2 / t) * np.abs(scipy.fft.fftshift(scipy.fft.fft(g)))**2 for g in G]), scipy.fft.fftshift(scipy.fft.fftfreq(n, dt))
    else:
        return np.stack([(dt**2 / t) * np.abs(scipy.fft.fftshift(scipy.fft.fft(g)))**2 for g in G])


def noise_power(PSD: np.ndarray, freq: np.ndarray, integrand: list = [100, 1000], *, int_method: Literal['trapezoid', 'simpson'] = 'trapezoid') -> np.ndarray:
    """
    Calculate noise power by integration of PSD

    Parameters
    ----------
    PSD : np.ndarray
        power spectral density
    freq : np.ndarray
        frequency of PSD
    integrand : list, optional
        integrand of ∫PSD df, by default [100, 1000]
    int_method : Literal[&#39;trapezoid&#39;, &#39;simpson&#39;], optional
        use trapezoidal rule or Simpson's rule to compute integral, by default 'trapezoid'

    Returns
    -------
    NP : np.ndarray
        1D noise power array
    """
    integrand = sorted(integrand)
    idx = (freq >= integrand[0]) & (freq <= integrand[1])
    if int_method == 'trapezoid':
        return 2 * scipy.integrate.trapezoid(x=freq[idx], y=PSD[:, idx], axis=1)  # include negative frequency
    elif int_method == 'simpson':
        return 2 * scipy.integrate.simpson(x=freq[idx], y=PSD[:, idx], axis=1)
    else:
        raise ValueError('Unknown integration method')


class Flicker_noise_data:
    """
    Flicker noise object

    Parameters
    ----------
    sampling_rate : float, optional
        points per second, by default 40000
    xscale : Literal[&#39;linear&#39;, &#39;log&#39;], optional
        linear or log scale of x axis, by default 'log'
    yscale : Literal[&#39;linear&#39;, &#39;log&#39;], optional
        linear or log scale of y axis, by default 'log'
    subplots_kw : tuple, optional
        plt.subplots kwargs, by default None
    int_method : Literal[&#39;trapezoid&#39;, &#39;simpson&#39;], optional
        _description_, by default 'trapezoid'
    """                 

    def __init__(self,
                 sampling_rate: float = 40000,
                 xscale: Literal['linear', 'log'] = 'log',
                 yscale: Literal['linear', 'log'] = 'log',
                 *,
                 fig: plt.Figure = None,
                 ax: matplotlib.axes.Axes = None,
                 subplots_kw: tuple = None,
                 int_method: Literal['trapezoid', 'simpson'] = 'trapezoid'):
        self.sampling_rate = sampling_rate
        self.xscale = xscale
        self.yscale = yscale
        self.int_method = int_method
        if any([fig, ax]): self.fig, self.ax = fig, ax
        else: self.fig, self.ax = plt.subplots(**subplots_kw) if subplots_kw else plt.subplots()
        self.ax.set_xlabel('Conductance ($G/G_0$)')
        self.ax.set_ylabel('Noise power')
        self.ax.set_xscale(xscale)
        self.ax.set_yscale(yscale)

    @property
    def trace(self):
        if self.ax.lines: return self.ax.lines[0].get_data()[0].shape[0]
        else: return 0

    def set_data(self, G: np.ndarray, integrand=[100, 1000], n: float = 0, auto_fit=False):
        """
        Set 2D G array into NP-G plot

        Parameters
        ----------
        G : np.ndarray
            2D G array with shape (trace, length)
        integrand : list, optional
            integrand of ∫PSD df, by default [100, 1000]
        n : float, optional
            noise power scaling exponent, by default 0
        auto_fit : bool, optional
            calculate n by fitting NP=c*G^n, by default False

        Returns
        -------
        n : float
            noise power scaling exponent
        """        
        self.PSD, self.freq = PSD(G, self.sampling_rate)
        self.NP = noise_power(self.PSD, self.freq, integrand, int_method=self.int_method)
        self.Gmean = G.mean(axis=1)
        return self.plot(self.Gmean, self.NP, n, auto_fit)

    def fit(self, Gmean: np.ndarray, NP: np.ndarray, *, return_c:bool=False) -> float:
        """
        Calculate n by fitting NP=c*G^n

        Parameters
        ----------
        Gmean : np.ndarray
            1D Gmean array
        NP : np.ndarray
            1D NP array
        return_c : bool, optional
            return fitting constant, 

        Returns
        -------
        n : float
            noise power scaling exponent
        c : float, optional
            fitting constant, by default False
        """        
        n, c = scipy.optimize.curve_fit(lambda x, n, c: n * x + c,
                                        Gmean if self.xscale == 'linear' else np.log10(np.abs(Gmean)),
                                        NP if self.yscale == 'linear' else np.log10(np.abs(NP)),
                                        bounds=[[0, -np.inf], [3, np.inf]])[0]
        if return_c: return n, c
        else: return n

    def plot(self, Gmean: np.ndarray, NP: np.ndarray, n: float = 0, auto_fit=False) -> float:
        """
        Set Gmean, NP array into NP-G plot

        Parameters
        ----------
        Gmean : np.ndarray
            1D Gmean array
        NP : np.ndarray
            1D noise power array
        n : float, optional
            noise power scaling exponent, by default 0
        auto_fit : bool, optional
            calculate n by fitting NP=c*G^n, by default False

        Returns
        -------
        n : float
            noise power scaling exponent
        """        
        if self.ax.lines: self.ax.lines[0].remove()
        if auto_fit: n = self.fit(Gmean, NP)
        y = NP / Gmean**n if n else NP
        self.ax.plot(Gmean, y, '.')
        self.ax.set_ylabel('Noise power / ($G/G_0)^{%.2f}$' % round(n, 2) if n else 'Noise power')
        return n

    def hist2d(
        self,
        xlim: tuple[float, float],
        ylim: tuple[float, float],
        num_x_bins: int,
        num_y_bins: int,
        *,
        fig: plt.Figure = None,
        ax: matplotlib.axes.Axes = None,
        subplots_kw: tuple = None,
        set_colorbar: bool = False,
        contour: bool = False,
    ):
        """
        _summary_

        Parameters
        ----------
        xlim : tuple[float, float]
            max and min value of x
        ylim : tuple[float, float]
            max and min value of y
        num_x_bins : int
            number of x bins
        num_y_bins : int
            number of y bins
        subplots_kw : tuple, optional
            plt.subplots kwargs, by default None
        set_colorbar : bool, optional
            add colorbar, by default False
        contour : bool, optional
            add contour by fitting 2D Gaussian distribution surface, by default False

        Returns
        -------
        fig : plt.Figure, optional
            Figure object
        ax : matplotlib.axes.Axes, optional
            Axes object
        h : np.ndarray
            2d histogram height array, x bin edges, y bin edges
        param : np.ndarray, optional
            gaussian2d fitting parameters (A, ux, uy, sx, sy, theta)
        """
        (x_min, x_max), (y_min, y_max) = sorted(xlim), sorted(ylim)
        x_bins = np.linspace(x_min, x_max, num_x_bins + 1) if self.xscale == 'linear' else np.logspace(np.log10(x_min), np.log10(x_max), num_x_bins + 1) if self.xscale == 'log' else None
        y_bins = np.linspace(y_min, y_max, num_y_bins + 1) if self.yscale == 'linear' else np.logspace(np.log10(y_min), np.log10(y_max), num_y_bins + 1) if self.yscale == 'log' else None
        if not any([fig, ax]): fig, ax = plt.subplots(**subplots_kw) if subplots_kw else plt.subplots()
        if len(self.ax.lines) == 0: raise ValueError('No data was set. Use set_data first.')
        h = ax.hist2d(*self.ax.lines[0].get_data(), (x_bins, y_bins), cmap=cmap)
        ax.set(xscale=self.xscale, yscale=self.yscale, xlabel=self.ax.get_xlabel(), ylabel=self.ax.get_ylabel())
        if set_colorbar: fig.colorbar(h[-1], ax=ax, shrink=0.5)
        if contour:
            x = (h[1][:-1] + h[1][1:]) / 2 if self.xscale == 'linear' else (h[1][:-1] * h[1][1:])**0.5
            y = (h[2][:-1] + h[2][1:]) / 2 if self.xscale == 'linear' else (h[2][:-1] * h[2][1:])**0.5
            x, y = np.meshgrid(x, y, indexing='ij')
            param = scipy.optimize.curve_fit(
                gaussian2d, (x.ravel() if self.xscale == 'linear' else np.log10(x).ravel(), y.ravel() if self.yscale == 'linear' else np.log10(y).ravel()),
                h[0].ravel(),
                bounds=[[0, min(xlim) if self.xscale == 'linear' else np.log10(min(xlim)),
                         min(ylim) if self.yscale == 'linear' else np.log10(min(ylim)), 0, 0, -np.pi / 4],
                        [np.inf, max(xlim) if self.xscale == 'linear' else np.log10(max(xlim)),
                         max(ylim) if self.yscale == 'linear' else np.log10(max(ylim)), np.inf, np.inf, np.pi / 4]])[0]
            ax.contour(x, y, gaussian2d((x if self.xscale == 'linear' else np.log10(x), y if self.yscale == 'linear' else np.log10(y)), *param), levels=4, colors='k', alpha=0.5)
            return fig, ax, h[:-1], param
        else:
            return fig, ax, h[:-1]
