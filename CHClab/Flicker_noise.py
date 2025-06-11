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
    '''
    Extract traces from raw_data

    Args:
        raw_data (ndarray | str): 1D G array or 2D array (I, V) contains raw data, directory of files, zip file, or txt file
        I_raw (ndarray, optional): raw current in 1D array
        V_raw (ndarray, optional): raw voltage in 1D array
        length (int): length of extracted data per trace
        start_from (int): length between zero_point and start of trace
        zero_point (float): set x=0 at G=zero_point

    Returns:
        I (ndarray): current (A) in 2D array (#traces, length)
        V (ndarray): voltage (V) in 2D array (#traces, length)
    '''
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
    Args:
        G (np.ndarray): 2D G array with shape (trace, length)
        sampling_rate (float, optional)

    Returns:
        PSD (np.ndarray): power spectral density
        freq (np.ndarray): sample frequency of PSD
    """
    dt = 1 / sampling_rate
    n = G.shape[1]
    t = n / sampling_rate
    if return_freq:
        return np.stack([(dt**2 / t) * np.abs(scipy.fft.fftshift(scipy.fft.fft(g)))**2 for g in G]), scipy.fft.fftshift(scipy.fft.fftfreq(n, dt))
    else:
        return np.stack([(dt**2 / t) * np.abs(scipy.fft.fftshift(scipy.fft.fft(g)))**2 for g in G])


def noise_power(PSD: np.ndarray, freq: np.ndarray, integrand: list = [100, 1000]) -> np.ndarray:
    """
    Args:
        PSD (np.ndarray): power spectral density
        freq (np.ndarray): sample frequency of PSD
        interval (list, optional): integrand of df. Defaults to 100~1000 Hz.

    Returns:
        NP (np.ndarray): noise power
    """
    integrand = sorted(integrand)
    idx = (freq >= integrand[0]) & (freq <= integrand[1])
    return 2 * scipy.integrate.trapezoid(PSD[:, idx], freq[idx], axis=1)  # include negative frequency


class Flicker_noise_data:
    """
    Flicker noise object

    Args:
        sampling_rate (tuple): points per second
        xscale (str): linear or log scale of x axis
        yscale (str): linear or log scale of y axis
        kwargs (dict, optional): Hist2D kwargs

    Attributes:
        trace (int): number of traces
        fig (Figure): plt.Figure object
        ax (Axes): plt.Axes object
    """

    def __init__(self,
                 sampling_rate: float = 40000,
                 xscale: Literal['linear', 'log'] = 'log',
                 yscale: Literal['linear', 'log'] = 'log',
                 *,
                 fig: plt.Figure = None,
                 ax: matplotlib.axes.Axes = None,
                 figsize: tuple = None):
        self.sampling_rate = sampling_rate
        self.xscale = xscale
        self.yscale = yscale
        if any([fig, ax]): self.fig, self.ax = fig, ax
        else: self.fig, self.ax = plt.subplots(figsize=figsize) if figsize else plt.subplots()
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
        Set data into NP-G plot

        Args:
            G (ndarray): 2D G array with shape (trace, length)
            integrand (list): integrand of df. Defaults to 100~1000 Hz.
            n (float, optional): noise power scaling exponent

        Returns:
            n (float): noise power scaling exponent
        """

        psd, freq = PSD(G, self.sampling_rate)
        np = noise_power(psd, freq, integrand)
        Gmean = G.mean(axis=1)
        return self.plot(Gmean, np, n, auto_fit)

    def fit(self, Gmean: np.ndarray, NP: np.ndarray) -> float:
        """
        Args:
            Gmean (np.ndarray): 1D G array
            NP (np.ndarray): 1D NP array

        Returns:
            n (float): noise power scaling exponent
        """
        n, c = scipy.optimize.curve_fit(lambda x, n, c: n * x + c,
                                        Gmean if self.xscale == 'linear' else np.log10(np.abs(Gmean)),
                                        NP if self.yscale == 'linear' else np.log10(np.abs(NP)),
                                        bounds=[[0, -np.inf], [3, np.inf]])[0]
        return n

    def plot(self, Gmean: np.ndarray, NP: np.ndarray, n: float = 0, auto_fit=False) -> float:
        """
        Args:
            Gmean (np.ndarray): 1D G array
            NP (np.ndarray): 1D NP array
            n (float, optional): noise power scaling exponent
            auto_fit (bool, optional): auto fit n

        Returns:
            n (float): noise power scaling exponent
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
    ):
        (x_min, x_max), (y_min, y_max) = sorted(xlim), sorted(ylim)
        x_bins = np.linspace(x_min, x_max, num_x_bins + 1) if self.xscale == 'linear' else np.logspace(np.log10(x_min), np.log10(x_max), num_x_bins + 1) if self.xscale == 'log' else None
        y_bins = np.linspace(y_min, y_max, num_y_bins + 1) if self.yscale == 'linear' else np.logspace(np.log10(y_min), np.log10(y_max), num_y_bins + 1) if self.yscale == 'log' else None
        if not any([fig, ax]): fig, ax = plt.subplots(**subplots_kw) if subplots_kw else plt.subplots()
        if len(self.ax.lines) == 0: raise ValueError('No data was set. Use set_data first.')
        h = ax.hist2d(*self.ax.lines[0].get_data(), (x_bins, y_bins), cmap=cmap)[-1]
        ax.set(xscale=self.xscale, yscale=self.yscale, xlabel=self.ax.get_xlabel(), ylabel=self.ax.get_ylabel())
        if set_colorbar: fig.colorbar(h, ax=ax, shrink=0.5)
        return fig, ax
