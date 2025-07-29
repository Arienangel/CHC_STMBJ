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
    if G.ndim==1:
        G=np.expand_dims(G, 0)
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
        use trapezoidal rule or Simpson's rule to compute integral, by default 'trapezoid'
    n : float, optional
        noise power scaling exponent, default value for LineNP and HistNP object, by default 0
    scatter : bool, optional
        create NP-Gmean scatter plot, by default False
    hist : bool, optional
        create NP-Gmean histogram, by default False
    """

    def __init__(self,
                 sampling_rate: float = 40000,
                 xscale: Literal['linear', 'log'] = 'log',
                 yscale: Literal['linear', 'log'] = 'log',
                 *,
                 int_method: Literal['trapezoid', 'simpson'] = 'trapezoid',
                 n: float = 0,
                 scatter: bool = False,
                 hist: bool = False):
        self.sampling_rate = sampling_rate
        self.xscale = xscale
        self.yscale = yscale
        self.int_method = int_method
        self.Gmean = np.array([])
        self.NP = np.array([])
        self.n = n
        self.scatter = self.get_scatter() if scatter else None
        self.hist = self.get_hist() if hist else None

    def get_scatter(self, xlabel: str = 'Conductance ($G/G_0$)', ylabel: str = 'Noise power', n: float = None, auto_fit: bool = False, **kwargs):
        """
        Create NP-Gmean scatter plot

        Parameters
        ----------
        xlabel : str, optional
            set xlabel, by default 'Conductance (/G_0$)'
        ylabel : str, optional
            set ylabel, by default 'Noise power'
        n : float, optional
            noise power scaling exponent, by default self.n
        auto_fit : bool, optional
            calculate n by fitting NP=c*G^n, by default False
        kwargs
            LineNP kwargs

        Returns
        -------
        LineNP
            LineNP object
        """
        if n is None: n=self.n
        self.scatter = LineNP(xscale=self.xscale, yscale=self.yscale, xlabel=xlabel, ylabel=ylabel, n=n, **kwargs)
        if self.Gmean.size:
            self.scatter.add_data(self.Gmean, self.NP, auto_fit=auto_fit)
        return self.scatter

    def get_hist(self,
                 Glim: tuple[float, float] = (1e-5, 1),
                 NPlim: tuple[float, float] = (1e-7, 1e-2),
                 num_G_bins: float = 100,
                 num_NP_bins: float = 100,
                 xlabel: str = 'Conductance ($G/G_0$)',
                 ylabel: str = 'Noise power',
                 set_colorbar: bool = False,
                 n: float = None,
                 auto_fit: bool = False,
                 **kwargs):
        """
        Create NP-Gmean histogram

        Parameters
        ----------
        Glim : tuple[float, float], optional
            max and min value of G, by default (-0.5, 0.5)
        NPlim : tuple[float, float], optional
            max and min value of NP, by default (1e-5, 10**0.5)
        num_G_bins : float, optional
            number of G bins, by default 500
        num_NP_bins : float, optional
            number of NP bins, by default 550
        xlabel : str, optional
            set xlabel, by default 'Conductance (/G_0$)'
        ylabel : str, optional
            set ylabel, by default 'Noise power'
        set_colorbar : bool, optional
            add colorbar, by default False
        n : float, optional
            noise power scaling exponent, by default self.n
        auto_fit : bool, optional
            calculate n by minimized gaussian2d theta, by default False
        kwargs
            HistNP kwargs

        Returns
        -------
        HistNP
            HistNP object
        """
        if n is None: n=self.n
        self.hist = HistNP(Glim, NPlim, num_G_bins, num_NP_bins, self.xscale, self.yscale, xlabel=xlabel, ylabel=ylabel, set_colorbar=set_colorbar, n=n, **kwargs)
        if self.Gmean.size:
            self.hist.add_data(self.Gmean, self.NP, auto_fit=auto_fit)
        return self.hist

    @property
    def trace(self):
        return self.Gmean.size

    def add_data(self,
                 G: np.ndarray = None,
                 integrand=[100, 1000],
                 n: float = None,
                 auto_fit: bool = False,
                 Gmean: np.ndarray = None,
                 NP: np.ndarray = None,
                 *args,
                 int_method: Literal['trapezoid', 'simpson'] = None,
                 update: bool = True,
                 **kwargs):
        """
        Add NP and Gmean data or calculate NP and Gmean from G

        Parameters
        ----------
        G : np.ndarray
            2D G array with shape (trace, length)
        integrand : list, optional
            integrand of ∫PSD df, by default [100, 1000]
        n : float, optional
            noise power scaling exponent, by default self.scatter.n or self.hist.n
        auto_fit : bool, optional
            auto fit LineNP, by default False
        Gmean : np.ndarray
            1D Gmean array
        NP : np.ndarray
            1D noise power array
        int_method : Literal[&#39;trapezoid&#39;, &#39;simpson&#39;], optional
            use trapezoidal rule or Simpson's rule to compute integral, by default 'trapezoid'
        update : bool, optional
            update scatter plot and histogram, by default True
        args
            Axes.plot args
        kwargs
            Axes.plot kwargs
        """
        if NP is None: NP = noise_power(*PSD(G, self.sampling_rate), integrand, int_method=int_method or self.int_method)
        if Gmean is None: Gmean = G.mean(axis=1)
        self.NP = np.concatenate([self.NP, NP])
        self.Gmean = np.concatenate([self.Gmean, Gmean])
        if update:
            if self.scatter:
                try:
                    if auto_fit: n = self.scatter.fitting(Gmean, NP)
                    self.scatter.add_data(self.Gmean, self.NP, n or self.scatter.n, *args, **kwargs)
                except Exception as E:
                    pass
            if self.hist:
                try:
                    self.hist.add_data(self.Gmean, self.NP, n or self.hist.n, *args, **kwargs)
                except Exception as E:
                    pass


class LineNP(Line2D):
    """
    2D noise power-conductance scatter plot

    Parameters
    ----------
    n : float, optional
        noise power scaling exponent, by default 0
    xscale : str, optional
        linear or log scale of x axis (G), by default 'log'
    yscale : str, optional
        linear or log scale of y axis (NP), by default 'log'
    xlabel : str, optional
        set xlabel, by default 'Conductance (/G_0$)'
    ylabel : str, optional
        set ylabel, by default 'Noise power'
    kwargs
        Line2D kwargs
    """

    def __init__(self, xscale='log', yscale='log', *, xlabel: str = 'Conductance ($G/G_0$)', ylabel: str = 'Noise power', n: float = 0, **kwargs):
        super().__init__(xscale, yscale, xlabel=xlabel, ylabel=ylabel, **kwargs)
        self.n = n

    @property
    def trace(self):
        if len(self.lines): return self.ax.lines[0].get_data()[0].shape[0]
        else: return 0

    def add_data(self, Gmean: np.ndarray, NP: np.ndarray, n: float = None, auto_fit: bool = False, *args, clear: bool = True, **kwargs):
        """
        Set data into noise power-conductance scatter plot

        Parameters
        ----------
        Gmean : np.ndarray
            1D Gmean array
        NP : np.ndarray
            1D NP array
        n : float, optional
            noise power scaling exponent, by default None
        auto_fit : bool, optional
            calculate n by fitting NP=c*G^n, by default False
        clear : bool, optional
            clear previous data, by default True
        kwargs
            Line2D.add_data kwargs
        """
        if auto_fit: n = self.fitting(Gmean, NP)
        if n is not None: self.n = n
        self.ax.set_ylabel('Noise power / ($G/G_0)^{%.2f}$' % round(self.n, 2) if self.n else 'Noise power')
        if clear: self.clear_data()
        super().add_data(Gmean, NP / Gmean**self.n if self.n else NP, marker=".", linestyle='None', *args, **kwargs)

    def fitting(self, Gmean: np.ndarray, NP: np.ndarray, *, return_c: bool = False) -> float:
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


class HistNP(Hist2D):
    """
    2D noise power-conductance histogram

    Parameters
    ----------
    Glim : _type_
        _description_
    NPlim : _type_
        _description_
    num_G_bins : _type_
        _description_
    num_NP_bins : _type_
        _description_
    xscale : str, optional
        linear or log scale of x axis (G), by default 'log'
    yscale : str, optional
        linear or log scale of y axis (NP), by default 'log'
    xlabel : str, optional
        set xlabel, by default 'Conductance (/G_0$)'
    ylabel : str, optional
        set ylabel, by default 'Noise power'
    set_colorbar : bool, optional
        _description_, by default False
    n : float, optional
        noise power scaling exponent, by default 0
    kwargs
        Hist2D kwargs
    """

    def __init__(self,
                 Glim,
                 NPlim,
                 num_G_bins,
                 num_NP_bins,
                 xscale='log',
                 yscale='log',
                 *,
                 xlabel: str = 'Conductance ($G/G_0$)',
                 ylabel: str = 'Noise power',
                 set_colorbar=False,
                 n: float = 0,
                 **kwargs):
        super().__init__(Glim, NPlim, num_G_bins, num_NP_bins, xscale, yscale, xlabel=xlabel, ylabel=ylabel, set_colorbar=set_colorbar, **kwargs)
        self.n = n

    def add_data(self, Gmean: np.ndarray, NP: np.ndarray, n: float = None, auto_fit: bool = False, *args, clear: bool = True, **kwargs):
        """
        Set data into noise power-conductance histogram

        Parameters
        ----------
        Gmean : np.ndarray
            _description_
        NP : np.ndarray
            _description_
        n : float, optional
            noise power scaling exponent, by default None
        auto_fit : bool, optional
            calculate n by minimized gaussian2d theta, by default False
        clear : bool, optional
            clear previous data, by default True
        kwargs
            Hist2D.add_data kwargs
        """
        if auto_fit: n = self.fitting(Gmean, NP)
        if n is not None: self.n = n
        self.ax.set_ylabel('Noise power / ($G/G_0)^{%.2f}$' % round(self.n, 2) if self.n else 'Noise power')
        if clear: self.clear_data()
        super().add_data(Gmean, NP / Gmean**self.n if self.n else NP, trace=Gmean.size, *args, **kwargs)

    def contour(self,
                x_range: list[float, float] = [-np.inf, np.inf],
                y_range: list[float, float] = [-np.inf, np.inf],
                p0: list = None,
                bounds: list = None,
                *args,
                split: bool = False,
                **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """
        Get contour by fitting 2D Gaussian distribution surface

        Parameters
        ----------
        x_range : list[float, float], optional
            x range used to fit, by default [-np.inf, np.inf]
        y_range : list[float, float], optional
            y range used to fit, by default [-np.inf, np.inf]
        p0 : list, optional
            initial guess for the parameters in multi_gaussian function, use log scale if xscale=='log', by default [1,0,0,1,1,0]
        bounds : list, optional
            lower and upper bounds of parameters
        split : bool, optional
            split surface into multiple gaussian surfaces, by default False
        args
            plt.contour args
        kwargs
            plt.contour kwargs

        Returns
        -------
        z : np.ndarray
            Gaussian distribution surface
        params : np.ndarray
            gaussian2d fitting parameters (A, ux, uy, sx, sy, theta)
        """
        x = self.x_ if self.xscale == 'linear' else np.log10(self.x_)
        y = self.y_ if self.yscale == 'linear' else np.log10(self.y_)
        fx = np.where((self.x > min(x_range)) & (self.x < max(x_range)))[0]
        fy = np.where((self.y > min(y_range)) & (self.y < max(y_range)))[0]
        params = scipy.optimize.curve_fit(
            multi_gaussian2d, (x[fx][:, fy].ravel(), y[fx][:, fy].ravel()),
            self.height[fx][:, fy].ravel(),
            p0=p0 if p0 is not None else [1, 0, 0, 1, 1, 0],
            bounds=bounds or [[0, self.x_min if self.xscale == 'linear' else np.log10(self.x_min), self.y_min if self.yscale == 'linear' else np.log10(self.y_min), 0, 0, -np.pi],
                              [np.inf, self.x_max if self.xscale == 'linear' else np.log10(self.x_max), self.y_max if self.yscale == 'linear' else np.log10(self.y_max), np.inf, np.inf, np.pi]])[0]
        z = gaussian2d((x, y), *params.reshape(6, params.size // 6)) if split else multi_gaussian2d((x, y), *params)
        return z, params

    def plot_contour(self,
                     x_range: list[float, float] = [-np.inf, np.inf],
                     y_range: list[float, float] = [-np.inf, np.inf],
                     p0: list = None,
                     bounds: list = None,
                     *args,
                     split: bool = False,
                     **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """
        Get contour by fitting 2D Gaussian distribution surface and plot surface

        Parameters
        ----------
        x_range : list[float, float], optional
            x range used to fit, by default [-np.inf, np.inf]
        y_range : list[float, float], optional
            y range used to fit, by default [-np.inf, np.inf]
        p0 : list, optional
            initial guess for the parameters in multi_gaussian function, use log scale if xscale=='log', by default [1,0,0,1,1,0]
        bounds : list, optional
            lower and upper bounds of parameters
        split : bool, optional
            split surface into multiple gaussian surfaces, by default False
        args
            plt.contour args
        kwargs
            plt.contour kwargs

        Returns
        -------
        z : np.ndarray
            Gaussian distribution surface
        params : np.ndarray
            gaussian2d fitting parameters (A, ux, uy, sx, sy, theta)
        """
        z, params = self.contour(x_range, y_range, p0, bounds, split=split)
        if split:
            for zi in z:
                self.ax.contour(self.x_, self.y_, zi, *args, **kwargs)
        else:
            self.ax.contour(self.x_, self.y_, z, *args, **kwargs)
        return z, params

    def fitting(self, *args, **kwargs):
        raise NotImplementedError