from .common import *


def PSD(G: np.ndarray, sampling_rate: float = 40000) -> tuple[np.ndarray, np.ndarray]:
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
    return np.stack([(dt**2 / t) * np.abs(scipy.fft.fftshift(scipy.fft.fft(g)))**2 for g in G]), scipy.fft.fftshift(scipy.fft.fftfreq(n, dt))


def noise_power(PSD: np.ndarray, freq: np.ndarray, integrand: list = [100, 1000]) -> np.ndarray:
    """
    Args:
        PSD (np.ndarray): power spectral density
        freq (np.ndarray): sample frequency of PSD
        interval (list, optional): integrand. Defaults to 100~1000 Hz.

    Returns:
        NP (np.ndarray): noise power
    """
    integrand = sorted(integrand)
    idx = (freq >= integrand[0]) & (freq <= integrand[1])
    return 2 * scipy.integrate.trapezoid(PSD[:, idx], freq[idx], axis=1)  # include negative frequency



class Flicker_Noise:

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
        PSD, freq = PSD(G, self.sampling_rate)
        NP = noise_power(PSD, freq, integrand)
        Gmean = np.mean(G, axis=1)
        return self.plot(Gmean, NP, n, auto_fit)

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
        if auto_fit:
            n, c = scipy.optimize.curve_fit(lambda x, n, c: n * x + c,
                                            Gmean if self.xscale == 'linear' else np.log10(Gmean),
                                            NP if self.yscale == 'linear' else np.log10(NP),
                                            bounds=[[0, -np.inf], [3, np.inf]])[0]
        y = NP / Gmean**n if n else NP
        self.ax.plot(Gmean, y, '.')
        self.ax.set_ylabel('Noise power / ($G/G_0)^{%.2f}$' % round(n, 2) if n else 'Noise power')
        return n
