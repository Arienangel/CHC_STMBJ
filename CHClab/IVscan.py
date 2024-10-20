from .common import *


def extract_data(raw_data: Union[np.ndarray, str, list] = None,
                 upper: float = 1.45,
                 lower: float = -1.45,
                 length_segment: int = 1200,
                 num_segment: int = 1,
                 offset: list = [0, 0],
                 units: list = [1e-6, 1],
                 mode: Literal['gradient', 'height'] = 'height',
                 tolerance: int = 0,
                 *,
                 I_raw: np.array = None,
                 V_raw: np.array = None,
                 **kwargs):
    '''
    Extract traces from raw_data

    Args:
        raw_data (ndarray | str): 2D array (I, V) contains raw data, directory of files, zip file, or txt file
        height (float, optional): peak height of voltage
        length_segment (int, optional): length of one segment
        num_segment (int, oprional):numbber of segments
        offset (list, optional): length from first point to first peak and last point to last peak
        units (list, optional): default: (μA, V)
        mode (str, optional): method used to find peaks
        tolerance (int, optional): tolerance of length_segment (length_segment-tolerance~length_segment+tolerance)
        I_raw (ndarray, optional): raw current in 1D array
        V_raw (ndarray, optional): raw voltage in 1D array

    Returns:
        I (ndarray): current (A) in 2D array (#traces, length)
        V (ndarray): voltage (V) in 2D array (#traces, length)
    '''
    if raw_data is None: raw_data = np.stack([I_raw, V_raw])
    elif not isinstance(raw_data, np.ndarray): raw_data = load_data(raw_data, **kwargs)[::-1]
    I, V = raw_data * np.expand_dims(units, 1)
    match mode:
        case 'gradient':
            peaks = scipy.signal.find_peaks(np.abs(np.gradient(np.gradient(V))), distance=length_segment / 4)[0]
        case 'height':
            peaks = np.concatenate([scipy.signal.find_peaks(V, height=upper, distance=length_segment / 4)[0], scipy.signal.find_peaks(-V, height=-lower, distance=length_segment / 4)[0]])
    start_seg_index = peaks[np.isin(peaks + length_segment, (peaks + np.expand_dims(np.arange(-tolerance, tolerance + 1), -1)).ravel())]
    if start_seg_index.size == 0: return np.zeros((2, 0, length_segment * num_segment + sum(offset)))
    V_seg = np.stack([V[i:i + length_segment] for i in start_seg_index])
    rm_start_seg_index = np.concatenate([scipy.signal.argrelmax(V_seg, axis=1)[0], scipy.signal.argrelmin(V_seg, axis=1)[0], np.where(V_seg.min(axis=1) > lower)[0], np.where(V_seg.max(axis=1) < upper)[0]])  # remove non-monotonically increasing/decreasing voltage
    start_seg_index = np.delete(start_seg_index, rm_start_seg_index)
    if start_seg_index.size == 0: return np.zeros((2, 0, length_segment * num_segment + sum(offset)))
    # segment
    if num_segment == 1:
        return np.stack([[I[p - offset[0]:p + length_segment + offset[1]], V[p - offset[0]:p + length_segment + offset[1]]] for p in start_seg_index], axis=1)
    # fullcycle
    else:
        start_full_index = start_seg_index[np.isin(start_seg_index + length_segment * num_segment, (peaks + np.expand_dims(np.arange(-tolerance * num_segment, tolerance * num_segment + 1), -1)).ravel())]
        is_start_full_index = (np.isin(np.expand_dims(start_full_index, axis=1) + np.expand_dims((np.arange(num_segment) + 1) * length_segment, axis=0), (start_seg_index + length_segment + np.expand_dims(np.arange(-tolerance * num_segment, tolerance * num_segment + 1), -1)).ravel()).sum(axis=1)
                               == num_segment) & (start_full_index + length_segment * num_segment + offset[1] < I.size) & (start_full_index - offset[0] >= 0)
        if start_full_index[is_start_full_index].size == 0: return np.zeros((2, 0, length_segment * num_segment + sum(offset)))
        return np.stack([[I[p - offset[0]:p + length_segment * num_segment + offset[1]], V[p - offset[0]:p + length_segment * num_segment + offset[1]]] for p in start_full_index[is_start_full_index]], axis=1)


def noise_remove(I: np.ndarray, V: np.ndarray, V0: float = 0, dV: float = None, I_limit: float = None, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    '''
    Remove noise traces

    Args:
        I (ndarray): current (A) in 2D array (#traces, length)
        V (ndarray): voltage (V) in 2D array (#traces, length)
        V0, dV (float, optional): Remove traces that I.min() is not between V0±dV
        I_limit (float, optional): Remove traces that I.max() is greater than I_limit

    Returns:
        I (ndarray): current (A) in 2D array (#traces, length)
        V (ndarray): voltage (V) in 2D array (#traces, length)
    '''
    if V.ndim == 1:
        I = np.expand_dims(I, 0)
        V = np.expand_dims(V, 0)
    if dV:
        zero_point = np.diagonal(V[:, np.abs(I).argmin(axis=1)])
        f = np.abs(zero_point - V0) < dV
        I, V = I[f], V[f]
    if I_limit:
        f = np.where(np.abs(I).max(axis=1) < I_limit, True, False)
        I, V = I[f], V[f]
    return I, V


def zeroing(I: np.ndarray, V: np.ndarray, V0: float = 0, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    '''
    Set minimum current value at V=0

    Args:
        I (ndarray): current (A) in 2D array (#traces, length)
        V (ndarray): voltage (V) in 2D array (#traces, length)
        V0 (float): set I min to this V

    Returns:
        I (ndarray): current (A) in 2D array (#traces, length)
        V (ndarray): voltage (V) in 2D array (#traces, length)
    '''
    if V.ndim == 1:
        I = np.expand_dims(I, 0)
        V = np.expand_dims(V, 0)
    zero_point = np.diagonal(V[:, np.abs(I).argmin(axis=1)])
    return I, V - np.expand_dims(zero_point, axis=1) + V0


def split_scan_direction(I: np.ndarray, V: np.ndarray, **kwargs) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    '''
    Split voltage scanning direction

    Args:
        I (ndarray): current (A) in 2D array (#traces, length)
        V (ndarray): voltage (V) in 2D array (#traces, length)

    Returns:
        ascending (tuple):  tuple of current and ascending voltage
        descending (tuple):  tuple of current and descending voltage
    '''
    if V.ndim == 1:
        I = np.expand_dims(I, 0)
        V = np.expand_dims(V, 0)
    filter = np.where((V[:, -1] - V[:, 0]) > 0, True, False)
    ascending = I[filter], V[filter]
    descending = I[~filter], V[~filter]
    return ascending, descending


class Hist_GV(Hist2D):

    def __init__(self, xlim: tuple[float, float] = (-1.5, 1.5), ylim: tuple[float, float] = (1e-5, 1e-1), num_x_bin: float = 300, num_y_bin: float = 300, xscale: Literal['linear', 'log'] = 'linear', yscale: Literal['linear', 'log'] = 'log', xlabel: Literal['bias', 'wk'] = 'bias', **kwargs) -> None:
        super().__init__(xlim, ylim, num_x_bin, num_y_bin, xscale, yscale, **kwargs)
        self.ax.set_xlabel('$E_{%s}\/(V)$' % xlabel)
        self.ax.set_ylabel('Conductance ($G/G_0$)')
        self.colorbar.set_label('Count/trace')
        self.xlabel = xlabel

    def add_data(self, I: np.ndarray = None, V: np.ndarray = None, *, G: np.ndarray = None, Vbias: float | np.ndarray = None, **kwargs) -> None:
        """
        Add data into 2D histogram (GV)

        Args:
            I (ndarray): 2D I array with shape (trace, length)
            V (ndarray): 2D V array with shape (trace, length)
            G (ndarray, optional): 2D G array with shape (trace, length), this will ignore I
            Vbias (float|np.ndarray, optional): only used if x-axis is not Ebias
        """
        if G is None:
            if self.xlabel == 'bias': G = conductance(I, V)
            else: G = conductance(I, Vbias)
        super().add_data(V, np.abs(G), **kwargs)


class Hist_IV(Hist2D):

    def __init__(self, xlim: tuple[float, float] = (-1.5, 1.5), ylim: tuple[float, float] = (1e-11, 1e-5), num_x_bin: float = 300, num_y_bin: float = 300, xscale: Literal['linear', 'log'] = 'linear', yscale: Literal['linear', 'log'] = 'log', xlabel: Literal['bias', 'wk'] = 'bias', **kwargs) -> None:
        super().__init__(xlim, ylim, num_x_bin, num_y_bin, xscale, yscale, **kwargs)
        self.ax.set_xlabel('$E_{%s}\/(V)$' % xlabel)
        self.ax.set_ylabel('Current (A)')
        self.colorbar.set_label('Count/trace')

    def add_data(self, I: np.ndarray, V: np.ndarray, **kwargs) -> None:
        """
        Add data into 2D histogram (IV)

        Args:
            I (ndarray): 2D I array with shape (trace, length)
            V (ndarray): 2D V array with shape (trace, length)
        """
        super().add_data(V, np.abs(I), **kwargs)

    def add_contour_G(self, levels: np.ndarray = np.logspace(-6, -1, 6), colors='k', linestyles: Literal['solid', 'dashed', 'dashdot', 'dotted'] = 'dotted', fontsize=8, **kwargs):
        """
        Add conductance contour into 2D histogram (IV)

        Args:
            levels (ndarray)
            linestyles (str, optional): 'solid', 'dashed', 'dashdot', 'dotted'
            colors (str, optional)
            fontsize (str, optional)
        """
        x, y = np.meshgrid(self.x, self.y)
        z = np.abs(y / x / G0)
        if self.yscale == 'log':
            c = self.ax.contour(x, y, z, levels=levels, colors=colors, linestyles=linestyles)
            fmt = matplotlib.ticker.LogFormatterMathtext()
            fmt.create_dummy_axis()
            self.ax.clabel(c, c.levels, fmt=fmt, fontsize=fontsize, **kwargs)
        else:
            c = self.ax.contour(x, y, z, levels=levels, colors=colors, linestyles=linestyles)
            fmt = matplotlib.ticker.ScalarFormatter()
            fmt.create_dummy_axis()
            self.ax.clabel(c, c.levels, fmt=fmt, fontsize=fontsize, **kwargs)


class Hist_IVt(Hist2D):

    def __init__(self,
                 xlim: tuple[float, float] = (0, 0.2),
                 y1lim: tuple[float, float] = (1e-11, 1e-5),
                 y2lim: tuple[float, float] = (-1.5, 1.5),
                 num_x_bin: float = 1000,
                 num_y1_bin: float = 300,
                 xscale: Literal['linear', 'log'] = 'linear',
                 y1scale: Literal['linear', 'log'] = 'log',
                 y2scale: Literal['linear', 'log'] = 'linear',
                 x_conversion: float = 40000,
                 xlabel: Literal['bias', 'wk'] = 'bias',
                 **kwargs) -> None:
        super().__init__(xlim, y1lim, num_x_bin, num_y1_bin, xscale, y1scale, **kwargs)
        self.colorbar.remove()
        self.ax2 = self.ax.twinx()
        self.plot2 = pd.Series()
        self.ax2.set_ylim(*sorted(y2lim))
        self.ax2.set_yscale(y2scale)
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Current (A)')
        self.ax2.set_ylabel('$E_{%s}\/(V)$' % xlabel)
        self.x_conversion = x_conversion

    def add_data(self, I: np.ndarray, V: np.ndarray, t: np.array = None, **kwargs) -> None:
        """
        Add data into 2D histogram (IVt)

        Args:
            I (ndarray): 2D I array with shape (trace, length)
            V (ndarray): 2D V array with shape (trace, length)
            t (ndarray, optional): 2D t array with shape (trace, length)
        """
        if t is None: t = np.mgrid[0:I.shape[0]:1, 0:I.shape[1]:1][1] / self.x_conversion
        elif t.ndim == 1: t = np.tile(t, (I.shape[0], 1))
        super().add_data(t, np.abs(I), **kwargs)
        V = pd.Series(list(V)).drop_duplicates()
        V_new = V[~V.isin(self.plot2)]
        if V_new.size > 0:
            self.ax2.plot(t[0], np.stack(V_new).T, color='black', linewidth=0.5)
            self.plot2 = pd.concat([self.plot2, V_new])


class Hist_GVt(Hist2D):

    def __init__(self,
                 xlim: tuple[float, float] = (0, 0.2),
                 y1lim: tuple[float, float] = (1e-5, 1e-1),
                 y2lim: tuple[float, float] = (-1.5, 1.5),
                 num_x_bin: float = 1000,
                 num_y1_bin: float = 300,
                 xscale: Literal['linear', 'log'] = 'linear',
                 y1scale: Literal['linear', 'log'] = 'log',
                 y2scale: Literal['linear', 'log'] = 'linear',
                 x_conversion: float = 40000,
                 xlabel: Literal['bias', 'wk'] = 'bias',
                 **kwargs) -> None:
        super().__init__(xlim, y1lim, num_x_bin, num_y1_bin, xscale, y1scale, **kwargs)
        self.colorbar.remove()
        self.ax2 = self.ax.twinx()
        self.plot2 = pd.Series()
        self.ax2.set_ylim(*sorted(y2lim))
        self.ax2.set_yscale(y2scale)
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Conductance ($G/G_0$)')
        self.ax2.set_ylabel('$E_{%s}\/(V)$' % xlabel)
        self.x_conversion = x_conversion
        self.xlabel = xlabel

    def add_data(self, I: np.ndarray = None, V: np.ndarray = None, t: np.array = None, *, G: np.ndarray = None, Vbias: float | np.ndarray = None, **kwargs) -> None:
        """
        Add data into 2D histogram (GVt)

        Args:
            I (ndarray): 2D I array with shape (trace, length)
            V (ndarray): 2D V array with shape (trace, length)
            t (ndarray, optional): 2D t array with shape (trace, length)
            G (ndarray, optional): 2D G array with shape (trace, length), this will ignore I
            Vbias (float|np.ndarray, optional): only used if x-axis is not Ebias
        """
        if G is None:
            if self.xlabel == 'bias': G = conductance(I, V)
            else: G = conductance(I, Vbias)
        if t is None: t = np.mgrid[0:G.shape[0]:1, 0:G.shape[1]:1][1] / self.x_conversion
        elif t.ndim == 1: t = np.tile(t, (G.shape[0], 1))
        super().add_data(t, np.abs(G), **kwargs)
        V = pd.Series(list(V)).drop_duplicates()
        V_new = V[~V.isin(self.plot2)]
        if V_new.size > 0:
            self.ax2.plot(t[0], np.stack(V_new).T, color='black', linewidth=0.5)
            self.plot2 = pd.concat([self.plot2, V_new])
