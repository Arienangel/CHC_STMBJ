from .common import *


def extract_data(raw_data: Union[np.ndarray, str, list] = None,
                 upper: float = 0.95,
                 lower: float = -0.95,
                 length_segment: int = 800,
                 num_segment: int = 1,
                 offset: list = [0, 0],
                 units: list = [1e-6, 1],
                 mode: Literal['gradient', 'height'] = 'height',
                 tolerance: int = 0,
                 *,
                 I_raw: np.array = None,
                 V_raw: np.array = None,
                 **kwargs) -> np.ndarray:
    """
    Extract traces from raw data

    Parameters
    ----------
    raw_data : Union[np.ndarray, str, list], optional
        2D array (I, V) contains raw data, directory of files, zip file, or txt file, by default None
    upper : float, optional
        extract traces with data greater than upper limit, by default 0.95
    lower : float, optional
        extract traces with data less than lower limit, by default -0.95
    length_segment : int, optional
        length of one segment, calculated by sampling rate*voltage range/scan rate, by default 800
    num_segment : int, optional
        number of segments, used to extract fullcycle, by default 1
    offset : list, optional
        length before first peak and after last peak, by default [0, 0]
    units : list, optional
        unit of I and V, by default [1e-6, 1]
    mode : Literal[&#39;gradient&#39;, &#39;height&#39;], optional
        method used to find peaks, by default 'height'
    tolerance : int, optional
        tolerance of length_segment (length_segment-tolerance~length_segment+tolerance), by default 0
    I_raw : np.array, optional
        raw current in 1D array , by default None
    V_raw : np.array, optional
        raw voltage in 1D array, by default None
    kwargs
        load_data kwargs

    Returns
    -------
    I : np.ndarray
        2D I array with shape (trace, length)
    V : np.ndarray
        2D V array with shape (trace, length)        
    """
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
    rm_start_seg_index = np.concatenate(
        [scipy.signal.argrelmax(V_seg, axis=1)[0],
         scipy.signal.argrelmin(V_seg, axis=1)[0],
         np.where(V_seg.min(axis=1) > lower)[0],
         np.where(V_seg.max(axis=1) < upper)[0]])  # remove non-monotonically increasing/decreasing voltage
    start_seg_index = np.delete(start_seg_index, rm_start_seg_index)
    if start_seg_index.size == 0: return np.zeros((2, 0, length_segment * num_segment + sum(offset)))
    # segment
    if num_segment == 1:
        return np.stack([[I[p - offset[0]:p + length_segment + offset[1]], V[p - offset[0]:p + length_segment + offset[1]]] for p in start_seg_index], axis=1)
    # fullcycle
    else:
        start_full_index = start_seg_index[np.isin(start_seg_index + length_segment * num_segment,
                                                   (peaks + np.expand_dims(np.arange(-tolerance * num_segment, tolerance * num_segment + 1), -1)).ravel())]
        is_start_full_index = (np.isin(
            np.expand_dims(start_full_index, axis=1) + np.expand_dims((np.arange(num_segment) + 1) * length_segment, axis=0),
            (start_seg_index + length_segment + np.expand_dims(np.arange(-tolerance * num_segment, tolerance * num_segment + 1), -1)).ravel()).sum(axis=1)
                               == num_segment) & (start_full_index + length_segment * num_segment + offset[1] < I.size) & (start_full_index - offset[0] >= 0)
        if start_full_index[is_start_full_index].size == 0: return np.zeros((2, 0, length_segment * num_segment + sum(offset)))
        return np.stack([[I[p - offset[0]:p + length_segment * num_segment + offset[1]], V[p - offset[0]:p + length_segment * num_segment + offset[1]]] for p in start_full_index[is_start_full_index]],
                        axis=1)


def noise_remove(I: np.ndarray, V: np.ndarray, V0: float = 0, dV: float = None, I_limit: float = None, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove noise traces

    Parameters
    ----------
    I : np.ndarray
        current in 2D array (traces, length)
    V : np.ndarray
        voltage in 2D array (traces, length)
    V0 : float, optional
    dV : float, optional
        remove traces that I.min() is not between V0±dV
    I_limit : float, optional
        remove traces that I.max() is greater than I_limit, by default None

    Returns
    -------
    I : np.ndarray
        2D I array with shape (trace, length)
    V : np.ndarray
        2D V array with shape (trace, length)        
    """
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
    """
    Set minimum current value at V=0

    Parameters
    ----------
    I : np.ndarray
        current in 2D array (traces, length)
    V : np.ndarray
        voltage in 2D array (traces, length)
    V0 : float, optional
        set I min to this V, by default 0

    Returns
    -------
    I : np.ndarray
        2D I array with shape (trace, length)
    V : np.ndarray
        2D V array with shape (trace, length)        
    """
    if V.ndim == 1:
        I = np.expand_dims(I, 0)
        V = np.expand_dims(V, 0)
    zero_point = np.diagonal(V[:, np.abs(I).argmin(axis=1)])
    return I, V - np.expand_dims(zero_point, axis=1) + V0


def split_scan_direction(I: np.ndarray, V: np.ndarray, **kwargs) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
    Split voltage scanning direction

    Parameters
    ----------
    I : np.ndarray
        current in 2D array (traces, length)
    V : np.ndarray
        voltage in 2D array (traces, length)

    Returns
    -------
    ascending : tuple
        tuple of current and ascending voltage
    descending : tuple
        tuple of current and descending voltage
    """
    if V.ndim == 1:
        I = np.expand_dims(I, 0)
        V = np.expand_dims(V, 0)
    filter = np.where((V[:, -1] - V[:, 0]) > 0, True, False)
    ascending = I[filter], V[filter]
    descending = I[~filter], V[~filter]
    return ascending, descending


class Hist_GV(Hist2D):
    """
    2D conductance-voltage histogram

    Parameters
    ----------
    Vlim : tuple[float, float], optional
        max and min value of V, by default (-1.0, 1.0)
    Glim : tuple[float, float], optional
        _description_, by default (1e-5, 1e-1)
    num_V_bins : float, optional
        number of V bins, by default 400
    num_G_bins : float, optional
        number of G bins, by default 400
    xscale : Literal[&#39;linear&#39;, &#39;log&#39;], optional
        linear or log scale of x axis, by default 'linear'
    yscale : Literal[&#39;linear&#39;, &#39;log&#39;], optional
        linear or log scale of y axis, by default 'log'
    Vtype : Literal[&#39;bias&#39;, &#39;wk&#39;], optional
        voltage type, Vbias or Vwk, by default 'bias'
    xlabel : str, optional
        set xlabel, set %s as Vtype, by default '$E_{%s}\/(V)$'
    ylabel : str, optional
        set ylabel, by default 'Conductance ($G/G_0$)'
    colorbar_label : str, optional
        set colorbar label, by default 'Count/trace'
    kwargs
        Hist2D kwargs
    """

    def __init__(self,
                 Vlim: tuple[float, float] = (-1.0, 1.0),
                 Glim: tuple[float, float] = (1e-5, 1e-1),
                 num_V_bins: float = 400,
                 num_G_bins: float = 400,
                 xscale: Literal['linear', 'log'] = 'linear',
                 yscale: Literal['linear', 'log'] = 'log',
                 Vtype: Literal['bias', 'wk'] = 'bias',
                 *,
                 xlabel: str = '$E_{%s}\/(V)$',
                 ylabel: str = 'Conductance ($G/G_0$)',
                 colorbar_label: str = 'Count/trace',
                 **kwargs) -> None:
        if '%s' in xlabel: xlabel = xlabel % Vtype
        super().__init__(Vlim, Glim, num_V_bins, num_G_bins, xscale, yscale, xlabel=xlabel, ylabel=ylabel, colorbar_label=colorbar_label, **kwargs)
        if 'bias' in Vtype: Vtype = 'bias'
        elif 'wk' in Vtype: Vtype = 'wk'
        else: raise ValueError('Unknown voltage type')
        self.Vtype = Vtype

    def add_data(self, I: np.ndarray = None, V: np.ndarray = None, *, G: np.ndarray = None, Vbias: float | np.ndarray = None, **kwargs) -> None:
        """
        Add data into 2D conductance-voltage histogram

        Parameters
        ----------
        I : np.ndarray, optional
            2D I array with shape (trace, length), by default None
        V : np.ndarray, optional
            2D V array with shape (trace, length), by default None
        G : np.ndarray, optional
            2D G array with shape (trace, length), this will ignore I, by default None
        Vbias : float | np.ndarray, optional
            only used if Vtype is not bias, by default None
        kwargs
            Hist2D.add_data kwargs
        """
        if G is None:
            if self.Vtype == 'bias': G = conductance(I, V)
            else: G = conductance(I, Vbias)
        super().add_data(V, np.abs(G), **kwargs)


class Hist_IV(Hist2D):
    """
    2D current-voltage histogram

    Parameters
    ----------
    Vlim : tuple[float, float], optional
        max and min value of V, by default (-1.0, 1.0)
    Ilim : tuple[float, float], optional
        and min value of I, by default (1e-11, 1e-5)
    num_V_bins : float, optional
        number of V bins, by default 400
    num_I_bins : float, optional
        _descr of I binsription_, by default 600
    xscale : Literal[&#39;linear&#39;, &#39;log&#39;], optional
        linear or log scale of x axis, by default 'linear'
    yscale : Literal[&#39;linear&#39;, &#39;log&#39;], optional
        linear or log scale of y axis, by default 'log'
    Vtype : Literal[&#39;bias&#39;, &#39;wk&#39;], optional
        voltage type, Vbias or Vwk, by default 'bias'
    xlabel : str, optional
        set xlabel, set %s as Vtype, by default '$E_{%s}\/(V)$'
    ylabel : str, optional
        set ylabel, by default 'Current (A)'
    colorbar_label : str, optional
        set colorbar label, by default 'Count/trace'
    kwargs
        Hist2D kwargs
    """

    def __init__(self,
                 Vlim: tuple[float, float] = (-1.0, 1.0),
                 Ilim: tuple[float, float] = (1e-11, 1e-5),
                 num_V_bins: float = 400,
                 num_I_bins: float = 600,
                 xscale: Literal['linear', 'log'] = 'linear',
                 yscale: Literal['linear', 'log'] = 'log',
                 Vtype: Literal['bias', 'wk'] = 'bias',
                 *,
                 xlabel: str = '$E_{%s}\/(V)$',
                 ylabel: str = 'Current (A)',
                 colorbar_label: str = 'Count/trace',
                 **kwargs) -> None:
        if '%s' in xlabel: xlabel = xlabel % Vtype
        super().__init__(Vlim, Ilim, num_V_bins, num_I_bins, xscale, yscale, xlabel=xlabel, ylabel=ylabel, colorbar_label=colorbar_label, **kwargs)
        if 'bias' in Vtype: Vtype = 'bias'
        elif 'wk' in Vtype: Vtype = 'wk'
        else: raise ValueError('Unknown voltage type')

    def add_data(self, I: np.ndarray, V: np.ndarray, **kwargs) -> None:
        """
        _summary_

        Parameters
        ----------
        I : np.ndarray
            2D I array with shape (trace, length)
        V : np.ndarray
            2D V array with shape (trace, length)
        kwargs
            Hist2D.add_data kwargs
        """
        super().add_data(V, np.abs(I), **kwargs)

    def add_contour_G(self, levels: np.ndarray = np.logspace(-6, -1, 6), colors='k', linestyles: Literal['solid', 'dashed', 'dashdot', 'dotted'] = 'dotted', fontsize=8, **kwargs):
        """
        Add conductance contour into 2D current-voltage histogram

        Parameters
        ----------
        levels : np.ndarray, optional
            contour levels, by default np.logspace(-6, -1, 6)
        colors : str, optional
            contour colors, by default 'k'
        linestyles : Literal[&#39;solid&#39;, &#39;dashed&#39;, &#39;dashdot&#39;, &#39;dotted&#39;], optional
            contour linestyles, by default 'dotted'
        fontsize : int, optional
            contour fontsize, by default 8
        kwargs
            plt.clabel kwargs
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
    """
    2D current-time histogram with voltage-time plot

    Parameters
    ----------
    tlim : tuple[float, float], optional
        max and min value of t, by default (0, 0.12)
    Ilim : tuple[float, float], optional
        max and min value of I, by default (1e-11, 1e-5)
    Vlim : tuple[float, float], optional
        max and min value of V, by default (-1.0, 1.0)
    num_t_bins : float, optional
        number of t bins, by default 600
    num_I_bins : float, optional
        number of I bins, by default 600
    xscale : Literal[&#39;linear&#39;, &#39;log&#39;], optional
        linear or log scale of x axis (t), by default 'linear'
    y1scale : Literal[&#39;linear&#39;, &#39;log&#39;], optional
        linear or log scale of y1 axis (I), by default 'log'
    y2scale : Literal[&#39;linear&#39;, &#39;log&#39;], optional
        linear or log scale of y2 axis (V), by default 'linear'
    x_conversion : float, optional
        sampling rate, points per second, by default 40000
    Vtype : Literal[&#39;bias&#39;, &#39;wk&#39;], optional
        voltage type, Vbias or Vwk, by default 'bias'
    xlabel : str, optional
        set xlabel, by default 'Time (s)'
    y1label : str, optional
        set y1label, by default 'Current (A)'
    y2label : str, optional
        set y2label, set %s as Vtype, by default '$E_{%s}\/(V)$'
    set_colorbar : bool, optional
        add colorbar, by default False
    kwargs
        Hist2D kwargs
    """

    def __init__(self,
                 tlim: tuple[float, float] = (0, 0.12),
                 Ilim: tuple[float, float] = (1e-11, 1e-5),
                 Vlim: tuple[float, float] = (-1.0, 1.0),
                 num_t_bins: float = 600,
                 num_I_bins: float = 600,
                 xscale: Literal['linear', 'log'] = 'linear',
                 y1scale: Literal['linear', 'log'] = 'log',
                 y2scale: Literal['linear', 'log'] = 'linear',
                 x_conversion: float = 40000,
                 Vtype: Literal['bias', 'wk'] = 'bias',
                 *,
                 xlabel: str = 'Time (s)',
                 y1label: str = 'Current (A)',
                 y2label: str = '$E_{%s}\/(V)$',
                 set_colorbar: bool = False,
                 **kwargs) -> None:
        if '%s' in y2label: y2label = y2label % Vtype
        super().__init__(tlim, Ilim, num_t_bins, num_I_bins, xscale, y1scale, xlabel=xlabel, ylabel=y1label, set_colorbar=set_colorbar, **kwargs)
        self.ax2 = self.ax.twinx()
        self.plot2 = pd.Series()
        self.ax2.set(yscale=y2scale, ylim=sorted(Vlim), ylabel=y2label)
        if 'bias' in Vtype: Vtype = 'bias'
        elif 'wk' in Vtype: Vtype = 'wk'
        else: raise ValueError('Unknown voltage type')
        self.x_conversion = x_conversion

    def add_data(self, I: np.ndarray, V: np.ndarray, t: np.array = None, **kwargs) -> None:
        """
        Add data into 2D current-time histogram

        Parameters
        ----------
        I : np.ndarray
            2D I array with shape (trace, length)
        V : np.ndarray
            2D V array with shape (trace, length)
        t : np.array, optional
            2D t array with shape (trace, length), by default None
        kwargs
            Hist2D.add_data kwargs
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
    """
    2D conductance-time histogram with voltage-time plot

    Parameters
    ----------
    tlim : tuple[float, float], optional
        max and min value of t, by default (0, 0.12)
    Glim : tuple[float, float], optional
        max and min value of G, by default (1e-5, 1e-1)
    Vlim : tuple[float, float], optional
        max and min value of V, by default (-1.0, 1.0)
    num_t_bins : float, optional
        number of t bins, by default 600
    num_G_bins : float, optional
        number of G bins, by default 400
    xscale : Literal[&#39;linear&#39;, &#39;log&#39;], optional
        linear or log scale of x axis (t), by default 'linear'
    y1scale : Literal[&#39;linear&#39;, &#39;log&#39;], optional
        linear or log scale of y1 axis (G), by default 'log'
    y2scale : Literal[&#39;linear&#39;, &#39;log&#39;], optional
        linear or log scale of y2 axis (V), by default 'linear'
    x_conversion : float, optional
        sampling rate, points per second, by default 40000
    Vtype : Literal[&#39;bias&#39;, &#39;wk&#39;], optional
        voltage type, Vbias or Vwk, by default 'bias'
    xlabel : str, optional
        set xlabel, by default 'Time (s)'
    y1label : str, optional
        set y1label, by default 'Current (A)'
    y2label : str, optional
        set y2label, set %s as Vtype, by default '$E_{%s}\/(V)$'
    set_colorbar : bool, optional
        add colorbar, by default False
    kwargs
        Hist2D kwargs
    """

    def __init__(self,
                 tlim: tuple[float, float] = (0, 0.12),
                 Glim: tuple[float, float] = (1e-5, 1e-1),
                 Vlim: tuple[float, float] = (-1.0, 1.0),
                 num_t_bins: float = 600,
                 num_G_bins: float = 400,
                 xscale: Literal['linear', 'log'] = 'linear',
                 y1scale: Literal['linear', 'log'] = 'log',
                 y2scale: Literal['linear', 'log'] = 'linear',
                 x_conversion: float = 40000,
                 Vtype: Literal['bias', 'wk'] = 'bias',
                 *,
                 xlabel: str = 'Time (s)',
                 y1label: str = 'Conductance ($G/G_0$)',
                 y2label: str = '$E_{%s}\/(V)$',
                 set_colorbar: bool = False,
                 **kwargs) -> None:
        if '%s' in y2label: y2label = y2label % Vtype
        super().__init__(tlim, Glim, num_t_bins, num_G_bins, xscale, y1scale, xlabel=xlabel, ylabel=y1label, set_colorbar=set_colorbar, **kwargs)
        self.ax2 = self.ax.twinx()
        self.plot2 = pd.Series()
        self.ax2.set(yscale=y2scale, ylim=sorted(Vlim), ylabel=y2label)
        if 'bias' in Vtype: Vtype = 'bias'
        elif 'wk' in Vtype: Vtype = 'wk'
        else: raise ValueError('Unknown voltage type')
        self.x_conversion = x_conversion
        self.Vtype = Vtype

    def add_data(self, I: np.ndarray = None, V: np.ndarray = None, t: np.array = None, *, G: np.ndarray = None, Vbias: float | np.ndarray = None, **kwargs) -> None:
        """
        Add data into 2D conductance-time histogram

        Parameters
        ----------
        I : np.ndarray, optional
            2D I array with shape (trace, length), by default None
        V : np.ndarray, optional
            2D V array with shape (trace, length), by default None
        t : np.array, optional
            2D t array with shape (trace, length), by default None
        G : np.ndarray, optional
            2D G array with shape (trace, length), this will ignore I, by default None
        Vbias : float | np.ndarray, optional
            only used if Vtype is not Vbias, by default None
        kwargs
            Hist2D.add_data kwargs
        """
        if G is None:
            if self.Vtype == 'bias': G = conductance(I, V)
            else: G = conductance(I, Vbias)
        if t is None: t = np.mgrid[0:G.shape[0]:1, 0:G.shape[1]:1][1] / self.x_conversion
        elif t.ndim == 1: t = np.tile(t, (G.shape[0], 1))
        super().add_data(t, np.abs(G), **kwargs)
        V = pd.Series(list(V)).drop_duplicates()
        V_new = V[~V.isin(self.plot2)]
        if V_new.size > 0:
            self.ax2.plot(t[0], np.stack(V_new).T, color='black', linewidth=0.5)
            self.plot2 = pd.concat([self.plot2, V_new])
