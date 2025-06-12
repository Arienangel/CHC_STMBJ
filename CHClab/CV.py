from .common import *
import re, io


class Segment:
    """
    IV segment

    Parameters
    ----------
    V : np.ndarray
        voltage
    I : np.ndarray
        current
    """

    def __init__(self, V: np.ndarray, I: np.ndarray):
        self.V = V
        self.I = I

    def __getitem__(self, index: slice):
        return Segment(self.V[index], self.I[index])

    def __len__(self):
        return self.V.size

    def extend(self, segments):
        self.I = np.concatenate([self.I, *[s.I for s in segments]])
        self.V = np.concatenate([self.V, *[s.V for s in segments]])

    def plot(self, *args, plot=None, **kwargs):
        """
        Plot segment

        Parameters
        ----------
        plot : PlotCV, optional
            _description_, by default None
        args
            PlotCV.add_segments args
        kwargs
            PlotCV.add_segments kwargs

        Returns
        -------
        PlotCV
            PlotCV object
        """
        plot = plot or PlotCV()
        plot.add_segments(self, *args, **kwargs)
        return plot


class Segments(list[Segment]):
    """
    IV segments

    Parameters
    ----------
    args: Segment
        multiple segments
    """

    def __init__(self, *args: Segment):
        super().__init__(*args)

    def __getitem__(self, index: int | slice | Iterable):
        if isinstance(index, Iterable):
            return Segments([self[i] for i in index])
        else:
            L = super().__getitem__(index)
            return L if isinstance(L, Segment) else Segments(L)

    def concat(self) -> Segment:
        """
        Concat all segments into one segment

        Returns
        -------
        Segment
            merged segment
        """

        return Segment(np.concatenate([s.V for s in self]), np.concatenate([s.I for s in self]))

    def plot(self, split_segment: bool = False, *args, label: list = None, plot=None, **kwargs):
        """
        Plot segments

        Parameters
        ----------
        split_segment : bool, optional
            split segments into multiple lines, by default False
        label : list, optional
            label of each segment, by default None
        plot : PlotCV, optional
            PlotCV object, by default None
        kwargs
            PlotCV.add_segments kwargs

        Returns
        -------
        PlotCV
            PlotCV object
        """
        plot = plot or PlotCV()
        if split_segment:
            plot.add_segments(self, label=label, *args, **kwargs)
        else:
            plot.add_segments(self.concat(), *args, **kwargs)
        return plot


class CVdata:
    """
    Cyclic voltammetry data

    Parameters
    ----------
    V : np.ndarray
        voltage
    I : np.ndarray
        current
    Vinit : float, optional
        start voltage, by default None
    Vmax : float, optional
        max voltage, by default None
    Vmin : float, optional
        min voltage, by default None
    Vfinal : float, optional
        end voltage, by default None
    polarity : bool, optional
        voltage scan from negative to positive if true, by default None
    scan_rate : float, optional
        voltage scan rate (V/s), by default None
    num_segment : int, optional
        number of segments, by default None
    interval : float, optional
        voltage change per point, by default None
    quiet_time : float, optional
        quiet time, by default None
    sensitivity : float, optional
        sensitivity, by default None
    """

    def __init__(self,
                 V: np.ndarray,
                 I: np.ndarray,
                 Vinit: float = None,
                 Vmax: float = None,
                 Vmin: float = None,
                 Vfinal: float = None,
                 polarity: bool = None,
                 scan_rate: float = None,
                 num_segment: int = None,
                 interval: float = None,
                 quiet_time: float = None,
                 sensitivity: float = None):
        self.V = V
        self.I = I
        self.fullcycle = Segment(V, I)
        segment_split_index = np.sort(np.concatenate([scipy.signal.find_peaks(self.V)[0], scipy.signal.find_peaks(-self.V)[0]]))
        Vsegment = np.array_split(V, segment_split_index)
        Isegment = np.array_split(I, segment_split_index)
        self.segments = Segments([Segment(V=Vsegment[i], I=Isegment[i]) for i in range(len(Vsegment))])
        self.Vinit = Vinit or V[0]
        self.Vmax = Vmax or V.max()
        self.Vmin = Vmin or V.min()
        self.Vfinal = Vfinal or V[-1]
        self.polarity = polarity or (V[1] - V[0]) > 0
        self.scan_rate = scan_rate
        self.num_segment = num_segment or len(Vsegment)
        self.interval = interval or np.abs((self.V[1:] - self.V[:-1]).mean())
        self.quiet_time = quiet_time
        self.sensitivity = sensitivity

    @staticmethod
    def extract_data_from_chi(file: str = None, string: str = None):
        """
        Extract CV data from chi760 output file

        Parameters
        ----------
        file : str, optional
            chi760 txt path, by default None
        string : str, optional
            chi760 txt content, by default None

        Returns
        -------
        CVdata
            CVdata object
        """
        if file is not None:
            with open(file, mode='r', encoding='utf-8') as f:
                data = f.read()
        else:
            data = string
        if r"Potential/V, Current/A" in data:
            V, I = np.genfromtxt(io.StringIO(re.search(r"Potential/V, Current/A\n\n((.|\n)*)\n", data).group(1)), delimiter=', ', unpack=True)
        elif r"Potential/V, i1/A, i2/A" in data:
            V, I1, I2 = np.genfromtxt(io.StringIO(re.search(r"Potential/V, i1/A, i2/A\n\n((.|\n)*)\n", data).group(1)), delimiter=', ', unpack=True)
            I = np.stack([I1, I2]).T
        else:
            raise NotImplementedError('Unknown format')
        Vinit = float(re.search(r"Init E \(V\) = (.*)\n", data).group(1))
        Vmax = float(re.search(r"High E \(V\) = (.*)\n", data).group(1))
        Vmin = float(re.search(r"Low E \(V\) = (.*)\n", data).group(1))
        try:
            Vfinal = float(re.search(r"Final E \(V\) = (.*)\n", data).group(1))
        except:
            Vfinal = None
        polarity = re.search(r"Init P/N = (.*)\n", data).group(1) == 'P'
        scan_rate = float(re.search(r"Scan Rate \(V/s\) = (.*)\n", data).group(1))
        num_segment = int(re.search(r"Segment = (.*)\n", data).group(1))
        interval = float(re.search(r"Sample Interval \(V\) = (.*)\n", data).group(1))
        quiet_time = float(re.search(r"Quiet Time \(sec\) = (.*)\n", data).group(1))
        sensitivity = float(re.search(r"Sensitivity \(A/V\) = (.*)\n", data).group(1))
        return CVdata(V, I, Vinit, Vmax, Vmin, Vfinal, polarity, scan_rate, num_segment, interval, quiet_time, sensitivity)

    def __getitem__(self, index: slice):
        return self.segments[index]

    def __iter__(self):
        for s in self.segments:
            yield s

    def __len__(self):
        return self.segments.__len__()

    def concat(self):
        return self.segments.concat()

    def plot(self, segment_index: int | slice | Iterable = None, split_segment: bool = False, *args, label: list = None, plot=None, **kwargs):
        """
        Plot cyclic voltammogram

        Parameters
        ----------
        segment_index : int | slice | Iterable, optional
            plot specific segments, plot all if none, by default None
        split_segment : bool, optional
            split segments into multiple lines, by default False
        label : list, optional
            label of each segment, by default None
        plot : PlotCV, optional
            PlotCV object, by default None
        kwargs
            PlotCV.add_segments kwargs

        Returns
        -------
        PlotCV
            PlotCV object
        """
        plot = plot or PlotCV()
        if segment_index is None: plot.add_segments(self.fullcycle, *args, **kwargs)
        elif isinstance(segment_index, int): self.segments[segment_index].plot(plot=plot, *args, **kwargs)
        else: self.segments[segment_index].plot(split_segment=split_segment, label=label, plot=plot, *args, **kwargs)
        return plot


class OCPdata:
    """
    Open circuit potential data

    Parameters
    ----------
    t : np.ndarray
        time
    V : np.ndarray
        voltage
    """

    def __init__(self, t: np.ndarray, V: np.ndarray):
        self.t = t
        self.V = V

    @staticmethod
    def extract_data_from_chi(file: str = None, string: str = None):
        """
        Extract OCP data from chi760 output file

        Parameters
        ----------
        file : str, optional
            chi760 txt path, by default None
        string : str, optional
            chi760 txt content, by default None

        Returns
        -------
        OCPdata
            OCPdata object
        """
        if string is None:
            with open(file, mode='r', encoding='utf-8') as f:
                data = f.read()
        else:
            data = string
        t, V = np.genfromtxt(io.StringIO(re.search(r"Time/sec, Potential/V\n\n((.|\n)*)\n", data).group(1)), delimiter=', ', unpack=True)
        return OCPdata(t, V)

    @property
    def mean(self):
        return np.mean(self.V)


class PlotCV:
    """
    Plot cyclic voltammogram

    Parameters
    ----------
    fig : plt.Figure, optional
        Figure object, by default None
    ax : matplotlib.axes.Axes, optional
        Axes object, by default None
    subplots_kw : tuple, optional
        plt.subplots kwargs, by default None
    prop_cycle : list, optional
        Axes.set_prop_cycle, by default None
    ax_set
        Axes.set kwargs
    """

    def __init__(self, *, fig: plt.Figure = None, ax: matplotlib.axes.Axes = None, subplots_kw: tuple = None, prop_cycle: list = None, **ax_set):
        if any([fig, ax]): self.fig, self.ax = fig, ax
        else: self.fig, self.ax = plt.subplots(**subplots_kw) if subplots_kw else plt.subplots()
        if prop_cycle is not None: self.ax.set_prop_cycle(color=prop_cycle)
        self.ax.set_xlabel('Voltage (V)')
        self.ax.set_ylabel('Current (A)')
        if ax_set: self.ax.set(**ax_set)

    def add_data(self, V: np.ndarray, I: np.ndarray, *args, **kwargs):
        """
        Add data into cyclic voltammogram

        Parameters
        ----------
        V : np.ndarray
            voltage
        I : np.ndarray
            current
        args
            Axes.plot args
        kwargs
            Axes.plot kwargs
        """
        self.ax.plot(V, I, *args, **kwargs)

    def add_segments(self, segments: CVdata | Segments | Segment, *args, label: list = None, **kwargs):
        """
        Add segments into cyclic voltammogram

        Parameters
        ----------
        segments : CVdata | Segments | Segment
            plot given segments
        label : list, optional
            label of each segment, by default None
        args
            Axes.plot args
        kwargs
            Axes.plot kwargs
        """

        if isinstance(segments, Segment):
            self.add_data(segments.V, segments.I, label=label, *args, **kwargs)
        else:
            for i, s in enumerate(segments):
                self.add_data(s.V, s.I, label=label[i], *args, **kwargs) if label else self.add_data(s.V, s.I, *args, **kwargs)

    def legend(self):
        self.ax.legend()
