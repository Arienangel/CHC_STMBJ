from .common import *
import re, io


class Segment:

    def __init__(self, V: np.ndarray, I: np.ndarray):
        self.V = V
        self.I = I

    @staticmethod
    def concat(segment: list):
        V = np.concatenate([s.V for s in segment])
        I = np.concatenate([s.I for s in segment])
        return Segment(V, I)


class CVdata:

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
        self.segment = [Segment(V=Vsegment[i], I=Isegment[i]) for i in range(len(Vsegment))]
        self.Vinit = Vinit or V[0]
        self.Vmax = Vmax or V.max()
        self.Vmin = Vmin or V.min()
        self.Vfinal = Vfinal or V[-1]
        self.polarity = polarity or (V[1] - V[0]) > 0
        self.scan_rate = scan_rate
        self.num_segment = num_segment or len(Vsegment)
        self.interval = interval or np.abs((V[1] - V[0]))
        self.quiet_time = quiet_time
        self.sensitivity = sensitivity

    @staticmethod
    def extract_data_from_chi(file: str = None, string: str = None):
        if file is not None:
            with open(file, mode='r', encoding='utf-8') as f:
                data = f.read()
        else:
            data = string
        V, I = np.genfromtxt(io.StringIO(re.search(r"Potential/V, Current/A\n\n((.|\n)*)\n", data).group(1)), delimiter=', ', unpack=True)
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

    def __getitem__(self, segment_index: Union[int, slice, tuple]):
        if isinstance(segment_index, tuple): return [self.segment[i] for i in segment_index]
        else: return self.segment[segment_index]

    def plot(self, segment_index: Union[int, list[int]] = None, set_legend: bool = False, *args, **kwargs):
        plot = PlotCV(xlim=(self.Vmin - 0.1, self.Vmax + 0.1))
        if segment_index is None: plot.add_segment(self.fullcycle, *args, **kwargs)
        else:
            for i in segment_index:
                plot.add_segment(self[i], label=i+1, *args, **kwargs)
            if set_legend: plot.ax.legend()
        return plot


class OCPdata:

    def __init__(self, t: np.ndarray, V: np.ndarray):
        self.t = t
        self.V = V

    @staticmethod
    def extract_data_from_chi(file: str = None, string: str = None):
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

    def __init__(self,
                 *,
                 fig: plt.Figure = None,
                 ax: matplotlib.axes.Axes = None,
                 figsize: tuple = None,
                 prop_cycle: list = None,
                 **kwargs):

        if any([fig, ax]): self.fig, self.ax = fig, ax
        else: self.fig, self.ax = plt.subplots(figsize=figsize) if figsize else plt.subplots()
        if prop_cycle is not None: self.ax.set_prop_cycle(color=prop_cycle)
        self.ax.set_xlabel('Voltage (V)')
        self.ax.set_ylabel('Current (A)')
        if kwargs: self.ax.set(**kwargs)

    def add_data(self, x: np.ndarray, y: np.ndarray, *args, **kwargs):
        self.ax.plot(x, y, *args, **kwargs)

    def add_segment(self, segment: Union[CVdata, Segment, list], *args, **kwargs):
        if not isinstance(segment, list): segment=[segment]
        for s in segment:
            self.add_data(s.V, s.I, *args, **kwargs)
