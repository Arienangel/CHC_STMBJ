from . import IVscan, STM_bj, CV, Flicker_noise


def _mpl_setup():
    import matplotlib
    matplotlib.rc('font', size=12)
    matplotlib.rc("figure", autolayout=True, figsize=(4.8, 3.6))


_mpl_setup()
