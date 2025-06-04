#%%
%load_ext autoreload
%autoreload 2
%cd ../..
# %% extract full cycle
from CHClab import IVscan
import numpy as np
import matplotlib.pyplot as plt

IVfull = IVscan.extract_data([r"./examples/test_data/IEwk.txt"], upper=0.95, lower=-0.15, length_segment=2400, num_segment=4, offset=[4800, 4800], units=[1e-6, 1], mode='height', tolerance=1)
IVfull.shape
# %% plot It histogram
Ifull, Vfull = IVfull
hIVt = IVscan.Hist_IVt(tlim=(0, 0.48),
                       Ilim=(1e-11, 1e-5),
                       Vlim=(-0.2, 1.0),
                       num_t_bins=480,
                       num_I_bins=200,
                       xscale='linear',
                       y1scale='log',
                       y2scale='linear',
                       x_conversion=40000,
                       Vtype='wk',
                       subplots_kw=dict(figsize=(16, 6)))
hIVt.add_data(Ifull, Vfull)
hIVt.trace
# %% extract segments and remove noise
Iseg, Vseg = IVscan.extract_data(np.stack([Ifull.ravel(), Vfull.ravel()]), upper=0.95, lower=-0.15, length_segment=2400, num_segment=1, offset=[0, 0], units=[1, 1], mode='height', tolerance=1)
Iseg.shape
# %% plot GV histogram
hGV = IVscan.Hist_GV(Vlim=(-0.2, 1.0), Glim=(5e-5, 5e-1), num_V_bins=600, num_G_bins=400, xscale='linear', yscale='log', Vtype='wk', subplots_kw=dict(figsize=(8, 6)))
hGV.add_data(I=Iseg, V=Vseg, Vbias=0.05)
# %%
