#%%
%load_ext autoreload
%autoreload 2
%cd ../..
# %% extract full cycle
from CHClab import IVscan
import numpy as np
import matplotlib.pyplot as plt

IVfull = IVscan.extract_data([r"./examples/test_data/IEbias.txt"], upper=0.75, lower=-0.75, length_segment=640, num_segment=4, offset=[640, 640], units=[1e-6, 1], mode='height', tolerance=1)
IVfull.shape
# %% plot It histogram
Ifull, Vfull = IVfull
hIVt = IVscan.Hist_IVt(tlim=(0, 0.096),
                       Ilim=(1e-11, 1e-5),
                       Vlim=(-1, 1),
                       num_t_bins=384,
                       num_I_bins=200,
                       xscale='linear',
                       y1scale='log',
                       y2scale='linear',
                       x_conversion=40000,
                       Vtype='bias',
                       subplots_kw=dict(figsize=(16, 6)))
hIVt.add_data(Ifull, Vfull)
hIVt.trace
# %% extract segments and remove noise
Iseg, Vseg = IVscan.extract_data(np.stack([Ifull.ravel(), Vfull.ravel()]), upper=0.75, lower=-0.75, length_segment=640, num_segment=1, offset=[0, 0], units=[1, 1], mode='height', tolerance=1)
Iseg, Vseg = IVscan.noise_remove(Iseg, Vseg, V0=0, dV=0.2, I_limit=1e-5)
Iseg, Vseg = IVscan.zeroing(Iseg, Vseg, V0=0)
Gseg = IVscan.conductance(Iseg, Vseg)
Iseg.shape
# %% plot IV histogram
hIV = IVscan.Hist_IV(Vlim=(-0.8, 0.8), Ilim=(1e-11, 1e-5), num_V_bins=400, num_I_bins=600, xscale='linear', yscale='log', Vtype='bias', subplots_kw=dict(figsize=(8, 6)))
hIV.add_data(Iseg, Vseg)
# %% plot GV histogram and fit master curve
hGV = IVscan.Hist_GV(Vlim=(-0.8, 0.8), Glim=(1e-5, 1e-1), num_V_bins=400, num_G_bins=400, xscale='linear', yscale='log', Vtype='bias', subplots_kw=dict(figsize=(8, 6)))
hGV.add_data(G=Gseg, V=Vseg)
# %% simple filter and fit master curve
f = ((Gseg > 1e-3) & (np.abs(Vseg) < 0.05)).sum(axis=1) > 10
fig, ax = plt.subplots(ncols=2, figsize=(16, 6))
hIVa = IVscan.Hist_IV(Vlim=(-0.8, 0.8), Ilim=(1e-11, 1e-5), num_V_bins=400, num_I_bins=600, xscale='linear', yscale='log', Vtype='bias', fig=fig, ax=ax[0], set_colorbar=False)
hIVa.add_data(Iseg[f], Vseg[f])
pIa, _ = hIVa.plot_fitting(axis='x', p0=[0.1, -6.5, 1], bounds=[[0, -11, 0], [np.inf, -5, np.inf]], sigma=[-1, 0, 1], default_values=[0, -10, 0], color='k', linestyle='--')
hIVb = IVscan.Hist_IV(Vlim=(-0.8, 0.8), Ilim=(1e-11, 1e-5), num_V_bins=400, num_I_bins=600, xscale='linear', yscale='log', Vtype='bias', fig=fig, ax=ax[1], set_colorbar=False)
hIVb.add_data(Iseg[~f], Vseg[~f])
pIb, _ = hIVb.plot_fitting(axis='x', p0=[0.1, -8.5, 1], bounds=[[0, -11, 0], [np.inf, -5, np.inf]], sigma=[-1, 0, 1], default_values=[0, -10, 0], color='k', linestyle='--')
hIVa.trace, hIVb.trace
# %% kmeans clustering
from sklearn.cluster import KMeans

GV = np.stack([np.histogram2d(Vseg[i], Gseg[i], [np.linspace(-0.8, 0.8, 40), np.logspace(-5, -1, 40)])[0].ravel() for i in range(Vseg.shape[0])])
n_clusters = 2
seed = np.random.randint(2**31)
kmeans = KMeans(n_clusters=n_clusters, random_state=seed).fit(GV)
fig, ax = plt.subplots(ncols=2, figsize=(16, 6))
for i in range(n_clusters):
    h = IVscan.Hist_IV(Vlim=(-0.8, 0.8), Ilim=(1e-11, 1e-5), num_V_bins=400, num_I_bins=600, xscale='linear', yscale='log', Vtype='bias', fig=fig, ax=ax[i], set_colorbar=False)
    h.add_data(Iseg[kmeans.labels_ == i], Vseg[kmeans.labels_ == i])
seed
# %%
