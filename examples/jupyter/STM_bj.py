#%%
%load_ext autoreload
%autoreload 2
%cd ../..
# %% extract data
from CHClab import STM_bj
import numpy as np
import matplotlib.pyplot as plt

G = STM_bj.extract_data([r"./examples/test_data/STM_bj.txt"], length=2000, upper=3.2, lower=1e-5, direction='pull')
X = STM_bj.get_displacement(G, zero_point=0.5, x_conversion=800)
G.shape
# %% plot 1D histogram and fit peaks
hG = STM_bj.Hist_G(Glim=(1e-5, 3.2), num_bins=550, x_scale='log', subplots_kw=dict(figsize=(8, 6)), set_grid=True)
hG.add_data(G)
pGa = hG.plot_fitting(x_range=[2e-3, 1e-1], p0=[1, -2, 1], bounds=[[0, -3, 0], [np.inf, -1, 5]])[1]
pGb = hG.plot_fitting(x_range=[2e-5, 5e-4], p0=[1, -4, 1], bounds=[[0, -5, 0], [np.inf, -3, 5]])[1]
pGa, pGb
# %% plot 2D histogram
hGS = STM_bj.Hist_GS(Xlim=(-0.5, 1), Glim=(1e-5, 3.16), num_X_bins=300, num_G_bins=550, xscale='linear', yscale='log', subplots_kw=dict(figsize=(8, 6)), set_colorbar=True)
hGS.add_data(G, X)
# %% simple filter
f = (((G > 2e-5) & (G < 1e-3) & (X > 0.15)).sum(axis=1) < 80)
fig, ax = plt.subplots(ncols=2, figsize=(16, 6))
hGSa = STM_bj.Hist_GS(Xlim=(-0.5, 1), Glim=(1e-5, 3.16), num_X_bins=300, num_G_bins=550, xscale='linear', yscale='log', fig=fig, ax=ax[0], set_colorbar=False)
hGSa.add_data(G[f], X[f])
hGSb = STM_bj.Hist_GS(Xlim=(-0.5, 1), Glim=(1e-5, 3.16), num_X_bins=300, num_G_bins=550, xscale='linear', yscale='log', fig=fig, ax=ax[1], set_colorbar=False)
hGSb.add_data(G[~f], X[~f])
hGSa.trace, hGSb.trace
# %% kmeans clustering
from sklearn.cluster import KMeans

GS = np.stack([np.histogram2d(X[i], G[i], [np.linspace(0.1, 0.5, 40), np.logspace(-5, -1, 40)])[0].ravel() for i in range(G.shape[0])])
n_clusters = 2
seed = np.random.randint(2**31)
k = KMeans(n_clusters=n_clusters, random_state=seed).fit(GS)
fig, ax = plt.subplots(ncols=2, figsize=(16, 6))
for i in range(n_clusters):
    h = STM_bj.Hist_GS(Xlim=(-0.5, 1), Glim=(1e-5, 3.16), num_X_bins=300, num_G_bins=550, xscale='linear', yscale='log', fig=fig, ax=ax[i], set_colorbar=False)
    h.add_data(G[k.labels_ == i], X[k.labels_ == i])
seed
# %% cross-correlation conductance histogram
from matplotlib.colors import SymLogNorm

hGcorr = STM_bj.Hist_Correlation(Glim=(1e-5, 3.2), num_bins=550, x_scale='log', norm=SymLogNorm(0.1, 0.25, -1, 1), subplots_kw=dict(figsize=(8, 6)), set_grid=True)
hGcorr.add_data(G)
#%%
