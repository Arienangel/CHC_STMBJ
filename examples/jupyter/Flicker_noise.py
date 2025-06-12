#%%
%load_ext autoreload
%autoreload 2
%cd ../..
# %% extract data
from CHClab import Flicker_noise
import numpy as np
import matplotlib.pyplot as plt

sampling_rate = 40000
G = Flicker_noise.extract_data([r"./examples/test_data/Flicker_noise.txt"], length=6000, start_from=4000, zero_point=0.5, units=[1e-6, 1])
G.shape
# %% simple filter
f = ((G < 1e-5).sum(axis=1) < 1000) & (np.log10(np.abs(G)).std(axis=1) < 1)
# %% calculate PSD, NP (optional)
PSD, freq = Flicker_noise.PSD(G[f], sampling_rate=sampling_rate)
NP = Flicker_noise.noise_power(PSD, freq, integrand=[100, 1000])
Gmean = G[f].mean(axis=1)
# %% Flicker noise data
F = Flicker_noise.Flicker_noise_data(sampling_rate=sampling_rate, xscale='log', yscale='log', int_method='trapezoid')
F.add_data(G[f], integrand=[100, 1000])
# %% NP-G plot with auto fit n
F.get_scatter(auto_fit=True)
F.scatter.n
# %% NP-G histogram with contour
F.get_hist(Glim=(1e-5, 1), NPlim=(1e-5, 1), num_G_bins=50, num_NP_bins=50, set_colorbar=False, n=1.2)
z, params=F.hist.plot_contour(x_range=[1e-5, 1], y_range=[1e-5, 1], p0=[1, -3, -3, 1, 1, 0], bounds=[[0, -5, -5, 0, 0, -np.pi], [np.inf, 0, 0, 2, 2, np.pi]])
params
# %%
