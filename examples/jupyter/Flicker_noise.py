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
# %% NP-G plot with specific n
F0 = Flicker_noise.Flicker_noise(sampling_rate=sampling_rate, xscale='log', yscale='log')
n = F0.set_data(G[f], integrand=[100, 1000], n=0)
n
# %% NP-G plotwith auto fit n
FN = Flicker_noise.Flicker_noise(sampling_rate=sampling_rate, xscale='log', yscale='log')
n = FN.set_data(G[f], integrand=[100, 1000], auto_fit=True)
n
# %%
