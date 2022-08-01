# -*- coding: utf-8 -*-
"""
Created on Mon May 16 12:26:40 2022

@author: Yuanhang Zhang
"""



import os
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import plt_config
from scipy.signal import savgol_filter
from Hamiltonian import Ising, XYZ

color = plt.rcParams['axes.prop_cycle'].by_key()['color']


system_size = [40]
H = Ising(system_size, periodic=False)
n = system_size[0]
param = 1

name = type(H).__name__
embedding_size = 32
n_head = 8
n_layers = 8

save_str = f'{name}_{embedding_size}_{n_head}_{n_layers}'

folder = 'results/'

E_sample = np.load(f'{folder}E_sample_{save_str}.npy')
m_sample = np.load(f'{folder}m_sample_{save_str}.npy')
dE = np.load(f'{folder}dE_{save_str}.npy')

E_90 = np.quantile(E_sample, 0.9, axis=1)
E_50 = np.quantile(E_sample, 0.5, axis=1)
E_10 = np.quantile(E_sample, 0.1, axis=1)
m_90 = np.quantile(m_sample, 0.9, axis=1)
m_50 = np.quantile(m_sample, 0.5, axis=1)
m_10 = np.quantile(m_sample, 0.1, axis=1)
dE_90 = np.quantile(dE, 0.9, axis=1)
dE_50 = np.quantile(dE, 0.5, axis=1)
dE_10 = np.quantile(dE, 0.1, axis=1)

mz_abs_dmrg = np.load(f'{folder}Ising_mz_abs.npy')
mz_abs_dmrg_smoothed = savgol_filter(mz_abs_dmrg, 15, 3)
x_mz_abs_dmrg = np.linspace(0, 2, 201)

n_data_point, ensemble_size = E_sample.shape
h = np.arange(n_data_point) / (n_data_point-1) * 2   # [0, 2]
idx_trained = np.where((h >= 0.5) & (h <= 1.5))[0]
idx_ft_0 = np.where(h <= 0.5)[0]
idx_ft_1 = np.where(h >= 1.5)[0]

n_ft = 11
h_ft = np.linspace(0, 2, n_ft)
ensemble_ft = 10

E_ft = np.load(f'{folder}E_sample_ft_{save_str}.npy')
m_ft = np.load(f'{folder}m_sample_ft_{save_str}.npy')
dE_ft = np.load(f'{folder}dE_ft_{save_str}.npy')

E_ft_90 = np.quantile(E_ft, 0.9, axis=1)
E_ft_50 = np.quantile(E_ft, 0.5, axis=1)
E_ft_10 = np.quantile(E_ft, 0.1, axis=1)

dE_ft_90 = np.quantile(dE_ft, 0.9, axis=1)
dE_ft_50 = np.quantile(dE_ft, 0.5, axis=1)
dE_ft_10 = np.quantile(dE_ft, 0.1, axis=1)

m_ft_90 = np.quantile(m_ft, 0.9, axis=1)
m_ft_50 = np.quantile(m_ft, 0.5, axis=1)
m_ft_10 = np.quantile(m_ft, 0.1, axis=1)


fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.axvline(0.5, 0.03, 0.97, ls='--', color='gray', lw=1)
ax.axvline(1.5, 0.03, 0.97, ls='--', color='gray', lw=1)
ax.plot(h[idx_trained], E_50[idx_trained], label='Pre-trained TQS', color=color[0])
ax.plot(h[idx_ft_0], E_50[idx_ft_0], ls=':', label='Extrapolation', color=color[0])
ax.plot(h[idx_ft_1], E_50[idx_ft_1], ls=':', color=color[0])
ax.errorbar(h_ft, E_ft_50, [E_ft_50-E_ft_10, E_ft_90-E_ft_50],
            label='Fine-tuned TQS', color=color[1], ls='', marker='o', ms=8, capsize=3)
ax.fill_between(h, E_10, E_90, color=color[0], alpha=0.2)
ax.set_xlabel('$h$')
ax.set_ylabel('Energy')
# ax.set_title(name)
ax.legend()
plt.savefig(f'{folder}E_sample_{save_str}.svg', bbox_inches='tight')
# plt.show()
plt.close()

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(h[idx_trained], m_50[idx_trained], label='Pre-trained TQS', color=color[0])
ax.axvline(0.5, 0.03, 0.97, ls='--', color='gray', lw=1)
ax.axvline(1.5, 0.03, 0.97, ls='--', color='gray', lw=1)
ax.plot(h[idx_ft_0], m_50[idx_ft_0], color=color[0], ls=':', label='Extrapolation')
ax.plot(h[idx_ft_1], m_50[idx_ft_1], color=color[0], ls=':')
# ax.errorbar(h, m_50, [m_50-m_10, m_90-m_50])
ax.fill_between(h, m_10, m_90, color=color[0], alpha=0.2)
ax.errorbar(h_ft, m_ft_50, [m_ft_50-m_ft_10, m_ft_90-m_ft_50],
            label='Fine-tuned TQS', color=color[1], ls='', marker='o', ms=8, capsize=3)
# ax.plot(x_mz_abs_dmrg, mz_abs_dmrg, label='DMRG results', color=color[2])
ax.plot(x_mz_abs_dmrg, mz_abs_dmrg_smoothed, ls='--', label='DMRG', color=color[2])
ax.set_xlabel('$h$')
ax.set_ylabel(r'$|\langle \sigma^z \rangle|$')
# ax.set_title(name)
handles, labels = ax.get_legend_handles_labels()
# order = [0, 1, 3, 2]
# ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], fontsize=18)
ax.legend(fontsize=18)
plt.savefig(f'{folder}m_sample_{save_str}.svg', bbox_inches='tight')
# plt.show()
plt.close()

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.axvline(0.5, 0.03, 0.97, ls='--', color='gray', lw=1)
ax.axvline(1.5, 0.03, 0.97, ls='--', color='gray', lw=1)
ax.plot(h[idx_trained], dE_50[idx_trained], label='Pre-trained TQS', color=color[0])
ax.plot(h[idx_ft_0], dE_50[idx_ft_0], ls=':', label='Extrapolation', color=color[0])
ax.plot(h[idx_ft_1], dE_50[idx_ft_1], ls=':', color=color[0])
ax.errorbar(h_ft, dE_ft_50, [dE_ft_50-dE_ft_10, dE_ft_90-dE_ft_50],
            label='Fine-tuned TQS', color=color[1], ls='', marker='o', ms=8, capsize=3)
ax.fill_between(h, dE_10, dE_90, color=color[0], alpha=0.2)
ax.set_xlabel('$h$')
ax.set_ylabel(r'$\Delta E$')
ax.set_yscale('log')
# ax.set_title(name)
handles, labels = ax.get_legend_handles_labels()
order = [0, 1, 2]
# ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], fontsize=18)
ax.legend(fontsize=18)
plt.savefig(f'{folder}dE_{save_str}.svg', bbox_inches='tight')
# plt.show()
plt.close()
