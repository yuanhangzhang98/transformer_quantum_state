# -*- coding: utf-8 -*-
"""
Created on Mon May 16 15:38:55 2022

@author: Yuanhang Zhang
"""


from model import TransformerModel
from model_utils import sample, compute_observable
from Hamiltonian import Ising, XYZ
from optimizer import Optimizer
from evaluation import compute_E_sample, compute_magnetization

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_tensor_type(torch.cuda.FloatTensor
                                if torch.cuda.is_available()
                                else torch.FloatTensor)

try:
    os.mkdir('results/')
except FileExistsError:
    pass

system_sizes = torch.tensor([[40]], dtype=torch.int64, device='cpu')
H = Ising(system_sizes[0], periodic=False)
# H = XYZ(system_size[0], periodic=False)

n = H.n
param_dim = H.param_dim
embedding_size = 32
n_head = 8
n_hid = embedding_size
n_layers = 8
dropout = 0
minibatch = 10000
batch = 1000000
max_unique = 1000
ensemble_size = 10
model = TransformerModel(system_sizes, param_dim, embedding_size, n_head, n_hid, n_layers,
                         dropout=dropout, minibatch=minibatch)
num_params = sum([param.numel() for param in model.parameters()])
print('Number of parameters: ', num_params)

name = type(H).__name__

folder = 'results/'
save_str = f'{name}_{embedding_size}_{n_head}_{n_layers}'

model.load_state_dict(torch.load(f'{folder}ckpt_100000_{save_str}_0.ckpt'))

optim = Optimizer(model, [H])

n_data_point = 101
param = torch.tensor([1], dtype=torch.get_default_dtype())

E_samples = np.zeros((n_data_point, ensemble_size))
E_exacts = np.zeros((n_data_point, ensemble_size))
m_samples = np.zeros((n_data_point, ensemble_size))
E_dmrgs = np.load(f'{folder}E_dmrg_40.npy')

dEs = np.zeros((n_data_point, ensemble_size))

h = np.arange(n_data_point) / (n_data_point-1) * 2   # [0, 2]
with torch.no_grad():
    for ensemble_id in range(ensemble_size):
        for i in range(n_data_point):
            param[0] = h[i]
            model.set_param(system_sizes[0], param)

            start = time.time()
            print_str = f'{i} {h[i]:.2f} {ensemble_id} '
            samples, sample_weight = sample(model, batch, max_unique, symmetry=H.symmetry)

            t1 = time.time()

            E = H.Eloc(samples, sample_weight, model)
            E_sample = (E * sample_weight).sum()
            E_sample = E_sample.real.detach().cpu().numpy() / n

            print_str += f'{E_sample:.6f}\t'
            E_dmrg = E_dmrgs[i] / n
            print_str += f'{E_dmrg:.6f}\t'

            dE = np.abs((E_sample - E_dmrg) / E_dmrg)
            print_str += f'{dE:.6f}\t'

            t2 = time.time()

            # (n_op, batch)
            samples_pm = 2 * samples - 1
            mz = (samples_pm.mean(dim=0).abs() * sample_weight).sum().detach().cpu().numpy()

            print_str += f'{mz:.6f}\t'
            E_samples[i, ensemble_id] = E_sample
            m_samples[i, ensemble_id] = mz
            dEs[i, ensemble_id] = dE
            t3 = time.time()
            print_str += f'{t1-start:.4f} {t2-t1:.4f} {t3-t2:.4f}'
            print(print_str)

with open(f'results/E_sample_{save_str}.npy', 'wb') as f:
    np.save(f, E_samples)
with open(f'results/m_sample_{save_str}.npy', 'wb') as f:
    np.save(f, m_samples)
with open(f'results/dE_{save_str}.npy', 'wb') as f:
    np.save(f, dEs)

