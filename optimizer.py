# -*- coding: utf-8 -*-
"""
Created on Sun May 15 22:53:44 2022

@author: Yuanhang Zhang
"""


import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_utils import sample, compute_grad
from evaluation import compute_E_sample, compute_magnetization
import autograd_hacks
from SR import SR


class Optimizer:
    def __init__(self, model, Hamiltonians, point_of_interest=None):
        self.model = model

        self.Hamiltonians = Hamiltonians
        self.model.param_range = Hamiltonians[0].param_range
        self.loss_fn = nn.MSELoss()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9)

        # the following is for per-layer stochastic reconfiguration
        # currently very unstable and performs rather poorly
        # avoid using for now, need future improvements
        self.optim_SR = torch.optim.SGD(self.model.parameters(), lr=1.0)
        self.preconditioner = SR(self.model)

        self.save_freq = 100
        self.ckpt_freq = 10000
        self.point_of_interest = point_of_interest
    
    @staticmethod
    def lr_schedule(step, model_size, factor=5.0, warmup=4000, start_step=0):
        # using the lr schedule from the paper: Attention is all you need
        step = step + start_step
        if step < 1:
            step = 1
        return factor * (model_size ** (-0.5) * min(step**(-0.75), step * warmup ** (-1.75)))

    def minimize_energy_step(self, H, batch, max_unique, use_symmetry=True):
        symmetry = H.symmetry if use_symmetry else None
        samples, sample_weight = sample(self.model, batch, max_unique, symmetry)
        E = H.Eloc(samples, sample_weight, self.model, use_symmetry)
        E_mean = (E * sample_weight).sum()
        E_var = (((E - E_mean).abs()**2 * sample_weight).sum() / H.n**2).detach().cpu().numpy()
        Er = (E_mean.real / H.n).detach().cpu().numpy()
        Ei = (E_mean.imag / H.n).detach().cpu().numpy()
        loss, log_amp, log_phase = compute_grad(self.model, samples, sample_weight, E, symmetry)
        return loss, log_amp, log_phase, sample_weight, Er, Ei, E_var

    def train(self, n_iter, batch=10000, max_unique=1000,
              param_range=None, fine_tuning=False, use_SR=True, ensemble_id=0, start_iter=None):
        name, embedding_size, n_head, n_layers = type(self.Hamiltonians[0]).__name__,\
            self.model.embedding_size, self.model.n_head, self.model.n_layers
        if start_iter is None:
            start_iter = 0 if not fine_tuning else 100000
        system_sizes = self.model.system_sizes
        n_iter += 1
        if param_range is None:
            param_range = self.Hamiltonians[0].param_range
        self.model.param_range = param_range
        save_str = f'{name}_{embedding_size}_{n_head}_{n_layers}_{ensemble_id}' if not fine_tuning\
            else f'ft_{self.model.system_sizes[0].detach().cpu().numpy().item()}_' \
                 f'{param_range[0].detach().cpu().numpy().item():.2f}_' \
                 f'{name}_{embedding_size}_{n_head}_{n_layers}_{ensemble_id}'

        if use_SR:
            optim = self.optim_SR
            autograd_hacks.add_hooks(self.model)
        else:
            optim = self.optim
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optim,
            lambda step: self.lr_schedule(step, self.model.embedding_size, start_step=start_iter))

        if self.point_of_interest is not None:
            size_i, param_i = self.point_of_interest
            H_watch = type(self.Hamiltonians[0])(size_i, self.Hamiltonians[0].periodic)
            if self.Hamiltonians[0].symmetry is None:
                H_watch.symmetry = None
            E_watch = np.zeros(int(np.ceil(n_iter/self.save_freq)))
            m_watch = np.zeros((int(np.ceil(n_iter/self.save_freq)), 3))
            idx = 0

        E_curve = np.zeros(n_iter)
        E_vars = np.zeros(n_iter)

        for i in range(start_iter, start_iter + n_iter):
            start = time.time()
            self.model.set_param()
            size_idx = self.model.size_idx
            n = self.model.system_size.prod()
            H = self.Hamiltonians[size_idx]

            loss, log_amp, log_phase, sample_weight, Er, Ei, E_var = \
                self.minimize_energy_step(H, batch, max_unique, use_symmetry=True)

            t1 = time.time()

            if use_SR:
                autograd_hacks.clear_backprops(self.model)
                optim.zero_grad()
                log_amp.sum().backward(retain_graph=True)
                autograd_hacks.compute_grad1(self.model, loss_type='sum', grad_name='grad1')
                autograd_hacks.clear_backprops(self.model)

                optim.zero_grad()
                log_phase.sum().backward(retain_graph=True)
                autograd_hacks.compute_grad1(self.model, loss_type='sum', grad_name='grad2')
                autograd_hacks.clear_backprops(self.model)

                optim.zero_grad()
                loss.backward()
                autograd_hacks.clear_backprops(self.model)
                self.preconditioner.step(sample_weight)
                optim.step()
            else:
                optim.zero_grad()
                loss.backward()
                optim.step()

            scheduler.step()
            t2 = time.time()

            print_str = f'E_real = {Er:.6f}\t E_imag = {Ei:.6f}\t E_var = {E_var:.6f}\t'
            E_curve[i - start_iter] = Er
            E_vars[i - start_iter] = E_var

            end = time.time()
            print(f'i = {i}\t {print_str} n = {n}\t lr = {scheduler.get_lr()[0]:.4e} t = {(end-start):.6f}  t_optim = {t2-t1:.6f}')
            if i % self.save_freq == 0:
                with open(f'results/E_{save_str}.npy', 'wb') as f:
                    np.save(f, E_curve)
                with open(f'results/E_var_{save_str}.npy', 'wb') as f:
                    np.save(f, E_vars)
                if self.point_of_interest is not None:
                    E_watch[idx] = compute_E_sample(self.model, size_i, param_i, H_watch).real.detach().cpu().numpy()
                    m_watch[idx, :] = compute_magnetization(self.model, size_i, param_i, symmetry=H_watch.symmetry)\
                                        .real.detach().cpu().numpy()
                    idx += 1
                    with open(f'results/E_watch_{save_str}.npy', 'wb') as f:
                        np.save(f, E_watch)
                    with open(f'results/m_watch_{save_str}.npy', 'wb') as f:
                        np.save(f, m_watch)
                torch.save(self.model.state_dict(), f'results/model_{save_str}.ckpt')
                if i % self.ckpt_freq == 0:
                    torch.save(self.model.state_dict(), f'results/ckpt_{i}_{save_str}.ckpt')
