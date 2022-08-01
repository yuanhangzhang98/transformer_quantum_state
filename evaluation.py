import torch
from model_utils import sample, compute_observable, compute_psi


@torch.no_grad()
def compute_E_sample(model, system_size, param, H, batch=10000, max_unique=1000):
    model.set_param(system_size, param)
    samples, sample_weight = sample(model, batch, max_unique)
    E = H.Eloc(samples, sample_weight, model)
    return (E * sample_weight).sum()


@torch.no_grad()
def compute_magnetization(model, system_size, param, batch=10000, max_unique=1000, symmetry=None):
    model.set_param(system_size, param)
    samples, sample_weight = sample(model, batch, max_unique)
    n = system_size.prod()
    O = (['X', 'Y', 'Z'], [1, 1, 1], torch.arange(n).reshape(n, 1))
    magnetization = compute_observable(model, samples, sample_weight, O, batch_mean=True, symmetry=symmetry)
    magnetization = torch.tensor([mi.mean() for mi in magnetization])  # (3, )
    return magnetization  # (3, )
