import os
import numpy as np
import torch

import scipy
from scipy.optimize import Bounds
from scipy.optimize import minimize

# from torchmin import minimize_constr
from model import TransformerModel
from model_utils import compute_psi, sample
from Hamiltonian import Ising

from distutils.version import StrictVersion
assert StrictVersion(scipy.__version__) >= StrictVersion('1.7.1'), \
    'Scipy version must be >= 1.7.1'


class ParamPredictor:
    def __init__(self, model, H):
        self.model = model
        self.H = H

    def NLL(self, param, samples, sample_weight, system_size, symmetry=None):
        self.model.set_param(system_size, param)
        log_amp, _ = compute_psi(self.model, samples, symmetry)
        return -(log_amp * sample_weight).sum()

    def predict_MLE(self, samples, sample_weight, param_range=None, param0=None):
        @torch.no_grad()
        def fn(param):
            param = torch.tensor(param)
            return self.NLL(param, samples, sample_weight, self.model.system_size, self.H.symmetry)\
                    .detach().cpu().numpy().item()
            # return self.NLL(param, samples, sample_weight, self.model.system_size, self.H.symmetry)
        if param_range is None:
            param_range = self.H.param_range
        if param0 is None:
            param0 = param_range[0] + (param_range[1] - param_range[0]) / 2
            # param0 = param_range[0] + (param_range[1] - param_range[0]) * torch.rand(param_range[0].shape)

        param_range = param_range.detach().cpu().numpy()
        param0 = param0.detach().cpu().numpy()
        bounds = Bounds(param_range[0], param_range[1])

        print('Start minimization...')

        result = minimize(fn, param0, bounds=bounds, method='nelder-mead', tol=1e-9, options={'disp': True})
        # result = minimize_constr(fn, param0,
        #                          bounds={'lb': param_range[0],
        #                                  'ub': param_range[1]},
        #                          max_iter=100,
        #                          disp=3)

        print(f'Minimization finished. Result: {result.x}')
        return result.x


if __name__ == '__main__':
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
    max_unique = 100

    model = TransformerModel(system_sizes, param_dim, embedding_size, n_head, n_hid, n_layers,
                             dropout=dropout, minibatch=minibatch)
    num_params = sum([param.numel() for param in model.parameters()])
    print('Number of parameters: ', num_params)
    name = type(H).__name__
    folder = 'results/'
    data_folder = 'data/'
    save_str = f'{name}_{embedding_size}_{n_head}_{n_layers}'

    model.load_state_dict(torch.load(f'{folder}ckpt_100000_{save_str}_0.ckpt'))

    param_range = torch.tensor([[0.5], [1.5]], dtype=torch.get_default_dtype())
    param_predictor = ParamPredictor(model, H)

    n_measures = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    ensemble_size = 10
    n_data_point = 51
    params = torch.linspace(0.5, 1.5, n_data_point).unsqueeze(1)
    predictions_MLE = np.zeros((len(n_measures), n_data_point, ensemble_size))

    for ensemble_id in range(ensemble_size):
        for i, n_measure in enumerate(n_measures):
            for j, param in enumerate(params):
                measurements = np.load(f'{data_folder}Ising_{param.cpu().numpy().item():.2f}_DMRG.npy')  # (batch, n)
                idx = torch.multinomial(torch.ones(len(measurements)), n_measure, replacement=False)
                samples = torch.tensor(measurements, dtype=torch.get_default_dtype())
                samples = samples[idx]
                samples, sample_weight = torch.unique(samples, dim=0, return_counts=True)
                samples = samples.T
                sample_weight = sample_weight / sample_weight.sum()
                param_predicted_MLE = param_predictor.predict_MLE(samples, sample_weight, param_range, param0=None)
                predictions_MLE[i, j, ensemble_id] = param_predicted_MLE
                print(f'{ensemble_id} {i} {j}  param: {param}  MLE: {param_predicted_MLE}')
            with open(f'{folder}predictions_MLE.npy', 'wb') as f:
                np.save(f, predictions_MLE)
