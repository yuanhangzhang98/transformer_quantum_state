"""
An attempt on per-layer stochastic reconfiguration algorithm
Very unstable and performs rather poorly
May contain bugs
Needs improvements in the future
"""


import torch

class SR:
    def __init__(self, model, l=1.0, l_min=1e-3, l_decay_rate=0.99):
        self.model = model
        self._supported_layers = ['Linear']  # Supported layer class types
        self.l = l
        self.l_min = l_min
        self.l_decay_rate = l_decay_rate

    def step(self, sample_weight=None):
        """Performs one step of preconditioning."""
        for mod in self.model.modules():
            if mod.__class__.__name__ not in self._supported_layers:
                continue
            params = []
            grads = []
            grad1 = []
            grad2 = []
            for param in mod.parameters():
                if hasattr(param, 'grad1'):
                    params.append(param)
                    grads.append(param.grad)
                    grad1.append(param.grad1)
                    grad2.append(param.grad2)
            grads = torch.cat([grad_i.reshape(-1) for grad_i in grads])  # (n_param, )
            grad1 = torch.cat([grad1_i.reshape(grad1_i.shape[0], -1) for grad1_i in grad1], dim=1)  # (batch, n_param)
            grad2 = torch.cat([grad2_i.reshape(grad2_i.shape[0], -1) for grad2_i in grad2], dim=1)  # (batch, n_param)

            if sample_weight is not None:
                batch = grad1.shape[0]
                batch0 = len(sample_weight)
                seq_len = int(batch / batch0)
                assert batch == batch0 * seq_len
                grad1 = grad1.reshape(seq_len, batch0, -1)
                grad2 = grad2.reshape(seq_len, batch0, -1)
                grad1_mean = sample_weight @ grad1.mean(dim=0)
                grad2_mean = sample_weight @ grad2.mean(dim=0)
                S = (torch.einsum('b, abi, abj -> ij', sample_weight, grad1, grad1)
                     + torch.einsum('b, abi, abj -> ij', sample_weight, grad2, grad2)) / seq_len \
                    - grad1_mean.outer(grad1_mean) - grad2_mean.outer(grad2_mean)
            else:
                grad1_mean = grad1.mean(dim=0)
                grad2_mean = grad2.mean(dim=0)
                S = (grad1.t() @ grad1 + grad2.t() @ grad2) / grad1.shape[0] \
                    - grad1_mean.outer(grad1_mean) \
                    - grad2_mean.outer(grad2_mean)
            S += torch.eye(S.shape[0]) * self.l

            # compute S_inv @ grads
            # symmetric positive definite, use cholesky decomposition
            # u = torch.linalg.cholesky(S)
            # preconditioned_grad = torch.cholesky_solve(grads.view(-1, 1), u).view(-1)
            preconditioned_grad = torch.linalg.solve(S, grads.view(-1, 1)).view(-1)

            pointer = 0
            for param in params:
                param.grad = preconditioned_grad[pointer:pointer + param.numel()].view(param.shape)
                pointer += param.numel()

            self.l = max(self.l * self.l_decay_rate, self.l_min)
