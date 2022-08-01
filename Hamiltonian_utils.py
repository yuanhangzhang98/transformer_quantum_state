import numpy as np
import torch


def dec2bin(x, bits):
    # credit to Tiana
    # https://stackoverflow.com/questions/55918468/convert-integer-to-pytorch-tensor-of-binary-bits
    # mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).to(torch.get_default_dtype())


def bin2dec(b, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
    return torch.sum(mask * b, -1)


def idx_1d_to_nd(i, system_size):
    n_dim = len(system_size)
    mod_num = torch.zeros(n_dim - 1, dtype=torch.int64)
    for d in range(n_dim - 1):
        mod_num[-d - 1] = system_size[-1 - d:].prod()
    idx = []
    for d in range(n_dim - 1):
        idx.append(i // mod_num[d])
        i = i - idx[-1] * mod_num[d]
    idx.append(i)
    idx = torch.stack(idx, dim=1)
    return idx


def idx_nd_to_1d(idx, system_size):
    n_dim = len(system_size)
    mod_num = torch.zeros(n_dim - 1, dtype=torch.int64)
    for d in range(n_dim - 1):
        mod_num[-d - 1] = system_size[-1 - d:].prod()
    i = idx[:, -1]
    for d in range(n_dim - 1):
        i = i + idx[:, d] * mod_num[d]
    return i


def generate_spin_idx(system_size, interaction, periodic=False):
    system_size = torch.tensor(system_size, dtype=torch.int64).reshape(-1)
    n_dim = len(system_size)
    n = system_size.prod()
    if interaction == 'external_field':
        return torch.arange(n).reshape(n, 1)
    elif interaction == 'nearest_neighbor':
        if periodic:
            connection_0 = torch.arange(n)
            nd_idx = idx_1d_to_nd(connection_0, system_size)  # (n, n_dim)
            connection_1 = []
            for i in range(n_dim):
                new_idx = nd_idx.clone()
                new_idx[:, i] = (new_idx[:, i] + 1) % system_size[i]
                connection_1.append(idx_nd_to_1d(new_idx, system_size))
            connection_1 = torch.cat(connection_1, dim=-1)
            connection_0 = connection_0.expand(n_dim, -1).reshape(-1)
            return torch.stack([connection_0, connection_1], dim=1)  # (n*n_dim, 2)
        else:
            connection_0 = []
            connection_1 = []
            for i in range(n_dim):
                system_size_i = system_size.clone()
                system_size_i[i] -= 1
                connection_0_i = torch.arange(system_size_i.prod())
                nd_idx_i = idx_1d_to_nd(connection_0_i, system_size_i)
                new_idx_i = nd_idx_i.clone()
                new_idx_i[:, i] += 1
                connection_1.append(idx_nd_to_1d(new_idx_i, system_size))
                connection_0.append(idx_nd_to_1d(nd_idx_i, system_size))
            connection_0 = torch.cat(connection_0, dim=-1)
            connection_1 = torch.cat(connection_1, dim=-1)
            return torch.stack([connection_0, connection_1], dim=1)
    elif interaction == 'next_nearest_neighbor':
        if n_dim == 1:
            if periodic:
                connection_0 = torch.arange(n)
                nd_idx = idx_1d_to_nd(connection_0, system_size)  # (n, n_dim)
                connection_1 = []
                for i in range(n_dim):
                    new_idx = nd_idx.clone()
                    new_idx[:, i] = (new_idx[:, i] + 2) % system_size[i]
                    connection_1.append(idx_nd_to_1d(new_idx, system_size))
                connection_1 = torch.cat(connection_1, dim=-1)
                connection_0 = connection_0.expand(n_dim, -1).reshape(-1)
                return torch.stack([connection_0, connection_1], dim=1)  # (n*n_dim, 2)
            else:
                connection_0 = []
                connection_1 = []
                for i in range(n_dim):
                    system_size_i = system_size.clone()
                    system_size_i[i] -= 2
                    connection_0_i = torch.arange(system_size_i.prod())
                    nd_idx_i = idx_1d_to_nd(connection_0_i, system_size_i)
                    new_idx_i = nd_idx_i.clone()
                    new_idx_i[:, i] += 2
                    connection_1.append(idx_nd_to_1d(new_idx_i, system_size))
                    connection_0.append(idx_nd_to_1d(nd_idx_i, system_size))
                connection_0 = torch.cat(connection_0, dim=-1)
                connection_1 = torch.cat(connection_1, dim=-1)
                return torch.stack([connection_0, connection_1], dim=1)
        elif n_dim == 2:
            if periodic:
                connection_0 = torch.arange(n)
                nd_idx = idx_1d_to_nd(connection_0, system_size)  # (n, n_dim)
                connection_1 = []
                for i in range(n_dim):
                    new_idx = nd_idx.clone()
                    if i == 0:
                        new_idx[:, 0] = (new_idx[:, 0] + 1) % system_size[0]
                        new_idx[:, 1] = (new_idx[:, 1] + 1) % system_size[1]
                    elif i == 1:
                        new_idx[:, 0] = (new_idx[:, 0] - 1) % system_size[0]
                        new_idx[:, 1] = (new_idx[:, 1] + 1) % system_size[1]
                    else:
                        raise Exception('Invalid dimension for diagonal interaction (expected 2 dims)')
                    connection_1.append(idx_nd_to_1d(new_idx, system_size))
                connection_1 = torch.cat(connection_1, dim=-1)
                connection_0 = connection_0.expand(n_dim, -1).reshape(-1)
                return torch.stack([connection_0, connection_1], dim=1)  # (n*n_dim, 2)
            else:
                connection_0 = []
                connection_1 = []
                for i in range(n_dim):
                    system_size_i = system_size.clone()
                    system_size_i[0] -= 1
                    system_size_i[1] -= 1
                    connection_0_i = torch.arange(system_size_i.prod())
                    nd_idx_i = idx_1d_to_nd(connection_0_i, system_size_i)
                    new_idx_i = nd_idx_i.clone()
                    if i == 0:
                        new_idx_i[:, 0] += 1
                        new_idx_i[:, 1] += 1
                        connection_1.append(idx_nd_to_1d(new_idx_i, system_size))
                        connection_0.append(idx_nd_to_1d(nd_idx_i, system_size))
                    elif i == 1:
                        nd_idx_i[:, 0] += 1
                        new_idx_i[:, 1] += 1
                        connection_1.append(idx_nd_to_1d(new_idx_i, system_size))
                        connection_0.append(idx_nd_to_1d(nd_idx_i, system_size))
                    else:
                        raise Exception('Invalid dimension for diagonal interaction (expected 2 dims)')
                connection_0 = torch.cat(connection_0, dim=-1)
                connection_1 = torch.cat(connection_1, dim=-1)
                return torch.stack([connection_0, connection_1], dim=1)
        else:
            raise NotImplementedError('Next nearest neighbor interaction only implemented for 1 and 2 dims')
    elif interaction == 'nn_horizontal' or interaction == 'nn_vertical':
        assert n_dim == 2, 'Horizontal and vertical interactions only implemented for 2 dims'
        i = 0 if interaction == 'nn_horizontal' else 1
        if periodic:
            connection_0 = torch.arange(n)
            nd_idx = idx_1d_to_nd(connection_0, system_size)  # (n, n_dim)
            connection_1 = []
            new_idx = nd_idx.clone()
            new_idx[:, i] = (new_idx[:, i] + 1) % system_size[i]
            connection_1.append(idx_nd_to_1d(new_idx, system_size))
            connection_1 = torch.cat(connection_1, dim=-1)
            # connection_0 = connection_0.expand(n_dim, -1).reshape(-1)
            return torch.stack([connection_0, connection_1], dim=1)  # (n, 2)
        else:
            connection_0 = []
            connection_1 = []
            system_size_i = system_size.clone()
            system_size_i[i] -= 1
            connection_0_i = torch.arange(system_size_i.prod())
            nd_idx_i = idx_1d_to_nd(connection_0_i, system_size_i)
            new_idx_i = nd_idx_i.clone()
            new_idx_i[:, i] += 1
            connection_1.append(idx_nd_to_1d(new_idx_i, system_size))
            connection_0.append(idx_nd_to_1d(nd_idx_i, system_size))
            connection_0 = torch.cat(connection_0, dim=-1)
            connection_1 = torch.cat(connection_1, dim=-1)
            return torch.stack([connection_0, connection_1], dim=1)
    else:
        raise NotImplementedError(f'Interaction {interaction} is not implemented')