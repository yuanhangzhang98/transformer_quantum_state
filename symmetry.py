import numpy as np
import torch
from Hamiltonian_utils import dec2bin, bin2dec


def cyclic_permutation_idx(n):
    """
    Returns a list of cyclic permutation indices.

    Args:
        n (int): The number of elements in the permutation.

    Returns:
        idx: (n, n), A list of cyclic permutation indices.
    """
    idx = torch.zeros(n, n, dtype=torch.long)
    idx_i = torch.arange(n)
    for i in range(n):
        idx[i, :] = torch.cat([idx_i[i:], idx_i[:i]])
    return idx


class Symmetry:
    def __init__(self):
        self.permutation = None
        self.phase = None
        self.spin_inv_symm = False
        self.spin_inv_phase = None
        self.U1_symm = False

    def __call__(self, tensor):
        """
        tensor: (n, ...)
        return a tensor with all symmetry operations applied, (n_symm, n, ...)
        """
        tensor = tensor[self.permutation]
        phase = self.phase
        if self.spin_inv_symm:
            tensor = torch.cat([tensor, 1 - tensor], dim=0)
            phase = torch.cat([phase, self.spin_inv_phase * phase], dim=0)

        return tensor, phase

    def apply_random(self, tensor):
        """
        tensor: (n, batch)
        apply a random symmetry operation to the tensor
        """
        n, batch = tensor.shape
        idx = torch.randint(0, len(self.permutation), [batch])
        tensor = tensor[self.permutation[idx], torch.arange(batch).reshape(batch, 1)]  # (batch, n)
        if self.spin_inv_symm:
            inv_mask = torch.randint(0, 2, [batch], dtype=torch.bool)
            tensor[inv_mask] = 1 - tensor[inv_mask]
        return tensor.T

    def apply_with_weight(self, tensor, weight):
        tensor, _ = self(tensor)
        n_symm, n, batch = tensor.shape
        weight = weight.expand(n_symm, batch).reshape(-1)
        weight = weight / weight.sum()
        tensor = tensor.transpose(0, 1).reshape(n, -1)
        tensor, inv_idx = torch.unique(tensor, dim=1, return_inverse=True)
        weight_unique = torch.zeros(tensor.shape[1])
        weight_unique.index_add_(0, inv_idx, weight)
        assert torch.allclose(weight_unique.sum(), torch.tensor(1.))
        return tensor, weight_unique


    def add_symmetry(self, symmetry, *args):
        try:
            symmetry_func = getattr(self, symmetry)
            symmetry_func(*args)
        except AttributeError:
            raise ValueError('Unknown symmetry: {}'.format(symmetry))

    def spin_inversion(self, phase=1):
        self.spin_inv_symm = True
        self.spin_inv_phase = phase

    def U1(self, phase=None):
        self.U1_symm = True


class Symmetry1D(Symmetry):
    def __init__(self, n):
        super(Symmetry1D, self).__init__()
        self.n = n
        self.permutation = torch.arange(n).view(1, n)
        self.phase = torch.ones(1)

    def translation(self, phase=1):
        """
        perm: (batch, n), the permutation to be translated.
        return the translated permutations, (batch * n, n)
        """
        perm = self.permutation  # (batch, n)
        batch, n = perm.shape
        idx = cyclic_permutation_idx(n)  # (n, n)
        perm = perm[torch.arange(batch).reshape(batch, 1, 1), idx]  # (batch, n, n)
        self.permutation = perm.reshape(-1, n)  # (batch * n, n)

        phase_i = phase ** torch.arange(n)  # (n, )
        self.phase = torch.outer(self.phase, phase_i).reshape(-1)

    def reflection(self, phase=1):
        """
        perm: (batch, n), the permutation to be reflected.
        return the reflected permutations, (batch * 2, n)
        """
        perm = self.permutation
        self.permutation = torch.cat([perm, perm.flip(1)], dim=0)  # (batch * 2, n)
        self.phase = torch.cat([self.phase, phase * self.phase])  # (batch * 2)


class Symmetry2D(Symmetry):
    def __init__(self, nx, ny):
        super(Symmetry2D, self).__init__()
        self.nx = nx
        self.ny = ny
        self.n = nx * ny
        self.permutation = torch.arange(self.n).view(1, nx, ny)
        self.phase = torch.ones(1)

    def __call__(self, tensor):
        """
        tensor: (n, ...)
        return a tensor with all symmetry operations applied, (n_symm, n, ...)
        """
        if len(self.permutation.shape) == 3:
            self.permutation = self.permutation.reshape(-1, self.n)
        return super(Symmetry2D, self).__call__(tensor)

    def apply_random(self, tensor):
        if len(self.permutation.shape) == 3:
            self.permutation = self.permutation.reshape(-1, self.n)
        return super(Symmetry2D, self).apply_random(tensor)

    def translation_x(self, phase=1):
        perm = self.permutation
        batch, nx, ny = perm.shape
        idx = cyclic_permutation_idx(nx)  # (nx, nx)
        perm = perm[torch.arange(batch).reshape(batch, 1, 1, 1),
                    idx.reshape(nx, nx, 1),
                    torch.arange(ny)]  # (batch, nx, nx, ny)
        self.permutation = perm.reshape(-1, nx, ny)  # (batch * nx, nx, ny)

        phase_i = phase ** torch.arange(nx)  # (nx, )
        self.phase = torch.outer(self.phase, phase_i).reshape(-1)  # (batch * nx)

    def translation_y(self, phase=1):
        perm = self.permutation
        batch, nx, ny = perm.shape
        idx = cyclic_permutation_idx(ny)  # (ny, ny)
        perm = perm[torch.arange(batch).reshape(batch, 1, 1, 1),
                    torch.arange(nx).reshape(nx, 1, 1),
                    idx.reshape(1, ny, ny)]  # (batch, nx, ny, ny)
        self.permutation = perm.permute(0, 2, 1, 3).reshape(-1, nx, ny)  # (batch * ny, nx, ny)

        phase_i = phase ** torch.arange(ny)  # (ny, )
        self.phase = torch.outer(self.phase, phase_i).reshape(-1)  # (batch * ny)

    def reflection_x(self, phase=1):
        perm = self.permutation
        self.permutation = torch.cat([perm, perm.flip(1)], dim=0)  # (batch * 2, nx, ny)
        self.phase = torch.cat([self.phase, phase * self.phase])  # (batch * 2)

    def reflection_y(self, phase=1):
        perm = self.permutation
        self.permutation = torch.cat([perm, perm.flip(2)], dim=0)  # (batch * 2, nx, ny)
        self.phase = torch.cat([self.phase, phase * self.phase])  # (batch * 2)

    def rotation_90(self, phase=1):
        """
        perm: (batch, nx, ny), the permutation to be rotated.
        return the rotated permutations, (batch * 4, nx, ny)
        """
        perm = self.permutation
        batch, nx, ny = perm.shape
        assert nx == ny
        perm_1 = perm.permute(0, 2, 1).flip(2)
        perm_2 = perm.flip(1, 2)
        perm_3 = perm.permute(0, 2, 1).flip(1)
        self.permutation = torch.cat([perm, perm_1, perm_2, perm_3], dim=0)  # (batch * 4, nx, ny)
        self.phase = torch.cat([self.phase,
                                phase * self.phase,
                                phase ** 2 * self.phase,
                                phase ** 3 * self.phase])  # (batch * 4)

    def rotation_180(self, phase):
        """
        perm: (batch, nx, ny), the permutation to be rotated.
        return the rotated permutations, (batch * 2, nx, ny)
        """
        perm = self.permutation
        self.permutation = torch.cat([perm, perm.flip(1, 2)], dim=0)  # (batch * 2, nx, ny)
        self.phase = torch.cat([self.phase, phase * self.phase])  # (batch * 2)


class Symmetry_psi:
    def __init__(self, n):
        self.n = n
        basis = dec2bin(torch.arange(2**self.n), self.n)
        basis_t = torch.cat([basis[:, 1:], basis[:, :1]], dim=1)
        self.translation_idx = bin2dec(basis_t, self.n).to(torch.int64).cpu().numpy()
        self.reflection_idx = bin2dec(basis.flip(1), self.n).to(torch.int64).cpu().numpy()
        self.inversion_idx = bin2dec(1-basis, self.n).to(torch.int64).cpu().numpy()

    def translation(self, psi):
        # psi = psi[torch.arange(psi.shape[0]).unsqueeze(1), self.translation_idx]
        psi = psi[self.translation_idx]
        return psi

    def reflection(self, psi):
        # psi = psi[torch.arange(psi.shape[0]).unsqueeze(1), self.reflection_idx]
        psi = psi[self.reflection_idx]
        return psi

    def inversion(self, psi):
        # psi = psi[torch.arange(psi.shape[0]).unsqueeze(1), self.inversion_idx]
        psi = psi[self.inversion_idx]
        return psi


class Symmetry2D_psi:
    def __init__(self, nx, ny):
        self.nx = nx
        self.ny = ny
        self.n = nx * ny
        basis = dec2bin(torch.arange(2**(nx*ny)), nx*ny).reshape(-1, nx, ny)
        basis_tx = torch.cat([basis[:, 1:, :], basis[:, :1, :]], dim=1)
        basis_ty = torch.cat([basis[:, :, 1:], basis[:, :, :1]], dim=2)
        basis_rx = basis[:, :, :].flip(1)
        basis_ry = basis[:, :, :].flip(2)
        basis_r90 = basis.permute(0, 2, 1).flip(2)
        self.tx_idx = bin2dec(basis_tx.reshape(-1, nx*ny), nx*ny).to(torch.int64).cpu().numpy()
        self.ty_idx = bin2dec(basis_ty.reshape(-1, nx*ny), nx*ny).to(torch.int64).cpu().numpy()
        self.rx_idx = bin2dec(basis_rx.reshape(-1, nx*ny), nx*ny).to(torch.int64).cpu().numpy()
        self.ry_idx = bin2dec(basis_ry.reshape(-1, nx*ny), nx*ny).to(torch.int64).cpu().numpy()
        self.r90_idx = bin2dec(basis_r90.reshape(-1, nx*ny), nx*ny).to(torch.int64).cpu().numpy()
        self.inversion_idx = bin2dec(1 - basis.reshape(-1, nx*ny), self.n).to(torch.int64).cpu().numpy()

    def translation_x(self, psi):
        # psi = psi[torch.arange(psi.shape[0]).unsqueeze(1), self.tx_idx]
        psi = psi[self.tx_idx]
        return psi

    def translation_y(self, psi):
        # psi = psi[torch.arange(psi.shape[0]).unsqueeze(1), self.ty_idx]
        psi = psi[self.ty_idx]
        return psi

    def reflection_x(self, psi):
        # psi = psi[torch.arange(psi.shape[0]).unsqueeze(1), self.rx_idx]
        psi = psi[self.rx_idx]
        return psi

    def reflection_y(self, psi):
        # psi = psi[torch.arange(psi.shape[0]).unsqueeze(1), self.ry_idx]
        psi = psi[self.ry_idx]
        return psi

    def rotation_90(self, psi):
        # psi = psi[torch.arange(psi.shape[0]).unsqueeze(1), self.r90_idx]
        psi = psi[self.r90_idx]
        return psi

    def inversion(self, psi):
        # psi = psi[torch.arange(psi.shape[0]).unsqueeze(1), self.inversion_idx]
        psi = psi[self.inversion_idx]
        return psi
