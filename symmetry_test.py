"""
For testing purposes only
Determining the ground state symmetry of Hamiltonians
"""

import numpy as np
import torch
from scipy.sparse.linalg import eigsh
from Hamiltonian import Ising, XXZ, XYZ, J1J2
from symmetry import Symmetry_psi, Symmetry2D_psi

pi = np.pi

system_size = [10]
n = np.prod(system_size)
# param = np.array([0, 1])
# H = XXZ([n], periodic=True)
# param = 1
# H = Ising([n], periodic=True)
# param = 0.7
# H = J1J2(system_size, periodic=False)
param = torch.tensor([1, 1, 0.2])
H = XYZ(system_size, periodic=False)
full_H = H.full_H(param)


E, psi = eigsh(full_H, k=6, which='SA')
print(E)
E_ground = E[0]

for psi_idx in range(4):
    psi_ground = psi[:, psi_idx]

    E0 = psi_ground.conj().T @ full_H @ psi_ground
    print(E0)

    symm = Symmetry_psi(system_size[0])

    psi_Px = symm.reflection(psi_ground)
    psi_symm_0 = psi_ground + psi_Px
    psi_symm_0 = psi_symm_0 / np.sqrt(psi_symm_0 @ psi_symm_0.conj().T)
    psi_symm_1 = psi_ground - psi_Px
    psi_symm_1 = psi_symm_1 / np.sqrt(psi_symm_1 @ psi_symm_1.conj().T)

    E_symm_0 = psi_symm_0.conj().T @ full_H @ psi_symm_0
    E_symm_1 = psi_symm_1.conj().T @ full_H @ psi_symm_1
    print('Reflection', E_symm_0, E_symm_1)
