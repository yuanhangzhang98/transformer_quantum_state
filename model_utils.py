import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.binomial import Binomial


@torch.no_grad()
def sample(model, batch=10000, max_unique=1000, symmetry=None):
    """


    Parameters
    ----------
    model : The transformer model
    batch : int, optional
        Number of samples to generate. The default is 10000.
    max_unique: int, optional
        The maximum number of unique samples to generate. The default is 1000.
    symmetry : defined in symmetry.py, implements the symmetry operation

    Returns
    -------
    samples : (n, batch)
        sampled binary configurations

    TODO: cache the intermediate hidden states for reusing during inference
          can save about half inference time
          example implementation: https://github.com/facebookresearch/fairseq
          may take too much effort; optimize when necessary

    """
    batch0 = batch
    assert model.phys_dim == 2, "Only spin 1/2 systems are supported"
    n = model.system_size.prod()
    samples = torch.zeros(0, 1)
    sample_count = torch.tensor([batch], dtype=torch.int64)
    U1_symm = False
    if symmetry is not None:
        U1_symm = symmetry.U1_symm
    for i in range(n):
        log_amp, = model.forward(samples, compute_phase=False)  # (seq, batch, phys_dim)
        amp = log_amp[-1].exp()  # (batch, phys_dim)
        if U1_symm:
            n_down = samples.sum(dim=0)  # (batch, )
            n_up = samples.shape[0] - n_down
            up_mask = n_up >= n/2
            down_mask = n_down >= n/2
            amp[up_mask, 0] = 0
            amp[down_mask, 1] = 0
        if len(sample_count) < max_unique:
            distribution = Binomial(total_count=sample_count, probs=amp[:, 0])
            zero_count = distribution.sample()  # (batch, )
            one_count = sample_count - zero_count
            sample_count = torch.cat([zero_count, one_count], dim=0)
            mask = sample_count > 0

            batch = samples.shape[1]
            samples = torch.cat([torch.cat([samples, torch.zeros(1, batch)], dim=0),
                               torch.cat([samples, torch.ones(1, batch)], dim=0)], dim=1)
            samples = samples.T[mask].T  # (seq, batch), with updated batch
            sample_count = sample_count[mask]  # (batch, )
        else:
            # do not generate new branches
            sampled_spins = torch.multinomial(amp, 1)  # (batch, 1)
            samples = torch.cat([samples, sampled_spins.T], dim=0)
    if symmetry is not None:
        samples = symmetry.apply_random(samples)

    return samples, sample_count / batch0  # (n, batch), (batch, )


@torch.no_grad()
def sample_without_weight(model, batch=1000, symmetry=None):
    """


    Parameters
    ----------
    model : The transformer model
    batch : int, optional
        Number of samples to generate. The default is 10000.
    symmetry : defined in symmetry.py, implements the symmetry operation

    Returns
    -------
    samples : (n, batch)
        sampled binary configurations

    TODO: cache the intermediate hidden states for reusing during inference
          can save about half inference time
          example implementation: https://github.com/facebookresearch/fairseq
          may take too much effort; optimize when necessary

    """
    batch0 = batch
    assert model.phys_dim == 2, "Only spin 1/2 systems are supported"
    n = model.system_size.prod()
    samples = torch.zeros((0, batch))
    U1_symm = False
    if symmetry is not None:
        U1_symm = symmetry.U1_symm
    for i in range(n):
        log_amp, = model.forward(samples, compute_phase=False)  # (seq, batch, phys_dim)
        amp = log_amp[-1].exp()  # (batch, phys_dim)
        if U1_symm:
            n_down = samples.sum(dim=0)  # (batch, )
            n_up = samples.shape[0] - n_down
            up_mask = n_up >= n/2
            down_mask = n_down >= n/2
            amp[up_mask, 0] = 0
            amp[down_mask, 1] = 0

        sampled_spins = torch.multinomial(amp, 1)  # (batch, 1)
        samples = torch.cat([samples, sampled_spins.T], dim=0)
    if symmetry is not None:
        samples = symmetry.apply_random(samples)

    return samples


def compute_psi(model, samples, symmetry=None, check_duplicate=True):
    """


    Parameters
    ----------
    model : The transformer model
    samples : Tensor, (n, batch)
        samples drawn from the wave function
    symmetry : defined in symmetry.py, implements the symmetry operation
    check_duplicate : bool, optional
        whether to check for duplicate samples. The default is False.

    Returns
    -------
    log_amp : (batch, )
    log_phase : (batch, )

    extract the relevant part of the distribution, ignore the last output
    and the param distribution
    """

    if symmetry is not None:
        samples, phase = symmetry(samples)
        n_symm, n, batch0 = samples.shape
        samples = samples.transpose(0, 1).reshape(n, -1)  # (n, n_symm*batch0)
        # samples_dec = bin2dec(samples.T, n).reshape(n_symm, batch0)  # (n_symm, batch0)

        # in each symmetry sector, enforce the configuration with the lowest decimal representation to have zero phase
        # phase_start_point = torch.argmin(samples_dec, dim=0)  # (batch0, )
        # phase_idx = (n_symm - phase_start_point) % n_symm  # (batch0, ), index of the phase of the first sample

    if check_duplicate:
        samples, inv_idx = torch.unique(samples, dim=1, return_inverse=True)
    n, batch = samples.shape
    n_idx = torch.arange(n).reshape(n, 1)
    batch_idx = torch.arange(batch).reshape(1, batch)
    spin_idx = samples.to(torch.int64)

    log_amp, log_phase = model.forward(samples, compute_phase=True)  # (seq, batch, phys_dim)
    log_amp = log_amp[:-1]  # (n, batch, phys_dim)
    log_phase = log_phase[:-1]  # (n, batch, phys_dim)

    log_amp = log_amp[n_idx, batch_idx, spin_idx].sum(dim=0)  # (batch, )
    log_phase = log_phase[n_idx, batch_idx, spin_idx].sum(dim=0)  # (batch, )

    if check_duplicate:
        log_amp = log_amp[inv_idx]
        log_phase = log_phase[inv_idx]
    if symmetry is not None:
        log_amp = log_amp.reshape(n_symm, batch0)
        log_phase = log_phase.reshape(n_symm, batch0)
        # we are computing the phase of the first sample in the symmetry sector

        # .angle() produces nan in certain cases, use .atan2() as a temporary workaround
        # log_phase = (phase[phase_idx] * ((log_amp + 1j * log_phase) / 2).exp().mean(dim=0)).angle() * 2  # (batch0, )
        # log_phase = (phase[phase_idx] * ((log_amp + 1j * log_phase) / 2).exp().mean(dim=0))
        log_phase = (((log_amp + 1j * log_phase) / 2).exp().mean(dim=0))
        log_phase = log_phase.imag.atan2(log_phase.real) * 2  # (batch0, )
        log_amp = log_amp.exp().mean(dim=0).log()  # (batch0, )
    return log_amp, log_phase


def compute_grad(model, samples, sample_weight, Eloc, symmetry=None):
    """


    Parameters
    ----------
    model : The transformer model
    samples : (n, batch)
        batched sample from the transformer distribution
    sample_weight: (batch, )
        weight for each sample
    Eloc : (batch, ), complex tensor
        local energy estimator
    symmetry : defined in symmetry.py, implements the symmetry operation

    Returns
    -------
    None.

    Computes Gk = <<2Re[(Eloc-<<Eloc>>) Dk*]>>
    where Dk = d log Psi / d pk, pk is the NN parameter

    Note: since the transformer wavefunction is normalized, we should have
    <<Dk>> = 0, and Gk has the simplified expression
    Gk = <<2Re[Eloc Dk*]>>
    TODO: Check this

    """

    log_amp, log_phase = compute_psi(model, samples, symmetry, check_duplicate=False)
    E_model = (Eloc * sample_weight).sum().detach()  # (1, )
    scale = 1 / E_model.abs()
    if scale > 5:
        scale = 5
    E = Eloc - E_model  # (batch, )

    loss = ((E.real * log_amp + E.imag * log_phase) * sample_weight).sum() * scale
    return loss, log_amp, log_phase
    # loss.backward()


@torch.no_grad()
def compute_observable(model, samples, sample_weight, observable, batch_mean=True, symmetry=None):
    """


    Parameters
    ----------
    model : The transformer model
    samples : (n_param+n, batch, input_dim)
        samples drawn from the wave function
    sample_weight: (batch, )
        weight for each sample
    observable: tuple,
        (['XX', 'YY', 'ZZ'], [coef_XX, coef_YY, coef_ZZ], spin_idx)
        grouping up operators that act on the same indices to speed up
        (e.g., interaction in the Heisenberg model)
        pauli_str: string made up of 'X', 'Y', or 'Z', Pauli matrices
        coef: (1, ), (n_op, ) or (n_op, batch), coefficient of operator
        spin_idx: (n_op, n_site), indices that the Pauli operators act on
    batch_mean: bool, whether return the mean value over the batch or not
    symmetry : defined in symmetry.py, implements the symmetry operation

    Returns
    -------
    O: list, [value_XX, value_YY, value_ZZ], values of computed observables
        value:   (n_op, ) if batch_mean is True
            else (n_op, batch)

    Computes the expectation of observables, specified with Pauli strings
    """
    pauli_strs, coefs, spin_idx = observable
    n_type = len(pauli_strs)
    # ord('X')=88, maps 'X' to 0, 'Y' to 1, 'Z' to 2
    pauli_num = torch.tensor([[ord(c) - 88 for c in str_i] for str_i in pauli_strs])  # (n_type, n_site)
    X_sites = pauli_num == 0
    Y_sites = pauli_num == 1
    Z_sites = pauli_num == 2
    flip_sites = X_sites | Y_sites  # (n_type, n_site)
    phase_sites = Y_sites | Z_sites  # (n_type, n_site)

    # group up the computations that can be done at the same time
    # example: XX and YY share flip, while YY and ZZ share phase up to a constant
    flip_sites, inv_flip_idx = torch.unique(flip_sites, dim=0, return_inverse=True)  # (n_unique, n_site)
    phase_sites, inv_phase_idx = torch.unique(phase_sites, dim=0, return_inverse=True)  # (n_unique, n_site)
    flip_results = []
    phase_results = []

    # Y = -i Z X
    # compute phase like Z, flip like X, then account for the additional -i
    Y_count = Y_sites.sum(dim=1)  # (n_type, )
    Y_phase = torch.tensor([1, -1j, -1, 1j])[Y_count % 4]

    if flip_sites.any():
        log_amp, log_phase = compute_psi(model, samples, symmetry, check_duplicate=True)

    if phase_sites.any():
        spin_pm = (1 - 2 * samples).to(torch.get_default_dtype())  # +-1, (n, batch)

    for flip_sites_i in flip_sites:
        if flip_sites_i.any():
            flip_idx = spin_idx.T[flip_sites_i].T  # (n_op, n_flip)
            psixp_over_psix = compute_flip(model, samples, flip_idx, symmetry, log_amp, log_phase)  # (n_op, batch)
            flip_results.append(psixp_over_psix)
        else:
            flip_results.append(torch.ones(1))

    for phase_sites_i in phase_sites:
        if phase_sites_i.any():
            phase_idx = (spin_idx.T[phase_sites_i]).T
            Oxxp = compute_phase(spin_pm, phase_idx)  # (n_op, batch)
            phase_results.append(Oxxp)
        else:
            phase_results.append(torch.ones(1))

    results = []
    for i in range(n_type):
        coef = coefs[i]
        if not isinstance(coef, torch.Tensor):
            coef = torch.tensor(coef)
        if len(coef.shape) < 2:
            coef = coef.reshape(-1, 1)
        result_i = Y_phase[i] * phase_results[inv_phase_idx[i]] * flip_results[inv_flip_idx[i]]  # (n_op, batch)
        result_i = coef * result_i  # (n_op, batch)
        results.append(result_i)

    if batch_mean:
        results = [(sample_weight * result_i).mean(dim=1) for result_i in results]

    return results


def compute_flip(model, samples, flip_idx, symmetry, log_amp, log_phase):
    """


    Parameters
    ----------
    model: the transformer model
    samples : Tensor, (n, batch)
        samples drawn from the wave function
    flip_idx : Tensor, (n_op, n_flip)
        indices with either X or Y acting on it
    symmetry : defined in symmetry.py, implements the symmetry operation
    log_amp : Tensor, (batch, )
    log_phase : Tensor, (batch, )
        pre-computed wave function psi(x)

    Returns
    -------
    psi(x') / psi(x) : (n_op, batch)

        O_loc(x) = O_{x, x'} psi(x') / psi(x)
        This function computes psi(x') / psi(x) when x'!=x
    """

    n, batch = samples.shape
    n_op, n_flip = flip_idx.shape

    samples_flipped = samples.expand(n_op, -1, -1).transpose(0, 1).clone()  # (n, n_op, batch)
    flip_mask = torch.zeros_like(samples_flipped, dtype=torch.bool)
    #         (n_op, n_flip)        (n_op, 1)  indices selected: (n_op, n_flip, batch)
    flip_mask[flip_idx, torch.arange(n_op).unsqueeze(1), :] = 1
    samples_flipped[flip_mask] = 1 - samples_flipped[flip_mask]

    log_amp_1, log_phase_1 = compute_psi(model, samples_flipped.reshape(n, n_op * batch),
                                         symmetry, check_duplicate=True)  # (n_op*batch)
    log_amp_1 = log_amp_1.reshape(n_op, batch)
    log_phase_1 = log_phase_1.reshape(n_op, batch)

    # log_amp_1 = []
    # log_phase_1 = []
    #
    # for i in range(n_op):
    #     flip_idx_i = flip_idx[i]
    #     samples_flipped = samples.clone()
    #     samples_flipped[flip_idx_i] = 1 - samples_flipped[flip_idx_i]
    #     log_amp_i, log_phase_i = compute_psi(model, samples_flipped, symmetry, check_duplicate=True)
    #     log_amp_1.append(log_amp_i)
    #     log_phase_1.append(log_phase_i)
    #
    # log_amp_1 = torch.cat(log_amp_1, dim=0).reshape(n_op, batch)
    # log_phase_1 = torch.cat(log_phase_1, dim=0).reshape(n_op, batch)

    return (((log_amp_1 - log_amp) + 1j * (log_phase_1 - log_phase)) / 2).exp()


def compute_phase(spin_pm, phase_idx):
    """


    Parameters
    ----------
    spin_pm : Tensor, (n, batch)
        +-1, sampled spin configurations
    phase_idx : Tensor, (n_op, n_phase)
        indices with either Y or Z acting on it
        additional -i and spin flip for Y are computed outside this function

    Returns
    -------
    O_{x, x'} : (n_op, batch)
        where x is given
        O_loc(x) = O_{x, x'} psi(x') / psi(x)
    """
    n, batch = spin_pm.shape
    spin_pm_relevant = spin_pm[phase_idx.unsqueeze(-1), torch.arange(batch)]  # (n_op, n_phase, batch), +-1
    return spin_pm_relevant.prod(dim=1)  # (n_op, batch)

