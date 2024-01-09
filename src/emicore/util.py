import hashlib
import logging
import math
import os
import pickle
from functools import partial
from itertools import accumulate

import h5py
import numpy as np
import torch

from .qc import measure_energy, measure_energy_variance, measure_overlap, exact_spectrum, circuit_param_size


class SingularGramError(RuntimeError):
    pass


def grid_search_gamma(model, min_gamma=1.0, max_gamma=30.0, num=20, skip_middle=True, loss='loo'):
    '''Grid search best gamma and change model in-place.'''
    if len(model.x_train) <= 1:
        # cannot do grid search with less than 2 samples
        return model.kernel.gamma
    if skip_middle:
        gammas = np.linspace(min_gamma, max_gamma, num + 1)
        gammas = np.concatenate([gammas[:num // 2], gammas[num // 2 + 1:]])
    else:
        gammas = np.linspace(min_gamma, max_gamma, num)
    # gammas = gammas[(1. < gammas) * (gammas < 10.)]

    lossfn = {
        'loo': model.loocv_mll_closed,
        'mll': model.log_likelihood,
    }[loss]

    # fall back to original gamma if best
    sq_loss = [(lossfn().item(), model.kernel.gamma.item())]
    for gamma in gammas:
        model.kernel.gamma[()] = gamma
        try:
            model.reinit()
        except RuntimeError:
            logging.warning(f'SingularGramError on gamma = {gamma:.3e}. Skipping...')
            continue
        sq_loss.append((lossfn().item(), gamma))

    _, best_gamma = max(sq_loss)
    model.kernel.gamma[()] = best_gamma
    model.reinit()
    return best_gamma


def interval_schedule(param_string):
    '''Create a function from a schedule string in the form of ``[m*]n+[...]``, where * repeats the same number n, m
    times, and each number n specifies after how many steps the returned function should evaluate to True.
    The last number n is repeated forever.
    For example, ``'1+2*2+5'`` evaluates to True at steps ``0, 2, 4, 9, 14, 19, ...``.
    When the evaluated schedule is empty, the returned function always evaluates to ``False``.
    '''
    if not param_string:
        return lambda step: False
    if not isinstance(param_string, str):
        return param_string
    params = [[int(sub) for sub in elem.split('*', maxsplit=1)] for elem in param_string.split('+')]
    schedule = sum([elem if len(elem) == 1 else elem[0] * [elem[1]] for elem in params], [])
    if not schedule:
        return lambda step: False
    final_interval = schedule[-1]
    final_sum = sum(schedule[:-1])
    schedule = set(accumulate(schedule[:-1]))

    def hitstep(step):
        if step + 1 >= final_sum:
            return (step + 1 - final_sum) % final_interval == 0
        return step + 1 in schedule
    return hitstep


def arrhash(*args):
    hasher = hashlib.sha256()
    for arg in args:
        if isinstance(arg, (np.ndarray, torch.Tensor)):
            arr = np.array(arg)
            flag = arr.flags.writeable
            arr.flags.writeable = False
            hasher.update(arr.data)
            arr.flags.writeable = flag
        else:
            hasher.update(pickle.dumps(arg))
    return hasher.hexdigest()


def arrcache(fname, func, identifiers, keys='value'):
    if fname is None:
        return func()
    single = isinstance(keys, str)
    if single:
        keys = (keys,)

    identifier = arrhash(*identifiers)
    results = None

    if os.path.exists(fname):
        with h5py.File(fname, 'r') as fd:
            if identifier in fd:
                results = tuple(
                    torch.from_numpy(dset[()]) if dset.attrs.get('type', 'numpy') == 'torch' else dset[()]
                    for dset in (fd[f'{identifier}/{key}'] for key in keys)
                )

    if results is None:
        results = func()
        if single:
            results = (results,)
        try:
            with h5py.File(fname, 'a') as fd:
                for key, result in zip(keys, results):
                    fd[f'{identifier}/{key}'] = result
                    fd[f'{identifier}/{key}'].attrs['type'] = 'torch' if isinstance(result, torch.Tensor) else 'numpy'
        except OSError as error:
            raise RuntimeError(f'Unable to cache key \'{identifier}\'.') from error

    if single:
        results, = results
    return results


class DataSampler:
    def __init__(
        self,
        n_qbits,
        n_layers,
        j=(1., 1., 1.),
        h=(1., 1., 1.),
        rng=None,
        sector=-1,
        noise_level=0.,
        n_readout=0,
        pbc=True,
        circuit='esu2',
        cache_fname='',
    ):
        self.param_shape = (circuit_param_size(circuit, n_layers), n_qbits)
        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng

        self.kwargs = {
            'n_qbits': n_qbits,
            'n_layers': n_layers,
            'j': j,
            'h': h,
            'circuit': circuit,
            'mom_sector': sector,
            'noise_level': noise_level,
            'n_readout': n_readout,
            'pbc': pbc
        }

        self.energy_fn = partial(
            measure_energy,
            n_qbits,
            n_layers,
            j,
            h,
            mom_sector=self.kwargs['mom_sector'],
            n_readout=self.kwargs['n_readout'],
            pbc=self.kwargs['pbc'],
            circuit=self.kwargs['circuit'],
            noise_level=self.kwargs['noise_level']
        )
        self.energy_var_fn = partial(
            measure_energy_variance,
            n_qbits,
            n_layers,
            j,
            h,
            mom_sector=self.kwargs['mom_sector'],
            n_readout=self.kwargs['n_readout'],
            pbc=self.kwargs['pbc'],
            circuit=self.kwargs['circuit'],
        )

        self.cache_fname = cache_fname
        self._exact_diag = None

    def true_energy(self, angles):
        if isinstance(angles, np.ndarray):
            angles = torch.from_numpy(angles)
        return torch.tensor(self.energy_fn(angles.numpy())).to(angles)

    def true_energy_variance(self, angles):
        if isinstance(angles, np.ndarray):
            angles = torch.from_numpy(angles)
        kwargs = {}
        return self.energy_var_fn(angles, **kwargs)

    def sample(self, n_samples=1000, known=True):
        # expand and fill in template
        x_data = self.rng.uniform(0, 2 * math.pi, (n_samples, *self.param_shape))

        retval = torch.from_numpy(x_data)
        # compute true values if known
        if known:
            retval = (retval, self.true_energy(x_data))
        return retval

    def estimate_variance(self, n_samples):
        x_reg_est = self.sample(n_samples, known=False)
        _, y_reg_est_var = self.true_energy_variance(x_reg_est)
        return y_reg_est_var.mean(0)

    def estimate_ground_state_energy(self):
        return -1.24 * self.kwargs['n_qbits'] - 0.29

    def exact_diag(self):
        if self._exact_diag is None:
            args = (
                int(self.kwargs['n_qbits']),
                tuple(self.kwargs['j']),
                tuple(self.kwargs['h']),
                bool(self.kwargs['pbc']),
            )
            self._exact_diag = arrcache(
                self.cache_fname,
                lambda: exact_diag(*args),
                args,
                keys=('true_e0', 'true_e1', 'true_wf')
            )
        return self._exact_diag

    def exact_overlap(self, angles):
        _, _, exact_wf = self.exact_diag()

        return measure_overlap(
            self.kwargs['n_qbits'],
            self.kwargs['n_layers'],
            angles=np.array(angles),
            exact_wf=exact_wf,
            mom_sector=self.kwargs['mom_sector'],
            circuit=self.kwargs['circuit'],
        )

    def exact_energy(self, angles):
        return self.true_energy(angles)


    @property
    def _cache_identifiers(self):
        return (
            int(self.kwargs['n_qbits']),
            int(self.kwargs['n_layers']),
            tuple(self.kwargs['j']),
            tuple(self.kwargs['h']),
            int(self.kwargs['n_readout']),
            int(self.kwargs['mom_sector']),
            str(self.kwargs['circuit']),
            float(self.kwargs['noise_level']),
            bool(self.kwargs['pbc']),
        )

    def cached_sample(self, n_samples, key='train', force_compute=False):
        return arrcache(
            self.cache_fname if not force_compute else None,
            lambda: self.sample(n_samples, known=True),
            (key, n_samples, *self._cache_identifiers),
            keys=(f'x_{key}', f'y_{key}')
        )


def plot_gp(ax, mean, covar, x_test, kernel, color='b'):
    yerr = 2 * np.diag(covar) ** .5
    ax.plot(x_test, mean, color=color, label=f"{kernel}")
    ax.fill_between(x_test, mean + yerr, mean - yerr, alpha=0.3, color=color)


def exact_diag(n_qbits, j, h, pbc=True):
    exact_eigvals, exact_eigvecs = exact_spectrum(n_qbits, j, h, pbc=pbc)
    true_e0 = exact_eigvals[0]
    true_e1 = exact_eigvals[1]
    true_wf = exact_eigvecs[:, 0]
    return true_e0, true_e1, true_wf
