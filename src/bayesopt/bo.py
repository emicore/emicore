import logging
import math

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.optimize import minimize

from ..utils import SingularGramError


class BayesianOptimization:
    def __init__(self, model, optimizer, true_fn, error_fn=None):
        self.model = model
        self.optimizer = optimizer
        self.true_fn = true_fn
        self.error_fn = error_fn
        self.optim_state = self.optimizer.init_state(self.model, self.true_fn)

    def step(self, step):
        self.optim_state['step'] = step
        x_best = self.optimizer(self.model, self.optim_state)
        y_best = self.true_fn(x_best)

        try:
            self.optimizer.update(self.model, self.optim_state, x_best, y_best)
        except SingularGramError:
            logging.info('Updated Gram matrix is non-pd! No points added!')

        if self.error_fn is not None:
            return self.error_fn(self.model, self.optim_state)


class Optimizer:
    def __init__(self, acquisition_fn, sampler=None):
        self.acquisition_fn = acquisition_fn
        self.sampler = sampler

    def init_state(self, model, true_fn):
        min_energy_ind = model.y_train.argmin()
        return {
            'step': 0,
            'x_start': model.x_train[min_energy_ind],
            'x_best': model.x_train[min_energy_ind],
            'y_start': model.y_train[min_energy_ind],
            'y_best': model.y_train[min_energy_ind],
        }

    def update(self, model, state, x_measured, y_measured):
        try:
            model.update(x_measured, y_measured)
        finally:
            state['x_start'] = x_measured.squeeze(0)
            state['y_start'] = y_measured
            if state['y_start'].item() < state['y_best'].item():
                state['y_best'] = state['y_start']
                state['x_best'] = state['x_start']

    def __call__(self, model, state):
        pass


class OneShotOptimizer(Optimizer):
    def __call__(self, model, state):
        x_candidates = self.sampler()
        x_candidates = torch.cat((x_candidates, state['x_best'][None, ...]))
        x_values = self.acquisition_fn(model, x_candidates, state['y_best'], state['step'])
        ind = torch.argmax(x_values).item()
        x_best = x_candidates[[ind]]
        return x_best


class GradientDescentOptimizer(Optimizer):
    def __init__(self, *args, n_iter=10, lr=1e-4, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr = lr
        self.n_iter = n_iter

    def __call__(self, model, state):
        x_candidates = self.sampler(lambda x: self.acquisition_fn(model, x, state['y_best'], state['step']))
        x_candidates = torch.cat((x_candidates, state['x_best'][None, ...]))
        x_candidates.requires_grad = True
        optim = torch.optim.Adam((x_candidates,), lr=self.lr)

        scheduler = ReduceLROnPlateau(
            optim,
            factor=0.92,
            verbose=True,
            patience=10,
            threshold=1e-4,
            min_lr=1e-7
        )

        for _ in range(self.n_iter):
            x_values = self.acquisition_fn(model, x_candidates, state['y_best'], state['step'])
            loss = -x_values.sum()
            optim.zero_grad()
            loss.backward(retain_graph=True)
            optim.step()
            scheduler.step(loss)
        ind = torch.argmax(x_values).item()
        x_best = x_candidates[[ind]].detach()
        return x_best


class TorchLBFGSOptimizer(Optimizer):
    def __init__(self, *args, lr=1., max_iter=20, max_eval=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr = lr
        self.max_iter = max_iter
        self.max_eval = max_eval

    def __call__(self, model, state):
        x_candidates = self.sampler(lambda x: self.acquisition_fn(model, x, state['y_best'], state['step']))
        x_candidates = torch.cat((x_candidates, state['x_best'][None, ...]))
        x_candidates.requires_grad = True
        optim = torch.optim.LBFGS((x_candidates,), lr=self.lr, max_iter=self.max_iter, max_eval=self.max_eval)
        x_values = None

        def closure():
            nonlocal x_values
            x_values = self.acquisition_fn(model, x_candidates, state['y_best'], state['step'])
            loss = -x_values.sum()
            optim.zero_grad()
            loss.backward(retain_graph=True)
            return loss

        optim.step(closure)

        ind = torch.argmax(x_values).item()
        x_best = x_candidates[[ind]].detach()
        return x_best


class LBFGSOptimizer(Optimizer):
    def __init__(self, *args, max_iter=None, max_eval=None, max_ls=None, gtol=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.options = {
            'disp': None,
            'maxcor': 10,
            'ftol': 2.220446049250313e-09,
            'gtol': 1e-10 if gtol is None else gtol,
            'eps': 1e-08,
            'maxfun': 15000 if max_eval is None else max_eval,
            'maxiter': 15000 if max_iter is None else max_iter,
            'iprint': -1,
            'maxls': 20 if max_ls is None else max_ls,
            'finite_diff_rel_step': None
        }

    def __call__(self, model, state):
        x_candidates = self.sampler(lambda x: self.acquisition_fn(model, x, state['y_best'], state['step']))
        x_candidates = torch.cat((x_candidates, state['x_best'][None, ...]))
        original_shape = x_candidates.shape

        def closure(x):
            x = torch.from_numpy(x).reshape(original_shape)
            x.requires_grad = True
            acq_values = self.acquisition_fn(model, x, state['y_best'], state['step'])
            loss = -acq_values.sum()
            loss.backward(retain_graph=True)
            return loss.detach().numpy(), x.grad.detach().flatten().numpy()

        x0 = x_candidates.flatten().numpy()
        _, grad = closure(x0)
        result = minimize(
            closure,
            x0,
            method='L-BFGS-B',
            jac=True,
            options=self.options
        )

        if not result.success:
            logging.warning('Optimization of acquisition function was unsuccessful: %s', result.message)

        x_candidates = torch.from_numpy(result.x).reshape(original_shape)
        x_values = self.acquisition_fn(model, x_candidates, state['y_best'], state['step'])
        ind = torch.argmax(x_values).item()
        x_best = x_candidates[[ind]]
        return x_best


def make_shift(pivot, shifts, axis):
    outshape = (*shifts.shape, *pivot.shape)
    coords = torch.cartesian_prod(*[torch.arange(n) for n in shifts.shape], *[torch.tensor((n,)) for n in axis])
    return (
        pivot[(None,) * shifts.ndim].expand(outshape)
        + torch.sparse_coo_tensor(coords.t(), shifts.flatten(), outshape)
    ) % math.tau


class LeastSquaresWave:
    def __init__(self, shifts):
        self.pinv = self.fit(shifts)

    @staticmethod
    def fit(shifts):
        points = torch.tensor([0., *shifts], dtype=torch.float64)
        data = torch.stack([points ** 0., points.cos(), points.sin()], dim=1)
        return torch.linalg.pinv(data)

    def solve(self, pivot, y_pivot, y_shifts, axis):
        c0, c1, c2 = self.pinv @ torch.tensor([y_pivot, *y_shifts], dtype=torch.float64)
        # atan2(c2, c1) + pi is the argmin of c1 * cos x + c2 * sin x
        theta = torch.atan2(c2, c1) + math.pi

        shift = torch.sparse_coo_tensor(list(zip(axis)), theta, pivot.shape)
        return (
            (pivot + shift) % math.tau,
            c0 + c1 * theta.cos() + c2 * theta.sin(),
        )

    def __call__(self, y_pivot, y_shifts, x_cand):
        c0, c1, c2 = self.pinv @ torch.tensor([y_pivot, *y_shifts], dtype=torch.float64)
        return c0 + c1 * x_cand.cos() + c2 * x_cand.sin()


class LineSearchOptimizer(Optimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self, model, state, x_measured, y_measured):
        try:
            model.update(x_measured, y_measured)
        finally:
            state['y_start'] = model.posterior(state['x_start'][None], diag=True).mean

    def choose_axis(self, model, state):
        '''Choose the next axis, setting ``state['k_best']``.'''
        raise NotImplementedError()

    def choose_measurements(self, model, state):
        '''Choose the next shifts from the pivot along the axis.'''
        raise NotImplementedError()

    def choose_pivot(self, model, state):
        '''Given the shift candidates, find the best point on the line.'''
        raise NotImplementedError()

    def require_stabilize(self, model, state):
        '''Given the pivot point, decide whether the current best point should be measured for stabilization.'''
        raise NotImplementedError()

    def __call__(self, model, state):
        if 'k_best' in state:
            # update x_start using previous axis, x_start and x_shift
            state['x_pivot'], state['y_pivot'] = self.choose_pivot(model, state)

            if self.require_stabilize(model, state):
                return state['x_pivot'][None]
            state['x_start'], state['y_start'] = state['x_pivot'], state['y_pivot']

            if state['y_start'].item() < state['y_best'].item():
                state['y_best'] = state['y_start']
        state['k_best'] = self.choose_axis(model, state)
        state['x_meas'] = self.choose_measurements(model, state)

        return state['x_meas']


class SMOOptimizer(LineSearchOptimizer):
    def __init__(self, *args, stabilize_interval=0, shift=math.pi / 3., **kwargs):
        super().__init__(*args, **kwargs)
        self.shifts = (-math.tau / 3., math.tau / 3.)
        self.lsw = LeastSquaresWave(self.shifts)
        self.stabilize_interval = stabilize_interval

    def choose_axis(self, model, state):
        '''Choose the next axis, setting ``state['k_best']``.'''
        shape = state['x_start'].shape
        if 'k_best' in state:
            k_flat = np.ravel_multi_index(state['k_best'], shape) + 1
        else:
            k_flat = 0
        return np.unravel_index(k_flat % shape.numel(), shape)

    def choose_measurements(self, model, state):
        '''Choose the next shifts from the pivot along the axis.'''
        return make_shift(state['x_start'], torch.tensor(self.shifts), state['k_best'])

    def choose_pivot(self, model, state):
        '''Given the shift candidates, find the best point on the line.'''
        y_start = model.posterior(state['x_start'][None], diag=True).mean
        x_pairs = make_shift(state['x_start'], torch.tensor(self.shifts), state['k_best'])
        y_pairs = model.posterior(x_pairs, diag=True).mean
        return self.lsw.solve(state['x_start'], y_start, y_pairs, state['k_best'])

    def require_stabilize(self, model, state):
        '''Given the pivot point, decide whether the current best point should be measured for stabilization.'''
        return state['step'] and self.stabilize_interval and not (state['step'] % self.stabilize_interval)


class EMICOREOptimizer(SMOOptimizer):
    def __init__(
        self,
        *args,
        pairsize=20,
        gridsize=100,
        samplesize=100,
        corethresh=1.0,
        corethresh_width=10,
        core_trials=10,
        smo_steps=100,
        smo_axis=False,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.gridsize = gridsize
        self.pairsize = pairsize
        self.samplesize = samplesize
        self.corethresh = corethresh
        self.corethresh_width = corethresh_width
        self.core_trials = core_trials
        self.smo_steps = smo_steps
        self.smo_axis = smo_axis

    def _emicore(self, model, state, axis):
        single_candidates = torch.linspace(0, math.tau, self.pairsize + 1)[:-1]
        pair_candidates = torch.cartesian_prod(*(single_candidates,) * 2)
        grid_candidates = torch.linspace(0, math.tau, self.gridsize + 1)[:-1]

        x_pairs = make_shift(
            state['x_start'],
            torch.cat((torch.tensor(self.shifts)[None], pair_candidates)),
            axis,
        )
        x_tests = make_shift(
            state['x_start'],
            grid_candidates,
            axis
        )

        peek_distr = model.peek_posterior(x_pairs, x_tests, diag=True)
        post_distr = model.posterior(x_tests, diag=True)
        coremask = peek_distr.std < state.get('corethresh', self.corethresh)
        coremask[:, 0] = True
        core = torch.tensor([torch.inf, 1.])[coremask * 1]

        best_distr = model.posterior(state['x_start'][None], diag=True)
        samplesize = (self.pairsize ** 2 + 1, self.samplesize)
        maximp = (
            best_distr.sample(samplesize).squeeze(-1) - (post_distr.sample(samplesize) * core[..., None, :]).amin(-1)
        ).clip_(min=0.).nan_to_num_(posinf=0.0).mean(-1)

        return x_pairs, maximp

    def choose_measurements(self, model, state):
        if state['step'] < self.smo_steps:
            return super().choose_measurements(model, state)

        if self.corethresh_width > 0:
            state.setdefault('energy_log', []).append(state['y_start'].item())
            if len(state['energy_log']) > self.corethresh_width:
                state['corethresh'] = max(0., (
                        state['energy_log'][-self.corethresh_width - 1] - state['energy_log'][-1]
                    ) / self.corethresh_width
                )

        x_pairs, maximp = self._emicore(model, state, state['k_best'])
        bestind = maximp.argmax()

        return x_pairs[bestind]

    def require_stabilize(self, model, state):
        stabilize = (
            super().require_stabilize(model, state)
            or (
                state.get('core_trial', 0) < self.core_trials
                and (model.posterior(state['x_pivot'][None]).std > state.get('corethresh', self.corethresh)).all()
            )
        )
        state['core_trial'] = (state.get('core_trial', 0) + 1) if stabilize else 0
        return stabilize

    def choose_axis(self, model, state):
        '''Choose the next axis, setting ``state['k_best']``.'''
        if self.smo_axis or state['step'] < self.smo_steps:
            return super().choose_axis(model, state)

        axes = [
            np.unravel_index(k_flat, state['x_start'].shape)
            for k_flat in range(state['x_start'].numel())
            if 'k_best' not in state or np.ravel_multi_index(state['k_best'], state['x_start'].shape) != k_flat
        ]

        axes, maximps = zip(*((axis, self._emicore(model, state, axis)[1].amax(0)) for axis in axes))
        return axes[torch.stack(maximps).argmax()]


class EILVSOptimizer(SMOOptimizer):
    def __init__(
        self,
        *args,
        gridsize=100,
        weighted=False,
        seq_reg=0.0,
        seq_reg_init=-20,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.gridsize = gridsize
        self.weighted = weighted
        self.seq_reg = seq_reg
        self.seq_reg_init = seq_reg_init

    def _axis_distr(self, model, state):
        shape = state['x_start'].shape
        eye = torch.eye(shape.numel())
        if 'k_best' in state:
            k_last = np.ravel_multi_index(state['k_best'], shape)
            eye = torch.cat([eye[:k_last], eye[k_last + 1:]], dim=0)

        shift = (eye * math.pi / 3.)[..., None, :] * torch.tensor([-1., 1.])[None, :, None]
        x_split = (state['x_start'][None, None] + shift.reshape(eye.shape[0], 2, *shape)) % math.tau
        distr = model.posterior(x_split, diag=False)
        return x_split, distr

    def _shift_index(self, k_distr):
        if k_distr.covrank == 2:
            return ()
        if k_distr.covrank == 1:
            return [[k_distr.var.argmax()]]
        if k_distr.covrank == 0:
            return []
        raise RuntimeError('Unexpected rank!')

    def choose_axis(self, model, state):
        '''Choose the next axis, setting ``state['k_best']``.'''
        shape = state['x_start'].shape
        numel = shape.numel()
        lins = torch.linspace(0, math.tau, self.gridsize + 2)[1:-1]
        eye = torch.eye(numel).reshape((numel, *shape))
        if 'k_best' in state:
            k_last = np.ravel_multi_index(state['k_best'], shape)
            eye = torch.cat([eye[:k_last], eye[k_last + 1:]], dim=0)
        x_candidates = (
            state['x_start'][None, None] + lins[(slice(None), None) + (None,) * len(shape)] * eye[None, :]
        ).reshape((self.gridsize, eye.shape[0], *shape)) % math.tau

        # TODO: remove this hack
        model._state = state
        acq_values = self.acquisition_fn(model, x_candidates, state['y_best'], state['step'])
        del model._state

        if self.weighted:
            x_split, distr = self._axis_distr(model, state)
            acq_values = acq_values / (distr.covrank + 1).to(state['x_start'])[None]
            state['min_rank'] = distr.covrank.min().item()

        if self.seq_reg:
            if 'k_best' in state:
                acq_values += (
                    state['step']
                    - torch.cat((state['k_step_last'][:k_last], state['k_step_last'][k_last + 1:]))[None]
                ) / numel
            else:
                state['k_step_last'] = torch.full((numel,), self.seq_reg_init)
                acq_values += (state['step'] - state['k_step_last'][None]) / numel

        if acq_values.isnan().any():
            logging.warning('Encountered nan values in acquisition function!')
        acq_dir_max = acq_values.reshape((self.gridsize, eye.shape[0])).nan_to_num(0).amax(0)
        acq_max = acq_dir_max.argmax()

        state['acq_dir_perc'] = acq_dir_max
        state['acq_max'] = acq_dir_max[acq_max].item()

        k_curr = acq_max + ('k_best' in state and acq_max >= k_last)
        if self.seq_reg:
            state['k_step_last'][k_curr] = state['step']

        return np.unravel_index(k_curr, shape)

    def __call__(self, model, state):
        x_meas = super().__call__(model, state)
        k_distr = model.posterior(x_meas, diag=False)

        index = self._shift_index(k_distr)
        return x_meas[index]


class AcquisitionFunction:
    def __call__(self, model, x_cand, step):
        pass


class ExpectedImprovement(AcquisitionFunction):
    def __call__(self, model, x_cand, y_best, step):
        f_min = y_best.item()
        distr = model.posterior(x_cand, diag=True)

        pdf = distr.pdf(f_min)
        cdf = distr.cdf(f_min)
        return (f_min - distr.mean) * cdf + distr.var * pdf


class WeightedExpectedImprovement(AcquisitionFunction):
    def __call__(self, model, x_cand, y_best, step):
        f_min = y_best.item()
        if x_cand.ndim != 4:
            raise RuntimeError('Number of dimensions does not match (gridsize, directions, wire_params, n_qbits).')

        gridsize, numel, n_params, n_qbits = x_cand.shape
        x_split = x_cand.reshape(gridsize // 2, 2, numel, n_params, n_qbits).permute(0, 2, 1, 3, 4)
        distr = model.posterior(x_split, diag=False)
        ranks = distr.covrank.to(x_cand).mean(0, keepdim=True) + 1.

        # TODO: remove this hack
        if hasattr(model, '_state'):
            model._state['min_rank'] = (ranks - 1.).min().item()

        pdf = distr.pdf(f_min)
        cdf = distr.cdf(f_min)
        result = ((f_min - distr.mean) * cdf + distr.var * pdf) / ranks[..., None]
        return result.permute(0, 2, 1).reshape(gridsize, numel)


class LowerConfidenceBound(AcquisitionFunction):
    def __init__(self, beta=1.0):
        self.beta = beta

    def __call__(self, model, x_cand, y_best, step):
        mean, var = model.posterior(x_cand, diag=True).diag_stats
        return - mean + self.beta * var ** .5


class AdaptiveLCB(AcquisitionFunction):
    def __init__(self, d=1, a=1, b=1, r=2*np.pi, delta=0.01):
        self.d = d
        self.a = a
        self.b = b
        self.r = r
        self.delta = delta

    def __call__(self, model, x_cand, y_best, step):
        mean, var = model.posterior(x_cand, diag=True).diag_stats
        t = step + 1.0
        beta_1 = 2 * np.log(2 * t * np.pi ** 2 / (3 * self.delta))
        beta_2 = 2 * self.d * np.log(
            t ** 2 * self.d * self.b * self.r * np.log(4 * self.d * self.a / self.delta) ** .5
        )
        exploration_term = ((beta_1 + beta_2) * var) ** .5
        exploitation_term = mean
        return - exploitation_term + exploration_term
