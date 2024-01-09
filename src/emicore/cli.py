import re
from collections import namedtuple
from argparse import Namespace
from typing import NamedTuple
from functools import wraps, partial

import click
import torch

from .gp import KERNELS, INDUCERS
from .bo import OneShotOptimizer, GradientDescentOptimizer, LBFGSOptimizer
from .bo import SMOOptimizer, EMICOREOptimizer
from .bo import ExpectedImprovement, LowerConfidenceBound


class FinalProperties:
    def __init_subclass__(cls, *args, **kwargs):
        updates = []
        for key, value in cls.__dict__.items():
            if key[:2] != '__' and callable(value) and hasattr(value, '_final_properties'):
                updates += [(name, property(partial(value, key=name))) for name in value._final_properties]

        for name, prop in updates:
            setattr(cls, name, prop)

    def __init__(self):
        self._dict = {}


def final_property(func):
    @property
    @wraps(func)
    def wrapped(self):
        try:
            return self._dict[func.__name__]
        except KeyError:
            result = func(self)
            self._dict[func.__name__] = result
            return result
    return wrapped


def final_properties(*names):
    def wrapping(func):
        @wraps(func)
        def wrapped(self, key):
            try:
                return self._dict[key]
            except KeyError:
                result = func(self)
                self._dict.update(zip(names, result))
                return self._dict[key]

        wrapped._final_properties = names
        return wrapped
    return wrapping


def csobj(dtype, sep=',', maxsplit=-1, length=-1):
    def wrapped(string):
        if isinstance(string, tuple):
            return string
        result = [dtype(elem) for elem in string.split(sep, maxsplit) if elem]
        if length > 0 and len(result) != length:
            raise RuntimeError(f'Invalid number of fields. Provided {len(result)} but expected {length}!')
        return result
    return wrapped


def _append_param(func, param):
    if isinstance(func, click.Command):
        func.params.append(param)
    else:
        if not hasattr(func, '__click_params__'):
            func.__click_params__ = []
        func.__click_params__.append(param)


def namedtuple_as_dict(input):
    if isinstance(input, tuple) and hasattr(input, '_asdict'):
        return namedtuple_as_dict(input._asdict())
    if isinstance(input, Namespace):
        return namedtuple_as_dict(vars(Namespace))
    if isinstance(input, dict):
        return {key: namedtuple_as_dict(value) for key, value in input.items()}
    if isinstance(input, (tuple, list)):
        return type(input)(namedtuple_as_dict(value) for value in input)
    if hasattr(input, '_dictsource'):
        return input._dictsource
    return input


class Data(NamedTuple):
    x: torch.Tensor
    y: torch.Tensor


class TrueSolution(NamedTuple):
    e0: torch.Tensor
    e1: torch.Tensor
    wf: torch.Tensor


class PositiveFloatParam(click.ParamType):
    name = 'positive_float'

    def convert(self, value, param, ctx):
        if not isinstance(value, float):
            value = float(value)
        if value <= 0.:
            self.fail(f'Value {value} is non-positive!', param, ctx)
        return value


PositiveFloat = PositiveFloatParam()


class AnnotatedFuncFromDict(click.ParamType):
    name = 'AnnotatedFuncFromDict'

    def __init__(self, fndict, partial=False):
        super().__init__()
        self.fndict = fndict
        self.partial = partial

    def convert(self, value, param, ctx):
        if callable(value):
            return value

        name, *kwargstr = value.split(':')
        if name not in self.fndict:
            available = '\', \''.join(self.fndict)
            raise click.BadParameter(
                f"No such function: '{name}'. Available functions are '{available}'")
        func = self.fndict[name]

        if func is None:
            return None

        if isinstance(func, type):
            annotations = func.__init__.__annotations__
        else:
            annotations = func.__annotations__

        kwargtups = dict([obj.split('=', 1) for obj in kwargstr if obj])
        missing = set(kwargtups).difference(annotations)
        if missing:
            invalid = '\', \''.join(missing)
            available = '\', \''.join(annotations)
            raise click.BadParameter(
                f"No such arguments for function '{name}': '{invalid}'. "
                f"Valid arguments are: '{available}'"
            )

        kwargs = {key: annotations[key](val) for key, val in kwargtups.items()}

        if self.partial:
            retval = partial(func, **kwargs)
        else:
            retval = func(**kwargs)

        try:
            retval._dictsource = {'funcname': name, **kwargs}
        except AttributeError:
            pass

        return retval


class OptionParams(click.ParamType):
    name = 'OptionParams'
    _rexp = re.compile(
        r'(?P<key>[^\s#=]+)=(?P<value>[^#,\n]+)[,\n]?|'
        r'(?P<comment>#[^\n]*$)|'
        r'(?P<whitespace>\s+)|'
        r'(?P<error>.+)',
        re.MULTILINE
    )

    @classmethod
    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)

        cls._types = {
            key: value
            for key, value in cls.__annotations__.items()
            if key not in click.ParamType.__annotations__
        }
        cls._defaults = {}
        cls._help = {}
        for key in cls._types:
            obj = getattr(cls, key, None)
            if isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[1], str):
                cls._defaults[key] = obj[0]
                cls._help[key] = obj[1]
            else:
                cls._defaults[key] = obj

        cls._return_type = namedtuple(
            cls.__name__,
            list(cls.__annotations__),
            defaults=[cls._defaults[key] for key in cls.__annotations__]
        )

    @classmethod
    def _parse(cls, string):
        for mt in cls._rexp.finditer(string):
            if mt['error'] is not None:
                raise RuntimeError(f'Parsing options failed at \'{mt["error"]}\'!')
            if mt.lastgroup in ('key', 'value'):
                yield (mt['key'], mt['value'])

    @classmethod
    def _members(cls):
        for key, dtype in cls._types.items():
            yield (key, dtype, cls._help.get(key, ''), cls._defaults.get(key, None))

    @classmethod
    def options(cls, prefix='', namespace=None, help=None):
        prefix = prefix.replace('_', '-')

        def decorator(func):
            for key, dtype, dhelp, default in cls._members():
                key = key.replace('_', '-')
                param = click.Option(
                    [f'--{prefix}-{key}' if prefix else f'--{key}'],
                    type=dtype,
                    help=dhelp,
                    default=default,
                    callback=cls._callback(namespace),
                    expose_value=False
                )
                _append_param(func, param)
            return func
        return decorator

    @classmethod
    def _callback(cls, namespace=None):
        def callback(ctx, param, value):
            key = param.name.replace('-', '_')
            if namespace:
                setattr(ctx.params.setdefault(namespace, Namespace()), key, value)
            else:
                ctx.params[key] = value
            return value
        return callback

    def convert(self, value, param, ctx):
        retval = {}
        if isinstance(value, str):
            for key, val in self._parse(value):
                key = key.replace('-', '_')
                if key not in self._types:
                    raise RuntimeError(f'No such option for {param.name}: {key}!')
                retval[key] = self._types[key](val)
        if isinstance(value, self._return_type):
            return value

        return self._return_type(**retval)

    def __repr__(self):
        members = {
            key.replace('_', '-'): (getattr(dtype, '__name__', str(dtype)), self._defaults.get(key, None))
            for key, dtype in self._types.items()
        }
        msg = ', '.join(sorted([
            f'{key}:{dtype}={default}' for key, (dtype, default) in members.items()
        ]))

        return f'{type(self).__name__}({msg})'

    def get_metavar(self, param):
        members = {
            key.replace('_', '-'): (
                getattr(dtype, '__name__', str(dtype)), self._defaults.get(key, None), self._help.get(key, '')
            )
            for key, dtype in self._types.items()
        }
        lines = [
            (f'{" " * 6}{key}={default},', f'  # <{dtype}> {dhelp}')
            for key, (dtype, default, dhelp) in members.items()
        ]
        maxlen = max(len(line[0]) for line in lines)
        msg = '\n'.join(sorted([
            f'{head:{maxlen}s}{tail}' for head, tail in lines
        ]))

        return f'\'\n{msg}\n\''

    @classmethod
    @property
    def defaults(cls):
        for key, dtype in cls._types.items():
            val = cls._defaults.get(key, None)
            yield (key, dtype(val) if val is not None else val)


OPTIMIZER_SETUPS = {
    'oneshot': (OneShotOptimizer, ()),
    'gd': (GradientDescentOptimizer, ('lr', 'n_iter')),
    'lbfgs': (LBFGSOptimizer, ('max_iter', 'max_eval', 'max_ls', 'gtol')),
    'smo': (SMOOptimizer, ('stabilize_interval',)),
    'emicore': (EMICOREOptimizer, (
        'stabilize_interval',
        'gridsize',
        'pairsize',
        'samplesize',
        'core_trials',
        'corethresh',
        'corethresh_width',
        'corethresh_scale',
        'coremin_scale',
        'smo_steps',
        'smo_axis',
        'pivot_steps',
        'pivot_scale',
        'pivot_mode',
    )),
}


ACQUISITION_FNS = {
    'lcb': (LowerConfidenceBound, {'beta': 0.1}),
    'ei': (ExpectedImprovement, {}),
}


class QCParams(OptionParams):
    n_layers: int = 1, 'Number of circuit layers'
    n_qbits: int = 2, 'Number of QBits'
    sector: int = 1, 'Sector -1 or 1'
    n_readout: int = 0, 'Number of shots'
    j_coupling: csobj(float, length=3) = '1,1,1', 'Nearest Neigh. interaction coupling'
    h_coupling: csobj(float, length=3) = '1,1,1', 'External magnetic field coupling'
    pbc: click.BOOL = True, 'Set Periodic/Open Boundary Conditions PBC or OBC. PBC default'
    circuit: click.Choice(['generic', 'esu2']) = 'generic', 'Circuit name'  # noqa: F821
    noise_level: float = 0.0, 'Circuit noise level'
    assume_exact: click.BOOL = False, 'Assume energy is exact or an estimate.'
    cache: click.Path(dir_okay=False) = None, 'Cache for ground state wave function'
    train_data: click.Path(exists=True) = None

class KernelParams(OptionParams):
    sigma_0: PositiveFloat = None, 'Prior variance'
    gamma: PositiveFloat = 2.0, 'Kernel width parameter'


class GPParams(OptionParams):
    kernel: click.Choice(list(KERNELS)) = 'vqe', 'Name of the kernel'
    reg_term: PositiveFloat = 1e-10, 'Observation noise'
    reg_term_estimates: int = 16, 'Number of estimates for reg_term'
    kernel_params: KernelParams() = '', 'Kernel options'
    prior_mean: click.BOOL = False, 'Setting non zero mean if True'
    inducer: AnnotatedFuncFromDict(INDUCERS) = None, 'Method of inducing point selection'


class AcqParams(OptionParams):
    func: click.Choice(list(ACQUISITION_FNS)) = 'ei', ''  # noqa: F821
    optim: click.Choice(list(OPTIMIZER_SETUPS)) = 'oneshot', ''
    lr: PositiveFloat = 1., ''
    n_iter: int = None, ''
    max_iter: int = 15000, ''
    max_eval: int = 15000, ''
    max_ls: int = None, ''
    gtol: PositiveFloat = 1e-70, ''
    stabilize_interval: int = None, ''
    pairsize: int = 20, ''
    gridsize: int = 100, ''
    samplesize: int = 100, ''
    corethresh: PositiveFloat = 1.0, ''
    corethresh_width: int = 10, ''
    corethresh_scale: float = 1.0, ''
    coremin_scale: float = 0.0, ''
    core_trials: int = 0, ''
    smo_steps: int = 100, ''
    smo_axis: click.BOOL = False, ''
    pivot_steps: int = 0, ''
    pivot_scale: float = 1.0, ''
    pivot_mode: click.Choice(['smo', 'loop']) = 'smo', ''  # noqa: F821


class HyperParams(OptionParams):
    optim: click.Choice(['grid', 'none']) = 'grid', ''  # noqa: F821
    loss: click.Choice(['mll', 'loo']) = 'loo', ''  # noqa: F821
    lr: PositiveFloat = 1e-4, ''
    threshold: PositiveFloat = 0.0, ''
    steps: int = 200, ''
    interval: str = '', ''
    max_gamma: PositiveFloat = 20.0, ''


class BOParams(OptionParams):
    train_samples: int = 1, ''
    candidate_samples: int = 500, ''
    candidate_shots: int = 10, ''
    n_iter: int = 50, 'Iterations for Bayesian Optimization'
    acq_params: AcqParams() = '', ''
    hyperopt: HyperParams() = '', ''
    iter_mode: click.Choice(['step', 'qc', 'readout']) = 'step', ''  # noqa: F821
