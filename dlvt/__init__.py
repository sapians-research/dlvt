"""Dynamic Leadership Vitality Theory (DLVT) executable model.

DLVT is a regularized two-state ODE model in preparation for peer review.
The state variables are subjective vitality ``V`` and enacted leadership scope
``C``. Experienced coordination load ``O`` and modeled impact flow ``I`` are
derived quantities::

    O = O0 + beta*C**eta
    I = C*V/(1 + phi*O)
    dV/dt = R*(1 - V/Vmax) - delta*O**gamma*V/(V + eps)
    dC/dt = alpha*I - mu*C

The bundled parameter vector is an illustrative reproducibility input, not an
empirical calibration. ``eps`` must be positive for the active boundary and
global results. ``Gamma = delta*O**gamma/R`` is a coefficient ratio; it does
not determine the sign of ``dV/dt``.

Canonical quick start::

    from dlvt import (
        classify_equilibrium,
        coordination_load,
        find_interior_equilibria,
        make_params,
        simulate,
    )

    p = make_params()
    t, V, C, O, I, gamma_ratio = simulate(p, V0=8.0, C0=0.5, T=120)
    equilibrium = find_interior_equilibria(p)[0]
    report = classify_equilibrium(p, threshold_fraction=0.5)

``classify_equilibrium`` reports the continuous equilibrium and makes the
illustrative threshold explicit. Historical names including ``complexity``,
``carrying_capacity``, ``is_zombie``, and ``classify_regime`` remain only as
compatibility APIs and are deprecated or scheduled for removal.
"""

from .model import (
    DEFAULT_PARAMS, PARAMETER_NAMES, make_params, validate_params,
    coordination_load, complexity, impact,
    dlvt_system, dlvt_exogenous, simulate
)
from .analysis import (
    carrying_capacity, drain_coefficient_threshold, trapping_scope_bound,
    trapping_capital_bound,
    find_interior_equilibria,
    jacobian_eigenvalues, is_low_vitality, is_zombie,
    classify_equilibrium, classify_regime, regime_map
)

__version__ = "2.2.0rc1"
__author__ = "W. Bendinelli"
__all__ = [
    # Model
    'DEFAULT_PARAMS', 'PARAMETER_NAMES', 'make_params', 'validate_params',
    'coordination_load', 'complexity', 'impact',
    'dlvt_system', 'dlvt_exogenous', 'simulate',
    # Analysis
    'carrying_capacity', 'drain_coefficient_threshold', 'trapping_scope_bound',
    'trapping_capital_bound',
    'find_interior_equilibria',
    'jacobian_eigenvalues', 'is_low_vitality', 'is_zombie',
    'classify_equilibrium', 'classify_regime', 'regime_map',
]
