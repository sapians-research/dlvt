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
illustrative threshold explicit.

Historical names remain only as compatibility APIs. Each emits a
``DeprecationWarning`` and is scheduled for removal in 3.0.0. The complete
list, canonical replacement first::

    coordination_load           <- complexity
    drain_coefficient_threshold <- carrying_capacity
    trapping_scope_bound        <- trapping_capital_bound
    is_low_vitality             <- is_zombie
    classify_equilibrium        <- classify_regime, dlvt.nondimensional.classify_point
    quasi_static_nullcline      <- dlvt.fastslow.slow_manifold
    sobol_indices(output='low_vitality')
                                <- sobol_indices(output='regime')

Deprecated with no direct replacement — build the map or screen from
continuous ``classify_equilibrium`` outputs plus an explicit display
threshold: ``regime_map``, ``dlvt.nondimensional.zombie_boundary_map``,
``zombie_boundary_map_beta``, and ``lhs_zombie_fraction``.

Two deprecated *mapping keys* survive alongside their canonical twins, since
a dict key cannot warn on access: ``classify_equilibrium(...)['equilibrium']``
carries ``zombie`` beside ``low_vitality``, and
``dlvt.analysis.basin_of_attraction_sweep(...)`` carries ``zombie_target``
beside ``equilibrium_target``. Both are removed in 3.0.0.
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

__version__ = "2.2.0"
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
