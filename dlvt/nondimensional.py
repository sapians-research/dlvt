"""
dlvt.nondimensional
===================
Nondimensionalization and global-sensitivity utilities for the DLVT model.

This module derives and verifies the reduced (dimensionless) form of the DLVT
system, computes exact structural invariances of the interior equilibrium, and
provides global-sensitivity screening tools (elasticities, threshold-display
maps, and Latin-hypercube sampling with Spearman rank correlations).

Dimensional system (Equations 3.3-3.4 of the paper)
---------------------------------------------------
    dV/dt = R*(1 - V/Vmax) - delta*O^gamma * V/(V + eps)
    dC/dt = alpha*I - mu*C
with
    O = O0 + beta*C^eta          (experienced coordination load)
    I = C*V / (1 + phi*O)        (energy-gated impact)

Derivation of the nondimensional (reduced) form
-----------------------------------------------
Introduce the dimensionless state and time

    v   = V / Vmax               (vitality as a fraction of capacity)
    tau = mu * t                 (time in units of the scope-contraction
                                  time 1/mu)
    w   = (O - O0) / O0          (excess experienced load relative to baseline)
        = beta * C^eta / O0

**Vitality equation.**  Since dV/dt = Vmax * mu * dv/dtau and
O = O0*(1 + w),

    Vmax*mu*dv/dtau = R*(1 - v)
                      - delta*O0^gamma*(1 + w)^gamma * v/(v + eps/Vmax),

so, dividing by mu*Vmax,

    dv/dtau = rho*(1 - v) - kappa*(1 + w)^gamma * v/(v + e),          (N1)

with rho = R/(mu*Vmax), kappa = delta*O0^gamma/(mu*Vmax), e = eps/Vmax.
Equation (N1) is exact for every eta.

**Coordination-load equation (general eta > 0).**  Since
w = beta*C^eta/O0, logarithmic differentiation gives

    dw/dt = eta*w*(1/C)*dC/dt.

Using dC/dt = C*(alpha*V/(1 + phi*O) - mu),

    dw/dt = eta*w*(alpha*Vmax*v / (1 + phi*O0*(1 + w)) - mu).

Dividing by mu (dw/dtau = (1/mu) dw/dt):

    dw/dtau = eta*w*[a*v/(1 + f*(1 + w)) - 1],                         (N2)

with a = alpha*Vmax/mu and f = phi*O0.

**Reduced system (general eta > 0).**

    dv/dtau = rho*(1 - v) - kappa*(1 + w)^gamma * v/(v + e)
    dw/dtau = eta*w*[a*v/(1 + f*(1 + w)) - 1]

For general eta, SEVEN independent dimensionless groups remain out of the
11 raw parameters:

    rho   = R / (mu*Vmax)              relative recovery rate      (base: 1.5)
    kappa = delta*O0^gamma / (mu*Vmax) relative baseline drain     (base: 0.01)
    a     = alpha*Vmax / mu            relative scope gain         (base: 5.0)
    f     = phi*O0                     baseline impact suppression (base: 0.15)
    e     = eps / Vmax                 relative regularisation     (base: 0.01)
    gamma                              drain nonlinearity          (base: 2.0)
    eta                                load-mapping exponent       (base: 1.0)

At the baseline restriction eta=1, six nontrivial groups remain.  Beta is
absorbed entirely into the scope scale: it appears nowhere in (N1)-(N2) and
only re-enters through the dimensional map
C=(O0*w/beta)^(1/eta).  This is the nondimensional statement of scope
absorption.

Exact structural consequences for the interior equilibrium
----------------------------------------------------------
At an interior equilibrium the pair (V*, O*) solves the CLOSED system

    V* = (mu/alpha) * (1 + phi*O*)                       (from dC/dt = 0)
    R*(1 - V*/Vmax) = delta*O*^gamma * V*/(V* + eps)     (from dV/dt = 0)

which involves neither O0, beta, nor eta.  Hence, exactly (not merely to
numerical tolerance):

* V* and O* have ZERO elasticity with respect to beta, eta, and O0
  (only C* = ((O* - O0)/beta)^(1/eta) depends on them);
* V* depends on (mu, alpha) only through the ratio mu/alpha;
* V* depends on (R, delta) only through the ratio R/delta, and the joint
  scaling (R, delta) -> (s*R, s*delta) leaves (V*, C*, O*) unchanged;
* in the eps -> 0 limit with gamma = 2 (and any eta), O* solves the quadratic

      delta*O*^2 + (R/Vmax)*(mu*phi/alpha)*O* + R*((mu/alpha)/Vmax - 1) = 0,

  which at baseline gives O* = 8.93313 and V* = 4.67994 (the eps = 0.1
  regularised values are O* = 9.00843, V* = 4.70253).

The illustrative contraction-to-accumulation ratio (mu/alpha)_crit at which
V* crosses the stipulated 0.5*Vmax display threshold is ~2.163 at baseline phi
(baseline mu/alpha = 2 sits ~8% below the flip).

Note that although the interior equilibrium (V*, O*) is independent of O0,
the transient DYNAMICS are not: O0 enters the reduced vector field through
kappa and f.

Global-sensitivity tools
------------------------
v_star_elasticities()    : central-difference elasticities d ln V*/d ln p_i
mu_alpha_critical()      : bisection for an explicit display-threshold crossing
zombie_boundary_map()    : deprecated compatibility API over (mu/alpha, phi)
zombie_boundary_map_beta(): deprecated compatibility API over (mu/alpha, beta)
                           that the boundary is invariant in beta
lhs_zombie_fraction()    : deprecated compatibility API for threshold screening
                           with Spearman
                           rank correlations.  This is a RANK-CORRELATION
                           SCREENING (a global-sensitivity proxy), not a full
                           variance-based Sobol decomposition.

References
----------
  Bendinelli, W. (2026). Dynamic Leadership Vitality Theory: A Formal Model
  manuscript in preparation.
"""

from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import qmc, spearmanr

from .analysis import (
    DISPLAY_THRESHOLD_FRACTION,
    classify_equilibrium,
    find_interior_equilibria,
)

# The 11 raw model parameters, in canonical order.
PARAM_NAMES: List[str] = [
    'R', 'Vmax', 'delta', 'gamma', 'O0', 'beta', 'eta', 'alpha', 'phi', 'mu',
    'eps',
]


# -- Reduced (nondimensional) form --------------------------------------------

def reduced_groups(p: Dict[str, float]) -> Dict[str, float]:
    """Map dimensional parameters to the reduced dimensionless groups.

    The reduced system (see module docstring) is

        dv/dtau = rho*(1 - v) - kappa*(1 + w)^gamma * v/(v + e)
        dw/dtau = eta*w*(a*v/(1 + f*(1 + w)) - 1)

    Parameters
    ----------
    p : Dict[str, float]
        Dimensional parameter dictionary with ``eta > 0``.

    Returns
    -------
    Dict[str, float]
        Keys: 'rho', 'kappa', 'a', 'f', 'e', 'gamma', 'eta'.
        Baseline values: rho=1.5, kappa=0.01, a=5.0, f=0.15,
        e=0.01, gamma=2, eta=1.

    Raises
    ------
    ValueError
        If ``eta <= 0``.
    """
    if p['eta'] <= 0.0:
        raise ValueError(f"eta must be positive (got eta={p['eta']}).")
    return {
        'rho':   p['R'] / (p['mu'] * p['Vmax']),
        'kappa': p['delta'] * p['O0'] ** p['gamma'] / (p['mu'] * p['Vmax']),
        'a':     p['alpha'] * p['Vmax'] / p['mu'],
        'f':     p['phi'] * p['O0'],
        'e':     p['eps'] / p['Vmax'],
        'gamma': p['gamma'],
        'eta':   p['eta'],
    }


def reduced_rhs(tau: float, y: List[float],
                g: Dict[str, float]) -> List[float]:
    """Right-hand side of the reduced (nondimensional) DLVT system.

    Parameters
    ----------
    tau : float
        Dimensionless time (tau = mu*t); the system is autonomous.
    y : List[float]
        Reduced state [v, w] with v = V/Vmax and w = (O - O0)/O0.
    g : Dict[str, float]
        Dimensionless groups from :func:`reduced_groups`.

    Returns
    -------
    List[float]
        [dv/dtau, dw/dtau].
    """
    v = max(y[0], 0.0)
    w = max(y[1], 0.0)
    dv = g['rho'] * (1.0 - v) \
        - g['kappa'] * (1.0 + w) ** g['gamma'] * v / (v + g['e'])
    dw = g['eta'] * w * (
        g['a'] * v / (1.0 + g['f'] * (1.0 + w)) - 1.0
    )
    return [dv, dw]


def from_dimensional(V: float, C: float,
                     p: Dict[str, float]) -> Tuple[float, float]:
    """Map dimensional state ``(V,C)`` to ``(v,w)`` for any ``eta>0``.

    Parameters
    ----------
    V, C : float
        Dimensional vitality and enacted leadership scope.
    p : Dict[str, float]
        Dimensional parameter dictionary.

    Returns
    -------
    Tuple[float, float]
        ``(v,w) = (V/Vmax, beta*C**eta/O0)``.
    """
    return V / p['Vmax'], p['beta'] * C**p['eta'] / p['O0']


def to_dimensional(v: np.ndarray, w: np.ndarray,
                   p: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Map reduced state ``(v,w)`` back to dimensional ``(V,C)``.

    Beta re-enters only through
    ``C=(O0*w/beta)**(1/eta)``; it does not appear in the reduced dynamics.

    Parameters
    ----------
    v, w : float or ndarray
        Reduced vitality and excess experienced load.
    p : Dict[str, float]
        Dimensional parameter dictionary.

    Returns
    -------
    Tuple[ndarray, ndarray]
        ``(V,C) = (Vmax*v, (O0*w/beta)**(1/eta))``.
    """
    w_arr = np.maximum(np.asarray(w), 0.0)
    C = (p['O0'] * w_arr / p['beta']) ** (1.0 / p['eta'])
    return p['Vmax'] * np.asarray(v), C


def simulate_reduced(p: Dict[str, float], V0: float = 8.0, C0: float = 0.5,
                     T: float = 120.0, t_eval: Optional[np.ndarray] = None,
                     max_step: float = 0.05,
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate the reduced system and map the result back to (t, V, C).

    This is the correctness oracle for the nondimensionalization: for every
    positive eta the returned trajectory must coincide with
    :func:`dlvt.model.simulate` up to integration error.

    Parameters
    ----------
    p : Dict[str, float]
        Dimensional parameter dictionary with positive eta.
    V0, C0 : float, optional
        Dimensional initial conditions (defaults match dlvt.model.simulate).
    T : float, optional
        Dimensional time horizon; the reduced system is integrated over
        tau in [0, mu*T].
    t_eval : ndarray, optional
        Dimensional times at which to evaluate the solution.  Defaults to
        200 evenly spaced points in [0, T].
    max_step : float, optional
        Maximum solver step in DIMENSIONAL time units (internally scaled by
        mu), so accuracy is comparable to dlvt.model.simulate.

    Returns
    -------
    t : ndarray
        Dimensional time grid.
    V : ndarray
        Vitality trajectory mapped back from v (V = Vmax*v).
    C : ndarray
        Enacted-scope trajectory mapped back from w (C = O0*w/beta).
    """
    g = reduced_groups(p)
    v0, w0 = from_dimensional(V0, C0, p)
    if t_eval is None:
        t_eval = np.linspace(0.0, T, 200)
    tau_eval = p['mu'] * np.asarray(t_eval)
    sol = solve_ivp(
        reduced_rhs, [0.0, p['mu'] * T], [v0, w0], args=(g,),
        method='RK45', max_step=p['mu'] * max_step, t_eval=tau_eval,
        rtol=1e-10, atol=1e-12,
    )
    V, C = to_dimensional(sol.y[0], sol.y[1], p)
    return np.asarray(t_eval), V, C


# -- Equilibrium helpers -------------------------------------------------------

def _generous_c_max(p: Dict[str, float]) -> float:
    """Scan window for find_interior_equilibria that avoids the small-beta bug.

    The equilibrium scope scales as C* = (O* - O0)/beta ~ 1/beta, so a
    fixed window silently loses the equilibrium at small beta (the source of
    the historical beta_crit = 0.1015 artifact; see
    dlvt.analysis.estimate_bifurcation_interval).
    """
    return max(300.0, 20.0 / p['beta'])


def stable_equilibrium(p: Dict[str, float]) -> Optional[Dict[str, object]]:
    """Return the lowest-C stable interior equilibrium, or None.

    Uses :func:`dlvt.analysis.find_interior_equilibria` with a generous,
    beta-aware C_max (see :func:`_generous_c_max`).

    Parameters
    ----------
    p : Dict[str, float]
        Parameter dictionary.

    Returns
    -------
    Optional[Dict]
        The equilibrium dict (keys 'V', 'C', 'O', 'I', 'stable',
        'eigenvalues') for the lowest-C stable equilibrium, or
        None if no stable interior equilibrium exists.
    """
    eqs = find_interior_equilibria(p, C_max=_generous_c_max(p))
    stable = [eq for eq in eqs if eq['stable']]
    if not stable:
        return None
    stable.sort(key=lambda eq: eq['C'])
    return stable[0]


def _v_star(p: Dict[str, float]) -> float:
    """Equilibrium V* at the lowest-C stable equilibrium; raises if none."""
    eq = stable_equilibrium(p)
    if eq is None:
        raise ValueError("No stable interior equilibrium for these parameters.")
    return float(eq['V'])


# -- Local elasticities of V* --------------------------------------------------

def v_star_elasticities(p: Dict[str, float],
                        rel: float = 1e-4) -> Dict[str, float]:
    """Central-difference elasticities d ln V* / d ln p_i for all 11 parameters.

    Each parameter is perturbed multiplicatively to p_i*(1 +/- rel) and the
    elasticity is the log-log central difference

        E_i = [ln V*(p_i*(1+rel)) - ln V*(p_i*(1-rel))]
              / [ln(1+rel) - ln(1-rel)].

    Structural expectations at baseline (see module docstring):
    E_beta = E_eta = E_O0 = 0 exactly; E_R = -E_delta (R/delta ratio);
    E_mu = -E_alpha (mu/alpha ratio).

    Parameters
    ----------
    p : Dict[str, float]
        Parameter dictionary; a stable interior equilibrium must exist at p
        and at each perturbed point.
    rel : float, optional
        Relative perturbation size, default 1e-4.

    Returns
    -------
    Dict[str, float]
        Elasticity for each of the 11 parameters in PARAM_NAMES.
    """
    denom = np.log(1.0 + rel) - np.log(1.0 - rel)
    out: Dict[str, float] = {}
    for name in PARAM_NAMES:
        p_hi = dict(p)
        p_lo = dict(p)
        p_hi[name] = p[name] * (1.0 + rel)
        p_lo[name] = p[name] * (1.0 - rel)
        v_hi = _v_star(p_hi)
        v_lo = _v_star(p_lo)
        out[name] = float((np.log(v_hi) - np.log(v_lo)) / denom)
    return out


# -- Critical mu/alpha ratio ----------------------------------------------------

def mu_alpha_critical(p: Dict[str, float], lo: float = 0.5, hi: float = 10.0,
                      tol: float = 1e-6, max_iter: int = 100) -> float:
    """Bisection for the ratio r = mu/alpha at a display-threshold crossing.

    Holds alpha fixed at p['alpha'] and varies mu = r*alpha (by the exact
    mu/alpha degeneracy of V*, only the ratio matters).  V*(r) is increasing
    in r, so the stipulated display-label change is a single crossing. If no stable
    interior equilibrium exists at some trial r (which happens for large r,
    where V* would exceed Vmax), that trial is treated as being above the
    threshold.

    Parameters
    ----------
    p : Dict[str, float]
        Baseline parameter dictionary.
    lo, hi : float, optional
        Initial bracket for r = mu/alpha.  V*(lo) must be below the
        display threshold and V*(hi) above it (or infeasible).
    tol : float, optional
        Absolute tolerance on r, default 1e-6.
    max_iter : int, optional
        Maximum bisection iterations, default 100.

    Returns
    -------
    float
        The critical ratio (mu/alpha)_crit.  Baseline: ~2.163.

    Raises
    ------
    ValueError
        If the initial bracket does not straddle the threshold.
    """
    target = DISPLAY_THRESHOLD_FRACTION * p['Vmax']

    def excess(r: float) -> Optional[float]:
        """V*(r) - target, or None if no stable interior equilibrium."""
        p_trial = dict(p)
        p_trial['mu'] = r * p['alpha']
        try:
            return _v_star(p_trial) - target
        except ValueError:
            return None

    g_lo = excess(lo)
    if g_lo is None or g_lo >= 0.0:
        raise ValueError(
            f"mu_alpha_critical: V*(lo={lo}) is not below the threshold; "
            "widen the bracket downward."
        )
    g_hi = excess(hi)
    if g_hi is not None and g_hi <= 0.0:
        raise ValueError(
            f"mu_alpha_critical: V*(hi={hi}) is not above the threshold; "
            "widen the bracket upward."
        )

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        g_mid = excess(mid)
        if g_mid is not None and g_mid < 0.0:
            lo = mid
        else:
            # Above threshold, or infeasible (equilibrium lost at high r).
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


# -- Deprecated compatibility-label maps ---------------------------------------

def classify_point(p: Dict[str, float]) -> str:
    """Return deprecated compatibility labels for a parameter point.

    The historical string values are preserved only for callers of the
    legacy map APIs below. ``'none'`` means that no stable interior
    equilibrium was returned; the other labels encode a comparison with the
    illustrative ``0.5*Vmax`` display threshold and are not scientific states.

    Parameters
    ----------
    p : Dict[str, float]
        Parameter dictionary.

    Returns
    -------
    str
        One of 'zombie', 'sustainable', 'none'.
    """
    warnings.warn(
        "classify_point() returns deprecated compatibility labels; use "
        "classify_equilibrium() and retain continuous outputs.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _classify_point_compat(p)


def _classify_point_compat(p: Dict[str, float]) -> str:
    """Internal non-warning helper for deprecated grid wrappers."""
    result = classify_equilibrium(p)
    if result['status'] == 'no-stable-interior-equilibrium':
        return 'none'
    return 'zombie' if result['status'] == 'low-vitality' else 'sustainable'


def zombie_boundary_map(p: Dict[str, float],
                        r_range: Tuple[float, float] = (1.0, 4.0),
                        phi_range: Tuple[float, float] = (0.02, 0.40),
                        n: int = 41) -> Dict[str, object]:
    """Deprecated compatibility-label map over the (mu/alpha, phi) plane.

    For each grid point, mu is set to r*alpha (alpha held fixed) and phi is
    set to the grid value; the point is classified with
    :func:`classify_point`. The historical label boundary in this plane is
    the curve (mu/alpha)_crit(phi); at baseline phi = 0.15 it sits at
    r ~ 2.163, about 8% above the baseline ratio mu/alpha = 2.

    Parameters
    ----------
    p : Dict[str, float]
        Baseline parameter dictionary.
    r_range : Tuple[float, float], optional
        (min, max) for r = mu/alpha, default (1.0, 4.0).
    phi_range : Tuple[float, float], optional
        (min, max) for phi, default (0.02, 0.40).
    n : int, optional
        Grid points per axis, default 41.

    Returns
    -------
    Dict
        Keys:
        - 'r_values'   : ndarray (n,), the mu/alpha axis
        - 'phi_values' : ndarray (n,), the phi axis
        - 'regimes'    : object ndarray (n, n); regimes[i, j] is the class at
                         phi=phi_values[i], r=r_values[j]
        - 'baseline'   : (mu/alpha, phi) of the input p
        - 'axes'       : ('mu/alpha', 'phi')
    """
    warnings.warn(
        "zombie_boundary_map() is deprecated; build maps from continuous "
        "equilibrium outputs and an explicit display threshold.",
        DeprecationWarning,
        stacklevel=2,
    )
    r_values = np.linspace(r_range[0], r_range[1], n)
    phi_values = np.linspace(phi_range[0], phi_range[1], n)
    regimes = np.empty((n, n), dtype=object)
    for i, phi in enumerate(phi_values):
        for j, r in enumerate(r_values):
            p_trial = dict(p)
            p_trial['mu'] = r * p['alpha']
            p_trial['phi'] = phi
            regimes[i, j] = _classify_point_compat(p_trial)
    return {
        'r_values': r_values,
        'phi_values': phi_values,
        'regimes': regimes,
        'baseline': (p['mu'] / p['alpha'], p['phi']),
        'axes': ('mu/alpha', 'phi'),
    }


def zombie_boundary_map_beta(p: Dict[str, float],
                             r_range: Tuple[float, float] = (1.0, 4.0),
                             beta_range: Tuple[float, float] = (0.05, 1.0),
                             n: int = 41) -> Dict[str, object]:
    """Deprecated compatibility-label map over (mu/alpha, beta).

    Same shape as :func:`zombie_boundary_map`, but the second axis is beta.
    Because V* has exactly zero elasticity in beta (Lemma 2 / the reduced
    form), every column of the returned grid (fixed r, varying beta) must be
    constant: the historical label boundary is vertical and invariant in beta.
    The boolean 'boundary_invariant_in_beta' reports this check.

    Parameters
    ----------
    p : Dict[str, float]
        Baseline parameter dictionary.
    r_range : Tuple[float, float], optional
        (min, max) for r = mu/alpha, default (1.0, 4.0).
    beta_range : Tuple[float, float], optional
        (min, max) for beta, default (0.05, 1.0).
    n : int, optional
        Grid points per axis, default 41.

    Returns
    -------
    Dict
        Keys: 'r_values', 'beta_values', 'regimes' (object ndarray (n, n),
        rows indexed by beta), 'baseline', 'axes',
        'boundary_invariant_in_beta' (bool).
    """
    warnings.warn(
        "zombie_boundary_map_beta() is deprecated; build maps from continuous "
        "equilibrium outputs and an explicit display threshold.",
        DeprecationWarning,
        stacklevel=2,
    )
    r_values = np.linspace(r_range[0], r_range[1], n)
    beta_values = np.linspace(beta_range[0], beta_range[1], n)
    regimes = np.empty((n, n), dtype=object)
    for i, beta in enumerate(beta_values):
        for j, r in enumerate(r_values):
            p_trial = dict(p)
            p_trial['mu'] = r * p['alpha']
            p_trial['beta'] = beta
            regimes[i, j] = _classify_point_compat(p_trial)
    invariant = all(
        len({regimes[i, j] for i in range(n)}) == 1 for j in range(n)
    )
    return {
        'r_values': r_values,
        'beta_values': beta_values,
        'regimes': regimes,
        'baseline': (p['mu'] / p['alpha'], p['beta']),
        'axes': ('mu/alpha', 'beta'),
        'boundary_invariant_in_beta': invariant,
    }


# -- Latin-hypercube global screening ---------------------------------------------

def lhs_zombie_fraction(p: Dict[str, float], n_samples: int = 600,
                        factor: float = 2.0, seed: int = 1
                        ) -> Dict[str, object]:
    """Deprecated compatibility wrapper for threshold screening of V*.

    Draws a seeded Latin hypercube (scipy.stats.qmc.LatinHypercube) over all
    11 raw parameters, each log-uniform on [p_i/factor, p_i*factor], and for
    each draw records whether a stable interior equilibrium exists and, if
    so, whether it falls below the illustrative ``0.5*Vmax`` threshold.
    Spearman rank correlations
    between each log-parameter and V* (over the stable draws) are returned as
    a global-sensitivity proxy.

    HONEST LABELLING: this is a rank-correlation SCREENING, not a full
    variance-based (Sobol) decomposition.  Spearman coefficients capture
    monotone marginal effects; they do not decompose interactions.

    Parameters
    ----------
    p : Dict[str, float]
        Baseline parameter dictionary (centre of the hypercube).
    n_samples : int, optional
        Number of LHS draws, default 600.
    factor : float, optional
        Half-width of the log-uniform range (each parameter spans
        [p_i/factor, p_i*factor]), default 2.0.
    seed : int, optional
        RNG seed for the Latin hypercube, default 1 (deterministic).

    Returns
    -------
    Dict
        Keys:
        - 'n_samples', 'factor', 'seed'  : the inputs
        - 'n_stable'                     : draws with a stable interior eq.
        - 'n_zombie'                     : deprecated key; below-threshold draws
        - 'frac_stable'                  : n_stable / n_samples
        - 'zombie_fraction_given_stable' : deprecated compatibility key
        - 'zombie_fraction_overall'      : deprecated compatibility key
        - 'v_stars'                      : ndarray of V* over stable draws
        - 'spearman'                     : dict param -> Spearman rho between
                                           log(param) and V* (stable draws)
        - 'spearman_pvalues'             : dict param -> two-sided p-value
        - 'method'                       : honest-labelling note

    Notes
    -----
    At baseline with factor=2.0 the expected below-threshold share among stable draws
    is ~0.49 +/- 0.1.  Equilibria are searched with the generous beta-aware
    C_max window (see :func:`_generous_c_max`).
    """
    warnings.warn(
        "lhs_zombie_fraction() is deprecated; use continuous-output screening "
        "and an explicitly named low-vitality threshold summary.",
        DeprecationWarning,
        stacklevel=2,
    )
    sampler = qmc.LatinHypercube(d=len(PARAM_NAMES), seed=seed)
    U = sampler.random(n=n_samples)

    log_x = np.empty_like(U)
    stable_mask = np.zeros(n_samples, dtype=bool)
    zombie_mask = np.zeros(n_samples, dtype=bool)
    v_stars = np.full(n_samples, np.nan)

    for i in range(n_samples):
        p_trial = dict(p)
        for j, name in enumerate(PARAM_NAMES):
            # log-uniform on [p/factor, p*factor]
            p_trial[name] = p[name] * factor ** (2.0 * U[i, j] - 1.0)
            log_x[i, j] = np.log(p_trial[name])
        eq = stable_equilibrium(p_trial)
        if eq is not None:
            stable_mask[i] = True
            v_stars[i] = eq['V']
            zombie_mask[i] = bool(
                eq['V'] < DISPLAY_THRESHOLD_FRACTION * p_trial['Vmax']
            )

    n_stable = int(stable_mask.sum())
    n_zombie = int(zombie_mask.sum())
    v_stable = v_stars[stable_mask]

    spearman: Dict[str, float] = {}
    pvalues: Dict[str, float] = {}
    for j, name in enumerate(PARAM_NAMES):
        rho_s, pv = spearmanr(log_x[stable_mask, j], v_stable)
        spearman[name] = float(rho_s)
        pvalues[name] = float(pv)

    return {
        'n_samples': n_samples,
        'factor': factor,
        'seed': seed,
        'n_stable': n_stable,
        'n_zombie': n_zombie,
        'frac_stable': n_stable / n_samples,
        'zombie_fraction_given_stable':
            (n_zombie / n_stable) if n_stable else float('nan'),
        'zombie_fraction_overall': n_zombie / n_samples,
        'v_stars': v_stable,
        'spearman': spearman,
        'spearman_pvalues': pvalues,
        'method': (
            'Log-uniform Latin-hypercube screening with Spearman rank '
            'correlations between log-parameters and V* (stable draws). '
            'This is a rank-correlation screening / global-sensitivity '
            'proxy, NOT a full variance-based Sobol decomposition.'
        ),
    }
