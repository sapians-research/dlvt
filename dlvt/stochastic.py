"""
dlvt.stochastic
===============
Stochastic robustness of the DLVT interior attractor (virtual experiment A).

The deterministic DLVT model (see :mod:`dlvt.model`) has a unique, globally
asymptotically stable interior equilibrium, relative to positive scope and
under the R8 existence condition ``F(O0)>0``. A natural
methodological question is whether that attractor *survives noise*: real
executives do not follow a deterministic vector field, so the claim "the
system settles at (V*, C*)" is only useful if the equilibrium is robust to
stochastic perturbation.

This module answers that question numerically with an Euler-Maruyama
integrator for the DLVT drift plus two independent noise channels:

    dV = [ R*(1 - V/Vmax) - delta*O^gamma * V/(V + eps) ] dt + sigma_V dW1
    dC = [ alpha*I - mu*C ]                                dt + sigma_C * C dW2

The V-channel uses *additive* noise (energy shocks that do not scale with the
current level) and the C-channel uses *multiplicative* noise (scope shocks
proportional to enacted scope, keeping C >= 0 in the small-noise limit).
State is projected to V in [0,Vmax] and C in [0,infinity) after every step.
This is a projected Euler–Maruyama numerical experiment, not a fully specified
reflected SDE with a local-time term. Its finite-step persistence results are
computational diagnostics, not a stochastic stability theorem.

Everything is deterministic given a seed: a fixed ``seed`` reproduces the exact
same Wiener increments (via :class:`numpy.random.default_rng`) and therefore
identical trajectories, which is what the accompanying tests pin.

Public API
----------
simulate_sde()          : single Euler-Maruyama path (t, V, C)
attractor_persistence() : ensemble persistence / escape metrics near (V*, C*)
escape_curve()          : persistence and escape metrics vs a list of sigma_V

References
----------
  Bendinelli, W. (2026). Dynamic Leadership Vitality Theory: A Formal Model
  manuscript in preparation.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from .model import coordination_load, impact
from .analysis import find_interior_equilibria


# -- Single-path Euler-Maruyama integrator ------------------------------------

def simulate_sde(p: Dict[str, float], V0: float, C0: float, T: float,
                 dt: float = 0.01, sigma_V: float = 0.0, sigma_C: float = 0.0,
                 seed: int = 0
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate the stochastic DLVT system by the Euler-Maruyama scheme.

    Integrates

        dV = drift_V dt + sigma_V dW1
        dC = drift_C dt + sigma_C * C dW2

    where ``drift_V`` and ``drift_C`` are the deterministic DLVT right-hand
    sides (Equations 3.3-3.4). ``dW1`` and ``dW2`` are independent Wiener
    increments with variance ``dt``. After each step V is clamped into
    ``[0, Vmax]`` and C into ``[0, inf)`` to preserve the physical domain.

    With ``sigma_V = sigma_C = 0`` this reduces to a deterministic Euler
    integrator of the DLVT drift, which converges to the interior attractor
    from any interior initial condition.

    Parameters
    ----------
    p : Dict[str, float]
        Parameter dictionary (use :func:`dlvt.model.make_params`).
    V0 : float
        Initial vitality, clamped into ``[0, Vmax]``.
    C0 : float
        Initial enacted-scope stock, clamped into ``[0, inf)``.
    T : float
        Integration horizon; the path spans ``[0, T]``.
    dt : float, optional
        Time step, default 0.01. Smaller steps reduce Euler-Maruyama bias.
    sigma_V : float, optional
        Additive noise amplitude on the V-channel, default 0.0.
    sigma_C : float, optional
        Multiplicative noise amplitude on the C-channel (scales with C),
        default 0.0.
    seed : int, optional
        Seed for :class:`numpy.random.default_rng`, default 0. A fixed seed
        yields bit-identical output arrays (used by the determinism test).

    Returns
    -------
    t : ndarray, shape (n_steps + 1,)
        Time grid ``0, dt, 2 dt, ..., n_steps*dt``.
    V : ndarray, shape (n_steps + 1,)
        Vitality path V(t), clamped to ``[0, Vmax]``.
    C : ndarray, shape (n_steps + 1,)
        Enacted-scope path C(t), projected to ``[0, inf)``.

    Notes
    -----
    The number of steps is ``n_steps = int(round(T / dt))``. Noise increments
    are drawn as ``sqrt(dt) * N(0, 1)``. The drift is evaluated with the same
    ``coordination_load``/``impact`` helpers as the deterministic model so the two
    integrators share a single source of truth. Projection is an algorithmic
    boundary rule and must not be interpreted as a reflected-SDE solution.

    Examples
    --------
    >>> from dlvt.model import make_params
    >>> t, V, C = simulate_sde(make_params(), 8.0, 5.0, T=100, sigma_V=0.1)
    >>> V[-1] > 0
    True
    """
    rng = np.random.default_rng(seed)
    n_steps = int(round(T / dt))
    Vmax = p['Vmax']

    t = np.linspace(0.0, n_steps * dt, n_steps + 1)
    V = np.empty(n_steps + 1)
    C = np.empty(n_steps + 1)
    V[0] = min(max(V0, 0.0), Vmax)
    C[0] = max(C0, 0.0)

    sqrt_dt = np.sqrt(dt)
    # Pre-draw all increments for speed and reproducibility.
    dW1 = rng.standard_normal(n_steps) * sqrt_dt
    dW2 = rng.standard_normal(n_steps) * sqrt_dt

    R = p['R']
    delta = p['delta']
    gamma = p['gamma']
    eps = p['eps']
    alpha = p['alpha']
    mu = p['mu']

    v = V[0]
    c = C[0]
    for i in range(n_steps):
        O = coordination_load(c, p)
        recovery = R * (1.0 - v / Vmax)
        drain = delta * O**gamma * v / (v + eps)
        I = impact(v, c, O, p)
        drift_V = recovery - drain
        drift_C = alpha * I - mu * c

        v = v + drift_V * dt + sigma_V * dW1[i]
        c = c + drift_C * dt + sigma_C * c * dW2[i]

        # Clamp to the physical domain.
        if v < 0.0:
            v = 0.0
        elif v > Vmax:
            v = Vmax
        if c < 0.0:
            c = 0.0

        V[i + 1] = v
        C[i + 1] = c

    return t, V, C


# -- Ensemble persistence / escape metrics ------------------------------------

def _interior_target(p: Dict[str, float]) -> Tuple[float, float]:
    """Return (V*, C*) of the lowest-C stable interior equilibrium.

    Raises ValueError if no interior equilibrium exists for ``p``.
    """
    eqs = find_interior_equilibria(p)
    if not eqs:
        raise ValueError(
            "No interior equilibrium for these parameters; the attractor "
            "persistence experiment is undefined."
        )
    stable = [e for e in eqs if e['stable']]
    pool = stable if stable else eqs
    pool = sorted(pool, key=lambda e: e['C'])
    eq = pool[0]
    return float(eq['V']), float(eq['C'])


def attractor_persistence(p: Dict[str, float], sigma_V: float,
                          sigma_C: float = 0.0, n_paths: int = 50,
                          T: float = 300.0, burn_in: float = 100.0,
                          band: float = 1.0, dt: float = 0.01,
                          seed: int = 1) -> Dict[str, object]:
    """Ensemble persistence metrics of the interior attractor under noise.

    Simulates ``n_paths`` independent Euler-Maruyama trajectories started at
    the deterministic target ``(V*, C*)`` and measures how tightly they stay
    near it after an initial ``burn_in``. The target is obtained from
    :func:`dlvt.analysis.find_interior_equilibria` (lowest-C stable
    equilibrium).

    Three families of metrics are returned:

    * **ensemble statistics** — mean and std of ``(V, C)`` over all paths and
      all post-burn-in samples;
    * **in-band fraction** — fraction of post-burn-in samples satisfying
      ``|V - V*| < band`` AND ``|C - C*| < band*(C*/V*)`` (the C tolerance is
      scaled by ``C*/V*`` so the band is comparably tight in each coordinate);
    * **escape fraction** — fraction of *paths* that hit ``V < 0.5`` or
      ``C < 1`` at any post-burn-in time (a proxy for the attractor being
      displaced toward the boundary).

    Parameters
    ----------
    p : Dict[str, float]
        Parameter dictionary.
    sigma_V : float
        Additive V-noise amplitude.
    sigma_C : float, optional
        Multiplicative C-noise amplitude, default 0.0.
    n_paths : int, optional
        Number of ensemble paths, default 50.
    T : float, optional
        Integration horizon per path, default 300.0.
    burn_in : float, optional
        Time discarded before collecting statistics, default 100.0.
    band : float, optional
        Half-width of the V acceptance band; the C half-width is
        ``band*(C*/V*)``. Default 1.0.
    dt : float, optional
        Euler-Maruyama step, default 0.01.
    seed : int, optional
        Base seed; path ``k`` uses ``seed + k``. Default 1.

    Returns
    -------
    Dict[str, object]
        Keys:
        - ``'target'`` : (V*, C*)
        - ``'sigma_V'``, ``'sigma_C'`` : the inputs
        - ``'n_paths'`` : int
        - ``'mean_V'``, ``'mean_C'`` : post-burn-in ensemble means
        - ``'std_V'``, ``'std_C'`` : post-burn-in ensemble stds
        - ``'in_band_fraction'`` : fraction of post-burn-in samples in-band
        - ``'escape_fraction'`` : fraction of paths that escaped
        - ``'band'`` : (band_V, band_C) half-widths actually used

    Notes
    -----
    Deterministic given ``seed``: the per-path seeds are ``seed, seed+1, ...``.
    """
    V_star, C_star = _interior_target(p)
    band_V = band
    band_C = band * (C_star / V_star)

    burn_idx = int(round(burn_in / dt))

    all_V: List[np.ndarray] = []
    all_C: List[np.ndarray] = []
    n_escaped = 0

    for k in range(n_paths):
        _, V, C = simulate_sde(
            p, V_star, C_star, T, dt=dt,
            sigma_V=sigma_V, sigma_C=sigma_C, seed=seed + k,
        )
        Vp = V[burn_idx:]
        Cp = C[burn_idx:]
        all_V.append(Vp)
        all_C.append(Cp)
        if np.any(Vp < 0.5) or np.any(Cp < 1.0):
            n_escaped += 1

    V_stack = np.concatenate(all_V)
    C_stack = np.concatenate(all_C)

    in_band = (np.abs(V_stack - V_star) < band_V) & \
              (np.abs(C_stack - C_star) < band_C)

    return {
        'target': (V_star, C_star),
        'sigma_V': sigma_V,
        'sigma_C': sigma_C,
        'n_paths': n_paths,
        'mean_V': float(np.mean(V_stack)),
        'mean_C': float(np.mean(C_stack)),
        'std_V': float(np.std(V_stack)),
        'std_C': float(np.std(C_stack)),
        'in_band_fraction': float(np.mean(in_band)),
        'escape_fraction': float(n_escaped / n_paths),
        'band': (band_V, band_C),
    }


def escape_curve(p: Dict[str, float], sigmas: List[float],
                 **kw) -> Dict[str, object]:
    """Persistence / escape metrics of the attractor vs a list of sigma_V.

    Repeatedly calls :func:`attractor_persistence` for each value of the
    additive V-noise amplitude in ``sigmas`` and stacks the resulting metrics
    into arrays suitable for plotting the escape curve (virtual experiment A).

    Parameters
    ----------
    p : Dict[str, float]
        Parameter dictionary.
    sigmas : List[float]
        Values of ``sigma_V`` to sweep.
    **kw
        Extra keyword arguments forwarded to :func:`attractor_persistence`
        (e.g. ``n_paths``, ``T``, ``burn_in``, ``band``, ``sigma_C``,
        ``seed``, ``dt``).

    Returns
    -------
    Dict[str, object]
        Keys:
        - ``'sigmas'`` : ndarray of the swept sigma_V values
        - ``'in_band_fraction'`` : ndarray, in-band fraction per sigma
        - ``'escape_fraction'`` : ndarray, escape fraction per sigma
        - ``'std_V'``, ``'std_C'`` : ndarrays of ensemble spread per sigma
        - ``'mean_V'``, ``'mean_C'`` : ndarrays of ensemble means per sigma
        - ``'target'`` : (V*, C*)
        - ``'records'`` : list of the full per-sigma metric dicts

    Notes
    -----
    Deterministic given the seed passed through ``kw`` (default 1). All
    sigmas reuse the same base seed, so differences across the curve reflect
    the noise amplitude, not reseeding.
    """
    sig = np.asarray(sigmas, dtype=float)
    records = [attractor_persistence(p, sigma_V=float(s), **kw) for s in sig]

    return {
        'sigmas': sig,
        'in_band_fraction': np.array([r['in_band_fraction'] for r in records]),
        'escape_fraction': np.array([r['escape_fraction'] for r in records]),
        'std_V': np.array([r['std_V'] for r in records]),
        'std_C': np.array([r['std_C'] for r in records]),
        'mean_V': np.array([r['mean_V'] for r in records]),
        'mean_C': np.array([r['mean_C'] for r in records]),
        'target': records[0]['target'] if records else None,
        'records': records,
    }
