"""
dlvt.recovery
=============
Experimental deterministic parameter-recovery simulation for the DLVT model.

The identifiability claims in :mod:`dlvt.nondimensional` are *structural*: the
interior equilibrium vitality ``V*`` depends on ``(mu, alpha)`` only through the
ratio ``mu/alpha`` and on ``(R, delta)`` only through ``R/delta`` (exact
degeneracies), while ``beta``, ``eta`` and ``O0`` have zero elasticity on
``V*``. Those are analytical statements. This module makes them *executable* as
a Monte-Carlo estimation experiment:

1. :func:`synthesize_panel` generates a noisy "panel" of leader trajectories
   (many leaders, sampled at discrete waves, with measurement noise), i.e. the
   kind of data a study would actually collect.

2. :func:`fit_reduced` fits a small number of free parameters back from the
   panel by least squares, integrating the *deterministic* DLVT model per
   leader from that leader's first observation as the initial condition. It is
   a pragmatic demonstration of estimation, not a production estimator.

3. :func:`ridge_profile` profiles the fitting loss along the exact degeneracy
   ridge ``(mu, alpha) -> (s*mu, s*alpha)`` (or ``(R, delta)``) and contrasts
   it with the steep profile obtained by scaling ``mu`` alone. The flat ridge
   is the operational signature of the ``mu/alpha`` degeneracy: the ratio is
   recoverable, the individual parameters much less so.

Everything is deterministic given seeds.

Public API
----------
synthesize_panel() : simulate a noisy multi-leader, multi-wave panel
fit_reduced()      : least-squares recovery of a few free parameters
ridge_profile()    : loss profile along the degeneracy ridge vs a single scale

References
----------
  Bendinelli, W. (2026). Dynamic Leadership Vitality Theory: A Formal Model
  manuscript in preparation.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from .model import dlvt_system, make_params
from .stochastic import simulate_sde


# -- Panel synthesis -----------------------------------------------------------

def synthesize_panel(p: Dict[str, float], n_leaders: int = 40,
                     n_waves: int = 12, wave_dt: float = 2.0,
                     obs_noise: float = 0.05, seed: int = 0,
                     sigma_V: float = 0.05,
                     missing_rate: float = 0.0,
                     dropout_rate: float = 0.0,
                     V0_range: Tuple[float, float] = (5.0, 9.0),
                     C0_range: Tuple[float, float] = (2.0, 10.0)
                     ) -> Dict[str, object]:
    """Simulate a noisy panel of leader trajectories.

    For each of ``n_leaders`` leaders a stochastic DLVT trajectory is generated
    with :func:`dlvt.stochastic.simulate_sde` from a random initial condition
    ``V0 ~ U[V0_range]``, ``C0 ~ U[C0_range]``. The trajectory is sampled at
    ``n_waves`` equally spaced observation times ``k*wave_dt`` (k = 0, ...,
    n_waves-1), and each sampled ``(V, C)`` is corrupted by independent
    *multiplicative lognormal* measurement noise with log-scale ``obs_noise``
    (so observations stay positive and the noise is scale-free). Optional
    missing waves and monotone dropout provide a simplified observation-process
    stress test. They are not a model of informative attrition.

    The choice of initial-condition ranges is *scientifically load-bearing*
    for identifiability experiments: the defaults start leaders far from the
    attractor, so the panel is transient-rich and the transient partially
    separates ``mu`` from ``alpha`` (their joint scaling changes the enacted-scope
    timescale). Passing ranges tightly around the equilibrium (e.g.
    ``V0_range=(V*-0.5, V*+0.5)``, ``C0_range=(C*-3, C*+3)``) produces an
    equilibrium-dominated panel in which only the ratio ``mu/alpha`` is
    identified — the flat ridge that :func:`ridge_profile` demonstrates.

    Parameters
    ----------
    p : Dict[str, float]
        True parameter dictionary used to generate the data.
    n_leaders : int, optional
        Number of leaders (panel units), default 40.
    n_waves : int, optional
        Number of observation waves per leader, default 12.
    wave_dt : float, optional
        Spacing between waves in model time units, default 2.0.
    obs_noise : float, optional
        Log-scale of the multiplicative lognormal measurement noise,
        default 0.05.
    seed : int, optional
        Master seed; leader ``i`` uses process seed ``seed + i`` and a
        dedicated measurement-noise stream. Default 0.
    sigma_V : float, optional
        Additive process-noise amplitude on V passed to
        :func:`dlvt.stochastic.simulate_sde`, default 0.05.
    missing_rate : float, optional
        Probability that a post-baseline leader-wave is missing completely at
        random, default 0. The same mask is applied to V and C.
    dropout_rate : float, optional
        Probability that a leader has a randomly located monotone post-baseline
        dropout, default 0. This is a stress-test mechanism, not an ignorable-
        attrition assumption.
    V0_range : Tuple[float, float], optional
        Uniform sampling range for initial vitality, default ``(5.0, 9.0)``
        (transient-rich panel; see Notes above).
    C0_range : Tuple[float, float], optional
        Uniform sampling range for initial enacted scope, default ``(2.0, 10.0)``.

    Returns
    -------
    Dict[str, object]
        Keys:
        - ``'t'`` : ndarray (n_waves,), the wave times
        - ``'V_obs'`` : ndarray (n_leaders, n_waves), noisy vitality
        - ``'C_obs'`` : ndarray (n_leaders, n_waves), noisy enacted scope
        - ``'V0'``, ``'C0'`` : ndarrays (n_leaders,), initial conditions
        - ``'true_params'`` : dict, a copy of ``p``
        - ``'n_leaders'``, ``'n_waves'``, ``'wave_dt'`` : the inputs
        - ``'observed_mask'`` : bool ndarray identifying complete waves
        - observation/process settings and seed

    Notes
    -----
    Deterministic given ``seed``. The wave times start at ``t = 0`` so the
    first observation is a noisy read of the (also noisy) initial state — the
    estimator in :func:`fit_reduced` uses it as the per-leader IC. The first
    wave is therefore always retained when missingness is requested.
    """
    if not 0.0 <= missing_rate < 1.0:
        raise ValueError("missing_rate must lie in [0, 1).")
    if not 0.0 <= dropout_rate < 1.0:
        raise ValueError("dropout_rate must lie in [0, 1).")
    if n_waves < 2:
        raise ValueError("n_waves must be at least 2.")
    rng = np.random.default_rng(seed)
    wave_times = wave_dt * np.arange(n_waves)
    T = float(wave_times[-1])
    dt = 0.01
    # Integer indices of the wave times on the SDE grid.
    wave_idx = np.round(wave_times / dt).astype(int)

    V0 = rng.uniform(V0_range[0], V0_range[1], size=n_leaders)
    C0 = rng.uniform(C0_range[0], C0_range[1], size=n_leaders)

    V_obs = np.empty((n_leaders, n_waves))
    C_obs = np.empty((n_leaders, n_waves))

    for i in range(n_leaders):
        _, V, C = simulate_sde(
            p, float(V0[i]), float(C0[i]), T, dt=dt,
            sigma_V=sigma_V, sigma_C=0.0, seed=seed + 1 + i,
        )
        Vi = V[wave_idx]
        Ci = C[wave_idx]
        # Multiplicative lognormal measurement noise (scale-free, positive).
        nV = rng.normal(0.0, obs_noise, size=n_waves)
        nC = rng.normal(0.0, obs_noise, size=n_waves)
        V_obs[i] = Vi * np.exp(nV)
        C_obs[i] = Ci * np.exp(nC)

    observed_mask = rng.random((n_leaders, n_waves)) >= missing_rate
    observed_mask[:, 0] = True
    for i in range(n_leaders):
        if rng.random() < dropout_rate:
            dropout_wave = int(rng.integers(1, n_waves))
            observed_mask[i, dropout_wave:] = False
    V_obs[~observed_mask] = np.nan
    C_obs[~observed_mask] = np.nan

    return {
        't': wave_times,
        'V_obs': V_obs,
        'C_obs': C_obs,
        'observed_mask': observed_mask,
        'V0': V0,
        'C0': C0,
        'true_params': dict(p),
        'n_leaders': n_leaders,
        'n_waves': n_waves,
        'wave_dt': wave_dt,
        'obs_noise': obs_noise,
        'sigma_V': sigma_V,
        'missing_rate': missing_rate,
        'dropout_rate': dropout_rate,
        'seed': seed,
    }


# -- Deterministic per-panel prediction ---------------------------------------

def _predict_panel(p: Dict[str, float], panel: Dict[str, object]
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministically predict (V, C) at every wave for every leader.

    Each leader is integrated with the deterministic DLVT model from its first
    observation ``(V_obs[i, 0], C_obs[i, 0])`` as the initial condition,
    evaluated at the wave times.
    """
    t = np.asarray(panel['t'], dtype=float)
    V_obs = np.asarray(panel['V_obs'])
    C_obs = np.asarray(panel['C_obs'])
    n_leaders = V_obs.shape[0]

    if not np.all(np.isfinite(V_obs[:, 0])) or not np.all(
        np.isfinite(C_obs[:, 0])
    ):
        raise ValueError("Each leader must have finite V and C at the first wave.")

    V_pred = np.empty_like(V_obs)
    C_pred = np.empty_like(C_obs)

    for i in range(n_leaders):
        V0 = float(V_obs[i, 0])
        C0 = float(C_obs[i, 0])
        sol = solve_ivp(
            dlvt_system, [t[0], t[-1]], [V0, C0], args=(p,),
            method='RK45', t_eval=t, rtol=1e-7, atol=1e-9,
        )
        if sol.y.shape[1] == len(t):
            V_pred[i] = np.maximum(sol.y[0], 0.0)
            C_pred[i] = np.maximum(sol.y[1], 0.0)
        else:  # integration hiccup: fall back to the IC (large residual)
            V_pred[i] = V0
            C_pred[i] = C0

    return V_pred, C_pred


def _panel_residuals(p: Dict[str, float], panel: Dict[str, object]
                     ) -> np.ndarray:
    """Flattened, per-channel-scaled residual vector for least squares.

    The C channel is rescaled by ``V_scale / C_scale`` so that vitality and
    enacted scope contribute on comparable footing despite their different
    magnitudes (C* ~ 32 vs V* ~ 4.7).
    """
    V_obs = np.asarray(panel['V_obs'])
    C_obs = np.asarray(panel['C_obs'])
    V_pred, C_pred = _predict_panel(p, panel)

    V_mask = np.isfinite(V_obs)
    C_mask = np.isfinite(C_obs)
    if not np.any(V_mask) or not np.any(C_mask):
        raise ValueError("The panel must contain finite V and C observations.")

    V_scale = max(float(np.nanmean(V_obs)), 1e-6)
    C_scale = max(float(np.nanmean(C_obs)), 1e-6)

    res_V = (V_pred - V_obs) / V_scale
    res_C = (C_pred - C_obs) / C_scale
    return np.concatenate([res_V[V_mask], res_C[C_mask]])


def panel_loss(p: Dict[str, float], panel: Dict[str, object]) -> float:
    """Scalar least-squares loss (0.5 * sum of squared scaled residuals)."""
    r = _panel_residuals(p, panel)
    return float(0.5 * np.dot(r, r))


# -- Least-squares recovery ----------------------------------------------------

def fit_reduced(panel: Dict[str, object], p_init: Dict[str, float],
                free: Sequence[str] = ('mu', 'alpha'),
                bounds_factor: float = 8.0,
                max_nfev: int = 200) -> Dict[str, object]:
    """Recover a few free parameters from the panel by least squares.

    Fits the parameters named in ``free`` (in log-space, so they stay
    positive) by minimising the scaled per-wave residuals between the
    deterministic DLVT prediction and the observed panel. All other
    parameters are held at their ``p_init`` values. Each leader is integrated
    from its own first observation as the initial condition.

    This is a *pragmatic demonstration* of parameter recovery, not a
    production estimator: it uses a plain Levenberg-Marquardt/TRF least-squares
    solve on a modest panel with no regularisation or standard errors.

    Parameters
    ----------
    panel : Dict[str, object]
        Output of :func:`synthesize_panel`.
    p_init : Dict[str, float]
        Initial (perturbed) parameter dictionary; the starting point of the
        optimisation and the source of all fixed parameter values.
    free : Sequence[str], optional
        Names of the parameters to fit, default ``('mu', 'alpha')``.
    bounds_factor : float, optional
        Each free parameter is bounded to ``[p_init/bounds_factor,
        p_init*bounds_factor]``. Default 8.0.
    max_nfev : int, optional
        Maximum number of residual evaluations, default 200.

    Returns
    -------
    Dict[str, object]
        Keys:
        - ``'params'`` : dict, the fitted full parameter dictionary
        - ``'free'`` : tuple, the fitted parameter names
        - ``'fitted'`` : dict, name -> fitted value (free params only)
        - ``'loss'`` : float, final scalar loss :func:`panel_loss`
        - ``'success'`` : bool, optimiser convergence flag
        - ``'n_eval'`` : int, residual evaluations used
        - ``'ratio'`` : float or None, ``free[0]/free[1]`` when exactly two
          free params are given (e.g. the recovered ``mu/alpha``)

    Notes
    -----
    Fitting is done on ``log(param)`` to enforce positivity and to make the
    ``mu/alpha`` degeneracy show up as a flat *direction* (equal log-shifts)
    rather than a curved one.
    """
    free = tuple(free)
    x0 = np.array([np.log(p_init[name]) for name in free])
    lo = np.array([np.log(p_init[name] / bounds_factor) for name in free])
    hi = np.array([np.log(p_init[name] * bounds_factor) for name in free])

    def residuals(x: np.ndarray) -> np.ndarray:
        p_try = dict(p_init)
        for name, val in zip(free, x):
            p_try[name] = float(np.exp(val))
        return _panel_residuals(p_try, panel)

    result = least_squares(
        residuals, x0, bounds=(lo, hi), method='trf',
        max_nfev=max_nfev, xtol=1e-10, ftol=1e-10,
    )

    p_fit = dict(p_init)
    fitted: Dict[str, float] = {}
    for name, val in zip(free, result.x):
        v = float(np.exp(val))
        p_fit[name] = v
        fitted[name] = v

    ratio: Optional[float] = None
    if len(free) == 2:
        ratio = fitted[free[0]] / fitted[free[1]]

    return {
        'params': p_fit,
        'free': free,
        'fitted': fitted,
        'loss': float(0.5 * np.dot(result.fun, result.fun)),
        'success': bool(result.success),
        'n_eval': int(result.nfev),
        'ratio': ratio,
    }


# -- Ridge / identifiability profile -------------------------------------------

def ridge_profile(panel: Dict[str, object], p_hat: Dict[str, float],
                  pair: Tuple[str, str] = ('mu', 'alpha'),
                  scales: Optional[np.ndarray] = None) -> Dict[str, object]:
    """Profile the loss along the degeneracy ridge vs a single-parameter scan.

    Demonstrates the ``mu/alpha`` (or ``R/delta``) identifiability degeneracy.
    Starting from the fitted ``p_hat``, two loss profiles are computed as a
    function of a multiplicative scale ``s``:

    * **joint ridge** — scale *both* members of ``pair`` by ``s``
      (``(mu, alpha) -> (s*mu, s*alpha)``), which leaves ``mu/alpha`` — and
      hence the interior equilibrium ``(V*, C*)`` — unchanged. The loss stays
      nearly flat (only the transient timescale moves), so this is the FLAT
      RIDGE.

    * **single scan** — scale only ``pair[0]`` by ``s`` (``mu -> s*mu``),
      which moves ``mu/alpha`` and therefore the equilibrium. The loss rises
      steeply.

    The summary ``ridge_flatness`` ratio quantifies the contrast:

        ridge_flatness = (max excess loss along the joint ridge)
                         / (excess loss when scaling pair[0] alone by 1.5)

    where "excess loss" is ``loss(s) - loss(1)``. A value ``<< 1`` confirms
    that the joint ridge is far flatter than a single-parameter move, i.e. the
    ratio is well identified while the individual parameters are not.

    Parameters
    ----------
    panel : Dict[str, object]
        Output of :func:`synthesize_panel`.
    p_hat : Dict[str, float]
        Reference parameter dictionary (typically the fitted params).
    pair : Tuple[str, str], optional
        The degeneracy pair to profile, default ``('mu', 'alpha')``. Use
        ``('R', 'delta')`` for the recovery/cost degeneracy.
    scales : ndarray, optional
        Multiplicative scale grid. Default ``np.geomspace(0.5, 2, 9)``.

    Returns
    -------
    Dict[str, object]
        Keys:
        - ``'scales'`` : ndarray, the scale grid
        - ``'loss_joint'`` : ndarray, loss along the joint ridge
        - ``'loss_single'`` : ndarray, loss scaling ``pair[0]`` alone
        - ``'excess_joint'`` : ndarray, ``loss_joint - loss(1)``
        - ``'excess_single'`` : ndarray, ``loss_single - loss(1)``
        - ``'loss_ref'`` : float, ``loss(1)`` at ``p_hat``
        - ``'ridge_flatness'`` : float, the summary ratio (<< 1 expected)
        - ``'pair'`` : the profiled pair

    Notes
    -----
    ``loss(1)`` is evaluated at ``s = 1`` and used as the common baseline for
    both excess-loss curves, so ``excess_joint[s=1] = excess_single[s=1] = 0``
    by construction.
    """
    if scales is None:
        scales = np.geomspace(0.5, 2.0, 9)
    scales = np.asarray(scales, dtype=float)
    a, b = pair

    loss_ref = panel_loss(dict(p_hat), panel)

    loss_joint = np.empty_like(scales)
    loss_single = np.empty_like(scales)
    for k, s in enumerate(scales):
        p_j = dict(p_hat)
        p_j[a] = p_hat[a] * s
        p_j[b] = p_hat[b] * s
        loss_joint[k] = panel_loss(p_j, panel)

        p_s = dict(p_hat)
        p_s[a] = p_hat[a] * s
        loss_single[k] = panel_loss(p_s, panel)

    excess_joint = loss_joint - loss_ref
    excess_single = loss_single - loss_ref

    # Excess loss from scaling pair[0] alone by 1.5 (the reference contrast).
    p_15 = dict(p_hat)
    p_15[a] = p_hat[a] * 1.5
    excess_single_15 = panel_loss(p_15, panel) - loss_ref

    max_excess_joint = float(np.max(np.abs(excess_joint)))
    denom = abs(excess_single_15)
    ridge_flatness = float(max_excess_joint / denom) if denom > 0 else float('inf')

    return {
        'scales': scales,
        'loss_joint': loss_joint,
        'loss_single': loss_single,
        'excess_joint': excess_joint,
        'excess_single': excess_single,
        'loss_ref': loss_ref,
        'ridge_flatness': ridge_flatness,
        'pair': pair,
    }
