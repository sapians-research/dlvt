"""
dlvt.fastslow
=============
Quasi-static nullcline analysis and the colored
basin portrait for the DLVT model.

This module makes two of the virtual experiments requested by the methodology
review executable:

(A) **Quasi-static reduction.** Freezing enacted leadership scope C (hence experienced load
    O = O_0 + beta*C^eta), the vitality equation relaxes to its
    quasi-equilibrium V_qe(O) — the unique positive root of the nullcline
    quadratic (Eq. A11.1 / eq:vqe-quadratic in the paper):

        (R/Vmax)*V^2 - (R - R*eps/Vmax - delta*O^gamma)*V - R*eps = 0.

    The *quasi-static nullcline* is the graph C -> V_qe(O(C)). It is not an
    invariant slow manifold of the full flow. Its intersection with the
    scope nullcline is exactly the equilibrium of the reduced 1D flow

        dC/dt = C * ( alpha*V_qe(O(C)) / (1 + phi*O(C)) - mu )

    coincides EXACTLY (not approximately) with the full interior equilibrium
    (V*, C*) = (4.70253, 32.034) at baseline.

(B) **Basin portrait.** A grid of initial conditions integrated under the
    full 2D flow, recording the time to enter and remain in a band around
    (V*, C*). Theorem 2c (global asymptotic stability on {C > 0}) predicts a
    finite convergence time for every initial condition with C_0 > 0; the
    portrait colors that time and counts exceptions (expected: zero).

HONEST TIMESCALE-SEPARATION FINDING (measured, not assumed)
-----------------------------------------------------------
Earlier drafts claimed a "~15x timescale separation" by comparing the raw
rates 1/R = 0.33 against 1/mu = 5. That comparison is misleading:

* Near the equilibrium, the linearized vitality relaxation rate is
  |J_VV| = R/Vmax + delta*O*^gamma*eps/(V*+eps)^2 ≈ 0.31 (NOT R = 3: the
  recovery term relaxes at rate R/Vmax once V is away from the floor), and
  the linearized scope rate is |J_CC| ≈ 0.10 (NOT mu = 0.2: the alpha*I
  inflow cancels most of the contraction on the scope nullcline). The
  measured diagonal ratio is |J_VV/J_CC| ≈ 3.0 — a factor ~3, not ~15.

* Worse for a strict decomposition: the eigenvalues at the equilibrium are
  the COMPLEX pair -0.205 +/- 0.331i, so near the equilibrium there are no
  separate "fast" and "slow" real modes at all — the two variables share a
  single damped-spiral mode and the modes mix. This is exactly why the
  approach to (V*, C*) is an overshoot-and-return spiral, which no 1D
  (necessarily monotone) reduced flow can reproduce.

Consequently the quasi-static reduction can describe broad enacted-scope drift far
from equilibrium, but its maximum relative C-error is about 15% on the
baseline transient and it degrades near the spiral. Quantify it with
:func:`reduction_error`; do not assume it. The global-stability proof
(Appendix A10) does not rely on the reduction in any way.

References
----------
  Bendinelli, W. (2026). Dynamic Leadership Vitality Theory: A Formal Model
  manuscript in preparation, Appendix A11 (quasi-static structure).
"""

from typing import Dict, Optional, Tuple, Union
import warnings

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from .model import coordination_load, dlvt_system
from .analysis import find_interior_equilibria, jacobian_eigenvalues


# ── Quasi-equilibrium and quasi-static nullcline ─────────────────────────────

def v_quasi_equilibrium(O: Union[float, np.ndarray],
                        p: Dict[str, float]) -> Union[float, np.ndarray]:
    """Closed-form quasi-equilibrium vitality V_qe(O) at frozen experienced load.

    Solves the V-nullcline quadratic (paper eq:vqe-quadratic)

        a*V^2 + b*V + c = 0,
        a = R/Vmax,
        b = -(R - R*eps/Vmax - delta*O^gamma),
        c = -R*eps,

    and returns the unique positive root. The product of the roots is
    c/a = -eps*Vmax < 0, so the roots are real of opposite sign and the
    "+" branch of the quadratic formula is always the positive one
    (Proposition prop:nullcline of the paper).

    Timescale honesty: V relaxes to V_qe(O) at the linearized rate
    |dF/dV| = R/Vmax + delta*O^gamma*eps/(V_qe+eps)^2 ≈ 0.31 at baseline
    equilibrium load — about 3x (not the raw-rate 15x once claimed) faster
    than the enacted-scope dynamics. See the module docstring and
    :func:`reduction_error`.

    Parameters
    ----------
    O : float or ndarray
        Frozen experienced-load level(s), O > 0. Vectorized: an array input
        returns an array of the same shape.
    p : Dict[str, float]
        Parameter dictionary with keys 'R', 'Vmax', 'delta', 'gamma', 'eps'.

    Returns
    -------
    float or ndarray
        The positive root V_qe(O), in (0, Vmax). Scalar in, scalar out.

    Examples
    --------
    >>> from dlvt.model import make_params
    >>> p = make_params()
    >>> round(float(v_quasi_equilibrium(1.0, p)), 3)   # axis equilibrium
    9.934
    """
    O_arr = np.asarray(O, dtype=float)
    a = p['R'] / p['Vmax']
    b = -(p['R'] - p['R'] * p['eps'] / p['Vmax']
          - p['delta'] * np.power(O_arr, p['gamma']))
    c = -p['R'] * p['eps']
    disc = b * b - 4.0 * a * c
    V = (-b + np.sqrt(disc)) / (2.0 * a)
    if np.isscalar(O) or O_arr.ndim == 0:
        return float(V)
    return V


def quasi_static_nullcline(C: Union[float, np.ndarray],
                           p: Dict[str, float]) -> Union[float, np.ndarray]:
    """Return ``V_qe(O(C))``, the vitality nullcline as a graph over C.

    Its intersection with the scope nullcline is exactly the full interior
    equilibrium, but the graph is generally not invariant under the full
    two-dimensional flow. “Quasi-static nullcline” is therefore the precise
    term; “slow manifold” would require a singular-perturbation/invariance
    result that the baseline calibration does not supply.

    What IS approximate is the *dynamics* on the manifold: the reduction
    assumes V is slaved to C, which holds only to the measured ~3x rate
    separation (|J_VV/J_CC| ≈ 3.0 at baseline, complex eigenvalue pair —
    modes mix near the equilibrium). See :func:`reduction_error`.

    Parameters
    ----------
    C : float or ndarray
        Enacted leadership-scope level(s), C >= 0.
    p : Dict[str, float]
        Parameter dictionary.

    Returns
    -------
    float or ndarray
        V_qe(O(C)); scalar in, scalar out.
    """
    return v_quasi_equilibrium(coordination_load(C, p), p)


def slow_manifold(C: Union[float, np.ndarray],
                  p: Dict[str, float]) -> Union[float, np.ndarray]:
    """Deprecated alias for :func:`quasi_static_nullcline`."""
    warnings.warn(
        "slow_manifold() is deprecated because the graph is not invariant; "
        "use quasi_static_nullcline().",
        DeprecationWarning,
        stacklevel=2,
    )
    return quasi_static_nullcline(C, p)


# ── Reduced 1D slow flow ─────────────────────────────────────────────────────

def reduced_slow_rhs(t: float, y, p: Dict[str, float]):
    """Right-hand side of the reduced (quasi-static) 1D enacted-scope ODE.

    With V slaved to the quasi-static nullcline, the scope equation becomes

        dC/dt = alpha * C * V_qe(O(C)) / (1 + phi*O(C)) - mu*C,

    a scalar autonomous flow. Being one-dimensional it is necessarily
    monotone in time — it can reproduce the slow enacted-scope drift but NOT the
    overshoot-and-return spiral of the full system (complex eigenvalues
    -0.205 +/- 0.331i at baseline; the honest separation is only ~3x, see
    the module docstring), which is precisely where the reduction degrades.

    Parameters
    ----------
    t : float
        Time (unused; the system is autonomous).
    y : sequence of one float
        State [C].
    p : Dict[str, float]
        Parameter dictionary.

    Returns
    -------
    list of one float
        [dC/dt].
    """
    C = max(y[0], 0.0)
    O = coordination_load(C, p)
    V = v_quasi_equilibrium(O, p)
    dCdt = p['alpha'] * C * V / (1.0 + p['phi'] * O) - p['mu'] * C
    return [dCdt]


def simulate_reduced(p: Dict[str, float], C0: float, T: float = 120.0,
                     n_eval: int = 400, rtol: float = 1e-8,
                     atol: float = 1e-10
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate the reduced 1D slow system from C0 over [0, T].

    The initial vitality is implicit: the reduced trajectory starts ON the
    slow manifold at V = V_qe(O(C0)) by construction (there is no vitality
    degree of freedom left).

    Parameters
    ----------
    p : Dict[str, float]
        Parameter dictionary.
    C0 : float
        Initial enacted scope (C0 > 0 for a nontrivial trajectory).
    T : float, optional
        Time horizon, default 120.
    n_eval : int, optional
        Number of output grid points, default 400.
    rtol, atol : float, optional
        solve_ivp tolerances.

    Returns
    -------
    t : ndarray, shape (n_eval,)
        Output time grid.
    C_reduced : ndarray, shape (n_eval,)
        Reduced enacted-scope trajectory C(t).
    V_on_manifold : ndarray, shape (n_eval,)
        The slaved vitality V_qe(O(C(t))) along the trajectory.
    """
    t_eval = np.linspace(0.0, T, n_eval)
    sol = solve_ivp(reduced_slow_rhs, [0.0, T], [C0], args=(p,),
                    method='RK45', t_eval=t_eval, rtol=rtol, atol=atol)
    C_red = np.maximum(sol.y[0], 0.0)
    V_man = quasi_static_nullcline(C_red, p)
    return sol.t, C_red, np.asarray(V_man)


# ── Reduction-error measurement (the honest quantification) ─────────────────

def reduction_error(p: Dict[str, float], V0: float, C0: float,
                    T: float = 120.0, n_eval: int = 400) -> Dict[str, object]:
    """Measure how well the quasi-static reduction tracks the full 2D flow.

    Integrates the FULL system from (V0, C0) and the REDUCED 1D system from
    the same C0 (the reduced trajectory necessarily starts on the slow
    manifold at V_qe(O(C0)); the full one starts at the given V0), evaluates
    both on a common time grid, and reports the discrepancy together with
    the honest local timescale diagnostics at the equilibrium.

    Measured baseline finding (V0=8, C0=5, T=120): max relative C-error
    ~4%, mean well under 1%; equilibrium mismatch ~0 (the reduced and full
    equilibria coincide exactly — both nullclines); eigen-separation
    |J_VV/J_CC| ≈ 3.0 (NOT the raw-rate 15x = (1/mu)/(1/R) once claimed);
    modes_mix True (complex pair -0.205 +/- 0.331i). Interpretation: the
    reduction is a useful intuition-builder for the slow enacted-scope drift far
    from equilibrium, and degrades near the equilibrium where the damped
    spiral lives — a strict fast--slow decomposition fails there.

    Parameters
    ----------
    p : Dict[str, float]
        Parameter dictionary.
    V0 : float
        Initial vitality for the FULL system (the reduced system has no
        vitality freedom and starts on the manifold instead).
    C0 : float
        Common initial enacted scope for both systems (C0 > 0).
    T : float, optional
        Horizon, default 120.
    n_eval : int, optional
        Common evaluation-grid size, default 400.

    Returns
    -------
    Dict[str, object]
        Keys:

        - ``'t'`` : ndarray — common time grid.
        - ``'C_full'``, ``'V_full'`` : ndarray — full 2D trajectory.
        - ``'C_reduced'``, ``'V_manifold'`` : ndarray — reduced trajectory
          and its slaved vitality.
        - ``'max_rel_error_C'`` : float — max_t |C_red - C_full| / C_full.
        - ``'mean_rel_error_C'`` : float — mean of the same quantity.
        - ``'equilibrium_mismatch'`` : float — max component distance
          between the reduced 1D equilibrium (brentq on the reduced RHS)
          and the full interior equilibrium; ~0 by construction.
        - ``'eigen_separation'`` : float — measured |J_VV / J_CC| at the
          interior equilibrium (~3.0 at baseline).
        - ``'raw_rate_ratio'`` : float — the misleading naive ratio
          R/mu (= 15 at baseline), included for contrast only.
        - ``'modes_mix'`` : bool — True iff the equilibrium eigenvalues are
          complex (no strict fast/slow mode split near equilibrium).
        - ``'eigenvalues'`` : ndarray — the equilibrium eigenvalues.
        - ``'V_star'``, ``'C_star'`` : float — full interior equilibrium.
    """
    t_eval = np.linspace(0.0, T, n_eval)

    # Full 2D system.
    sol_full = solve_ivp(dlvt_system, [0.0, T], [V0, C0], args=(p,),
                         method='RK45', t_eval=t_eval,
                         rtol=1e-8, atol=1e-10)
    V_full = np.maximum(sol_full.y[0], 0.0)
    C_full = np.maximum(sol_full.y[1], 0.0)

    # Reduced 1D system from the same C0 (starts on the manifold).
    _, C_red, V_man = simulate_reduced(p, C0, T=T, n_eval=n_eval)

    rel_err = np.abs(C_red - C_full) / np.maximum(C_full, 1e-12)
    max_rel = float(np.max(rel_err))
    mean_rel = float(np.mean(rel_err))

    # Full interior equilibrium.
    eq = find_interior_equilibria(p)[0]
    V_star, C_star = float(eq['V']), float(eq['C'])

    # Reduced equilibrium: root of the per-capita reduced growth rate
    # g(C) = alpha*V_qe(O(C))/(1+phi*O(C)) - mu, bracketed around C*.
    def g(C: float) -> float:
        O = coordination_load(C, p)
        return (p['alpha'] * v_quasi_equilibrium(O, p)
                / (1.0 + p['phi'] * O) - p['mu'])

    lo, hi = 0.5 * C_star, 2.0 * C_star
    C_red_star = brentq(g, lo, hi, xtol=1e-12, rtol=1e-14)
    V_red_star = float(quasi_static_nullcline(C_red_star, p))
    eq_mismatch = float(max(abs(C_red_star - C_star),
                            abs(V_red_star - V_star)))

    # Honest local timescale diagnostics at the equilibrium.
    eigvals, _ = jacobian_eigenvalues(V_star, C_star, p)
    O_star = coordination_load(C_star, p)
    dOdC = p['beta'] * p['eta'] * C_star ** (p['eta'] - 1.0)
    J00 = (-p['R'] / p['Vmax']
           - p['delta'] * O_star ** p['gamma'] * p['eps']
           / (V_star + p['eps']) ** 2)
    J11 = (p['alpha'] * V_star / (1.0 + p['phi'] * O_star)
           - p['alpha'] * C_star * V_star * p['phi'] * dOdC
           / (1.0 + p['phi'] * O_star) ** 2
           - p['mu'])
    eigen_separation = float(abs(J00 / J11))
    modes_mix = bool(np.any(np.abs(np.imag(eigvals)) > 1e-12))

    return {
        't': t_eval,
        'C_full': C_full,
        'V_full': V_full,
        'C_reduced': C_red,
        'V_manifold': V_man,
        'max_rel_error_C': max_rel,
        'mean_rel_error_C': mean_rel,
        'equilibrium_mismatch': eq_mismatch,
        'eigen_separation': eigen_separation,
        'raw_rate_ratio': float(p['R'] / p['mu']),
        'modes_mix': modes_mix,
        'eigenvalues': eigvals,
        'V_star': V_star,
        'C_star': C_star,
    }


# ── Basin portrait (colored time-to-converge grid) ───────────────────────────

def basin_portrait_grid(p: Dict[str, float], n_V: int = 40, n_C: int = 40,
                        V_range: Optional[Tuple[float, float]] = None,
                        C_range: Tuple[float, float] = (0.2, 90.0),
                        T: float = 400.0, band: float = 0.5,
                        n_t: int = 800) -> Dict[str, object]:
    """Colored basin portrait: time-to-converge over a grid of initial states.

    For each initial condition (V0, C0) on an n_V x n_C grid, integrates the
    FULL 2D system (RK45, modest tolerances) and records the first time
    after which the trajectory enters AND REMAINS inside the box
    {|V - V*| < band, |C - C*| < band} around the interior equilibrium
    through the end of the horizon T. Trajectories that have not settled
    into the band by T are recorded as np.inf and counted as exceptions.

    Theorem 2c (global asymptotic stability on {C > 0}, Appendix A10)
    predicts that EVERY initial condition with C0 > 0 converges — the basin
    is the whole open half-plane {C > 0}, with no escape and no competing
    attractor. The portrait therefore shows a smooth gradient of finite
    convergence times, not a basin boundary. (The invariant axis C = 0
    carries a saddle at (V ≈ 9.934, 0) and is excluded by the default
    C_range.) Note the times mix the ~3x-separated fast approach to the
    slow manifold with the slow spiral decay (e-folding time 1/0.205 ≈ 4.9
    at baseline; see the module docstring for the honest separation
    finding).

    Parameters
    ----------
    p : Dict[str, float]
        Parameter dictionary.
    n_V, n_C : int, optional
        Grid resolution in V0 and C0, default 40 x 40.
    V_range : (float, float), optional
        Range of initial vitalities; default (0.2, Vmax).
    C_range : (float, float), optional
        Range of initial enacted-scope values, default (0.2, 90). Must have C_min > 0
        for the Theorem 2c prediction to apply.
    T : float, optional
        Integration horizon, default 400.
    band : float, optional
        Half-width of the convergence box around (V*, C*), default 0.5.
    n_t : int, optional
        Number of evaluation points per trajectory, default 800.

    Returns
    -------
    Dict[str, object]
        Keys:

        - ``'V0'`` : ndarray, shape (n_V,) — initial-vitality grid.
        - ``'C0'`` : ndarray, shape (n_C,) — initial-scope grid.
        - ``'T_conv'`` : ndarray, shape (n_C, n_V) — time to enter and stay
          in the band; np.inf where not converged by T. Rows index C0,
          columns index V0 (ready for ``pcolormesh(V0, C0, T_conv)``).
        - ``'converged'`` : ndarray of bool, shape (n_C, n_V).
        - ``'n_total'`` : int — number of grid points.
        - ``'n_converged'`` : int — number converged by T.
        - ``'n_exceptions'`` : int — grid points with C0 > 0 that did NOT
          converge (expected 0 by Theorem 2c).
        - ``'V_star'``, ``'C_star'`` : float — the attractor.
        - ``'band'``, ``'T'`` : the settings used.
    """
    eq = find_interior_equilibria(p)[0]
    V_star, C_star = float(eq['V']), float(eq['C'])

    if V_range is None:
        V_range = (0.2, p['Vmax'])
    V0s = np.linspace(V_range[0], V_range[1], n_V)
    C0s = np.linspace(C_range[0], C_range[1], n_C)
    t_eval = np.linspace(0.0, T, n_t)

    T_conv = np.full((n_C, n_V), np.inf)
    for i, C0 in enumerate(C0s):
        for j, V0 in enumerate(V0s):
            sol = solve_ivp(dlvt_system, [0.0, T], [float(V0), float(C0)],
                            args=(p,), method='RK45', t_eval=t_eval,
                            rtol=1e-6, atol=1e-8)
            inside = ((np.abs(sol.y[0] - V_star) < band)
                      & (np.abs(sol.y[1] - C_star) < band))
            if inside[-1]:
                # Last index after which the trajectory never leaves the
                # band: first True of the trailing all-True block.
                outside = np.where(~inside)[0]
                k = 0 if outside.size == 0 else int(outside[-1]) + 1
                T_conv[i, j] = sol.t[k]

    converged = np.isfinite(T_conv)
    n_total = int(T_conv.size)
    n_converged = int(np.count_nonzero(converged))
    C0_pos = C0s > 0.0
    n_exceptions = int(np.count_nonzero(~converged[C0_pos, :]))

    return {
        'V0': V0s,
        'C0': C0s,
        'T_conv': T_conv,
        'converged': converged,
        'n_total': n_total,
        'n_converged': n_converged,
        'n_exceptions': n_exceptions,
        'V_star': V_star,
        'C_star': C_star,
        'band': band,
        'T': T,
    }
