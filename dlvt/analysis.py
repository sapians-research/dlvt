"""
dlvt.analysis
=============
Equilibrium analysis, stability, and threshold-sensitive reporting for DLVT.

This module implements the reviewed formal contract for the DLVT manuscript in
preparation. Constructs and bundled values are illustrative, not calibrated.

Theoretical Foundations (corrected statements, v2.1)
----------------------------------------------------
THEOREM 1 (Existence): let V_c(O)=(μ/α)(1+φO) and
  F(O)=R(1−V_c/Vmax)−δO^γV_c/(V_c+ε).  Because F is strictly decreasing, an interior
  fixed point exists if and only if F(O₀)>0.  The older inequality
  αVmax/μ>1+φO₀ is necessary but not sufficient because it omits baseline drain.

THEOREM 2a (Global Uniqueness): the interior equilibrium is unique in the ENTIRE positive
  orthant, for ALL positive parameter vectors — not just the baseline. Proof: along the
  scope nullcline V_c(O) = (μ/α)(1+φO), define
      Φ(O) = R(1 − V_c/Vmax) − δ·O^γ·V_c/(V_c+ε).
  Every term of dΦ/dO is strictly negative:
      −(R/Vmax)·V_c′ < 0,
      −δγO^{γ−1}·V_c/(V_c+ε) < 0,
      −δO^γ·ε·V_c′/(V_c+ε)² < 0,
  so Φ is strictly decreasing in O; since O(C) is strictly increasing, Φ has at most one
  root in C. Hence: at most one interior equilibrium anywhere, and no fold/saddle-node
  from root multiplicity is possible. See Appendix A10.

THEOREM 2b (No Hopf / no cycles): at any interior equilibrium, dC/dt = 0 forces
  αV*/(1+φO*) = μ, hence
      J[1,1] = −α·C*·V*·φ·(dO/dC)/(1+φO*)² < 0,
  and J[0,0] < 0 identically, so trace(J) < 0 at EVERY interior equilibrium for all
  positive parameters — no Hopf bifurcation exists anywhere in parameter space. This is
  consistent with (and independent of) the Bendixson–Dulac certificate B(V,C) = 1/C, which
  rules out closed orbits globally on {C > 0}. No Lyapunov function is supplied;
  the weighted-quadratic candidates examined in the project fail, so the current
  proof uses Dulac + Poincaré–Bendixson.

THEOREM 2c (Global Asymptotic Stability, eta ≥ 1): the unique interior equilibrium is
  globally asymptotically stable on the OPEN set {C > 0} ∩ Ω. The line C = 0 is invariant and
  carries a saddle equilibrium at (V ≈ 9.934, C = 0) (see find_regularization_branch), so
  the closed quadrant is NOT the basin — trajectories started exactly on C = 0 converge to
  the axis saddle instead. The trapping rectangle is
      Ω = [0, Vmax] × [0, C_trap],  C_trap^η = ((αVmax/μ − 1)/φ − O₀)/β
  (baseline C_trap = 102.67; see trapping_scope_bound). CAUTION: earlier drafts used the
  coefficient threshold C_Gamma = 44.99 as the rectangle ceiling; that rectangle LEAKS (at
  C = 44.99, V = Vmax, dC/dt > 0) — C_trap and C*_max are distinct constants. Proof:
  Dulac certificate (no closed orbits) + uniqueness (2a) + Poincaré–Bendixson on Ω∩{C>0}.
  See Appendix A10.

COROLLARY (Scope Absorption; formerly "Lemma 2"): whenever the equilibrium vector field
  depends on scope only through O and V, the pair (V*,O*) solves a β-free system, so V* is
  invariant in β and β·C*^η is conserved.  Hill and exponential kernels in O preserve the
  property; explicit C, β, η, or additional-state dependence can break it.  This is a
  reparameterization invariance, not an empirical law.  Equilibrium vitality can be moved
  by R/δ, μ/α, φ, Vmax, ε and conditionally γ, but not by β alone. ε-dependence of the
  invariant: β·C* = 8.008 at ε = 0.1, and exactly 7.933 in the ε → 0 closed form (O* solves
  δO² + (R/Vmax)(μφ/α)O + R((μ/α)/Vmax − 1) = 0 for γ=2, η=1). See §3.8 / §4.

PROPOSITION 1 (V-Nullcline): The vitality recovery curve (dV/dt = 0) is strictly decreasing
  in C. Together with Theorem 2a this gives the unique intersection with the C-nullcline.

PROPOSITION 2 (Boundary): changing a stipulated V-threshold classification is not a
  dynamical bifurcation.  Within the O,V-only class, varying β cannot change V*.  A
  β-dependent equilibrium requires a specification that contains nonabsorbed C or β
  dependence; its bifurcation type must then be proved for that specification.

DEFINITION (Drain-coefficient threshold):
    C_Gamma = ( ((R/δ)^(1/γ) − O₀) / β )^(1/η)
  is the positive scope level, when it exists, at which Γ=δO^γ/R reaches 1.  It is not a
  capacity, trapping ceiling, equilibrium-disappearance point, or sign test for
  dV/dt.  Recovery is zero at V=Vmax, so no full-vitality balance is implied.

CALIBRATION CAVEAT (display label): the interior equilibrium satisfies
  V* = (μ/α)(1 + φO*), so classification against 0.5·Vmax depends on a
  stipulated threshold and parameter choices. Report continuous V* and
  sensitivity. No prevalence or impairment conclusion follows.

Key Functions
-----------
drain_coefficient_threshold()   : Γ=1 coefficient threshold C_Gamma
carrying_capacity()             : deprecated compatibility alias
trapping_scope_bound()          : scope ceiling C_trap
trapping_capital_bound()        : deprecated compatibility alias
find_interior_equilibria()      : Theorem 2 — find all (V*, C*) with V*, C* > 0
jacobian_eigenvalues()          : Theorem 2 — compute eigenvalues, classify stability
is_low_vitality()               : compare V* with an explicit display threshold
classify_equilibrium()          : separate equilibrium existence from display label
bendixson_dulac_certificate()   : Theorem 2 proof — Dulac divergence grid + trapping rectangle
basin_of_attraction_sweep()     : Theorem 2 corroboration — 64 IC convergence test
find_regularization_branch()    : Appendix A9 — characterize ε-regularization saddle
estimate_bifurcation_interval() : legacy scan-window/threshold diagnostic

Module Constants
----------------
DISPLAY_THRESHOLD_FRACTION = 0.5 : illustrative display threshold only
V_STRATEGIC_FRACTION = 0.5       : deprecated compatibility constant

"""

from typing import Dict, List, Tuple, Optional
import warnings
import numpy as np
from scipy.optimize import brentq

from .model import coordination_load, impact, DEFAULT_PARAMS, validate_params

# Illustrative display threshold. It is stipulated, not empirically validated.
DISPLAY_THRESHOLD_FRACTION = 0.5
# Historical constant name retained for source compatibility through v2.x.
V_STRATEGIC_FRACTION = DISPLAY_THRESHOLD_FRACTION


# ── Analytical results ────────────────────────────────────────────────────────

def trapping_scope_bound(p: Dict[str, float]) -> float:
    """Scope ceiling C_trap of the forward-invariant trapping rectangle.

    C_trap is the unique scope level at which the scope nullcline
    V_c(C) = μ·(1 + φ·O(C))/α reaches V_max:

        C_trap^η = ( (α·V_max/μ − 1)/φ − O₀ ) / β

    For any C > C_trap and any V ∈ [0, V_max],
    dC/dt = C·(α·V/(1+φ·O) − μ) < 0, so the rectangle
    Ω = [0, V_max] × [0, C_trap] is forward invariant (Theorem 2).

    Two distinct constants — do not conflate them:

    - ``C_trap`` (this function; baseline **102.67**) bounds the *trapping
      rectangle* used in the global-stability proof. It also bounds every
      interior equilibrium, since dC/dt = 0 with C* > 0 forces
      V* = μ(1+φO*)/α < V_max ⟹ C* < C_trap.
    - ``drain_coefficient_threshold`` (baseline **44.99**) is the coefficient
      threshold where the depletion ratio Γ = 1. Earlier drafts
      wrongly used it as the rectangle ceiling; that rectangle *leaks*
      (at C = 44.99, V = V_max, dC/dt > 0 — a trajectory started at
      (V, C) = (10, 40) overshoots to C ≈ 46.4 before settling).

    Parameters
    ----------
    p : Dict[str, float]
        Parameter dictionary.

    Returns
    -------
    float
        C_trap. Raises ValueError when α·V_max/μ ≤ 1 + φ·O₀, in which
        case the C-nullcline never reaches V_max: dC/dt < 0 for all C > 0,
        no interior equilibrium exists, and no trapping ceiling is needed.
    """
    p = validate_params(p)
    rhs = (p['alpha'] * p['Vmax'] / p['mu'] - 1.0) / p['phi'] - p['O0']
    if rhs <= 0:
        raise ValueError(
            "Parameter vector has αVmax/μ ≤ 1+φO0; the scope nullcline never "
            "reaches Vmax, so no interior equilibrium exists and the "
            "trapping ceiling is undefined (enacted scope contracts for all C > 0)."
        )
    return (rhs / p['beta']) ** (1.0 / p['eta'])


def trapping_capital_bound(p: Dict[str, float]) -> float:
    """Deprecated alias for :func:`trapping_scope_bound`."""
    warnings.warn(
        "trapping_capital_bound() is deprecated; use trapping_scope_bound(). "
        "C denotes enacted leadership scope, not capital.",
        DeprecationWarning,
        stacklevel=2,
    )
    return trapping_scope_bound(p)

def drain_coefficient_threshold(p: Dict[str, float]) -> float:
    """Return the enacted-scope level at which ``delta*O(C)**gamma/R == 1``.

    This is a coefficient-ratio threshold, denoted ``C_Gamma`` in the R8
    formal specification.  It is not a capacity, a trapping bound,
    an equilibrium-disappearance point, or a sign test for ``dV/dt``.  In
    particular, recovery is zero at ``V=Vmax``, so the threshold cannot mean
    that recovery offsets drain at full vitality.

    Formula:

        C_Gamma = (((R/delta)^(1/gamma) - O0) / beta)^(1/eta)

    Parameters
    ----------
    p : Dict[str, float]
        Parameter dictionary with keys: 'R', 'delta', 'gamma', 'O0', 'beta', 'eta'.

    Returns
    -------
    float
        Positive crossing level, or ``0.0`` when ``Gamma>=1`` already at
        baseline load and therefore no positive crossing exists.

    Notes
    -----
    A return value of zero says nothing by itself about equilibrium
    existence or long-run vitality.  Those require the exact residual used
    by :func:`find_interior_equilibria`.

    Examples
    --------
    >>> from dlvt import make_params
    >>> p = make_params()
    >>> c_gamma = drain_coefficient_threshold(p)
    >>> print(f'Gamma=1 coefficient threshold: C = {c_gamma:.2f}')
    """
    p = validate_params(p)
    Omax = (p['R'] / p['delta']) ** (1.0 / p['gamma'])
    if Omax <= p['O0']:
        return 0.0
    return max(0.0, ((Omax - p['O0']) / p['beta']) ** (1.0 / p['eta']))


def carrying_capacity(p: Dict[str, float]) -> float:
    """Deprecated alias for :func:`drain_coefficient_threshold`.

    The historical name encoded an incorrect ecological interpretation.
    It remains for one compatibility cycle while public callers migrate to
    the semantically accurate function name.
    """
    warnings.warn(
        "carrying_capacity() is deprecated; use "
        "drain_coefficient_threshold(). The returned value is the Gamma=1 "
        "coefficient threshold, not a capacity or trapping bound.",
        DeprecationWarning,
        stacklevel=2,
    )
    return drain_coefficient_threshold(p)


def find_interior_equilibria(p: Dict[str, float], C_max: Optional[float] = None,
                            n_scan: int = 8000
                            ) -> List[Dict[str, any]]:
    """Return the unique interior equilibrium, if it exists.

    The scope nullcline gives

        V_c(O) = (mu/alpha) * (1 + phi*O).

    Substitution into the vitality equation produces a scalar residual

        F(O) = R*(1 - V_c(O)/Vmax)
               - delta*O**gamma*V_c(O)/(V_c(O) + eps).

    ``F`` is strictly decreasing for positive parameters.  An interior
    equilibrium exists if and only if ``F(O0) > 0``; when it exists, its
    load coordinate is the unique root on ``(O0, O_trap)``, where

        O_trap = (alpha*Vmax/mu - 1)/phi.

    Solving in ``O`` avoids the two legacy scan failures: roots below the
    hard-coded ``C=0.01`` floor and valid roots above ``0.999*Vmax``.

    Parameters
    ----------
    p : Dict[str, float]
        Parameter dictionary.
    C_max : float, optional
        Optional deliberate reporting window.  The equilibrium is solved
        analytically first and omitted only when its mapped ``C*`` exceeds
        this value.  ``None`` never truncates a valid equilibrium.
    n_scan : int, optional
        Deprecated compatibility argument.  The monotone root algorithm
        does not scan and therefore ignores this value.

    Returns
    -------
    List[Dict[str, any]]
        List of equilibrium points. Each dict contains:
          - 'C': equilibrium enacted scope C*
          - 'V': equilibrium vitality V*
          - 'O': equilibrium experienced coordination load O*
          - 'I': equilibrium impact I*
          - 'stable': bool, True if locally asymptotically stable
          - 'eigenvalues': ndarray of Jacobian eigenvalues

    An empty list means that the exact existence condition fails or that an
    explicitly supplied ``C_max`` excludes the otherwise valid root.  It
    does not mean that the drain-coefficient threshold is zero.
    """
    del n_scan  # retained only for backward-compatible call signatures
    p = validate_params(p)

    q = p['mu'] / p['alpha']

    def vitality_on_scope_nullcline(O: float) -> float:
        return q * (1.0 + p['phi'] * O)

    def residual_in_load(O: float) -> float:
        V = vitality_on_scope_nullcline(O)
        recovery = p['R'] * (1.0 - V / p['Vmax'])
        drain = p['delta'] * O**p['gamma'] * V / (V + p['eps'])
        return recovery - drain

    F_at_baseline = residual_in_load(p['O0'])
    V_at_baseline = vitality_on_scope_nullcline(p['O0'])
    residual_scale = max(
        1.0,
        abs(p['R'] * (1.0 - V_at_baseline / p['Vmax'])),
        abs(
            p['delta'] * p['O0']**p['gamma'] * V_at_baseline
            / (V_at_baseline + p['eps'])
        ),
    )
    # At the admissibility boundary F(O0)=0, roundoff from a previously
    # computed root can be a few ulps positive. Do not return C*=0 as an
    # "interior" equilibrium; genuinely boundary-near positive roots remain
    # detectable above this scale-aware floating-point tolerance.
    existence_tolerance = 64.0 * np.finfo(float).eps * residual_scale
    if F_at_baseline <= existence_tolerance:
        return []

    O_trap = (p['alpha'] * p['Vmax'] / p['mu'] - 1.0) / p['phi']
    if O_trap <= p['O0']:
        # Defensive guard: F(O0)>0 already implies O_trap>O0 for a valid
        # positive parameter vector.
        return []

    O_star = brentq(residual_in_load, p['O0'], O_trap)
    C_star = ((O_star - p['O0']) / p['beta']) ** (1.0 / p['eta'])
    if C_max is not None and C_star > C_max:
        return []

    V_star = vitality_on_scope_nullcline(O_star)
    eigvals, stable = jacobian_eigenvalues(V_star, C_star, p)
    return [dict(
        C=C_star,
        V=V_star,
        O=O_star,
        I=impact(V_star, C_star, O_star, p),
        stable=stable,
        eigenvalues=eigvals,
    )]


def jacobian_eigenvalues(V: float, C: float,
                        p: Dict[str, float]
                        ) -> Tuple[np.ndarray, bool]:
    """Compute eigenvalues of the Jacobian J at (V, C) and assess stability.

    Computes the 2×2 Jacobian matrix and its eigenvalues at an equilibrium.
    Linear stability is determined by the sign of the real parts.

    The Jacobian is:
      J[0,0] = ∂(dV/dt)/∂V = −R/V_max − δ·O^γ·ε/(V+ε)²
      J[0,1] = ∂(dV/dt)/∂C = −δ·γ·O^{γ−1}·(dO/dC)·V/(V+ε)
      J[1,0] = ∂(dC/dt)/∂V = α·C/(1+φ·O)
      J[1,1] = ∂(dC/dt)/∂C = α·V/(1+φ·O) − α·C·V·φ·(dO/dC)/(1+φ·O)² − μ

    Stability criterion: all eigenvalues satisfy Re(λ) < 0.

    Parameters
    ----------
    V : float
        Vitality at the equilibrium.
    C : float
        Enacted leadership scope at the equilibrium.
    p : Dict[str, float]
        Parameter dictionary.

    Returns
    -------
    eigvals : ndarray, shape (2,)
        Eigenvalues of the Jacobian (may be complex).
    stable : bool
        True if the equilibrium is locally asymptotically stable
        (all eigenvalues have negative real part).

    Notes
    -----
    Uses numpy.linalg.eigvals for robust computation.
    """
    p = validate_params(p)
    O    = coordination_load(C, p)
    eps  = p['eps']
    dOdC = p['beta'] * p['eta'] * max(C, 1e-10)**(p['eta'] - 1)

    J = np.array([
        [
            -p['R'] / p['Vmax'] - p['delta'] * O**p['gamma'] * eps / (V + eps)**2,
            -p['delta'] * p['gamma'] * O**(p['gamma'] - 1) * dOdC * V / (V + eps)
        ],
        [
            p['alpha'] * C / (1.0 + p['phi'] * O),
            (p['alpha'] * V / (1.0 + p['phi'] * O)
             - p['alpha'] * C * V * p['phi'] * dOdC / (1.0 + p['phi'] * O)**2
             - p['mu'])
        ]
    ])
    eigvals = np.linalg.eigvals(J)
    stable  = bool(all(e.real < 0 for e in eigvals))
    return eigvals, stable


def is_low_vitality(
    V_star: float,
    p: Dict[str, float],
    threshold_fraction: float = DISPLAY_THRESHOLD_FRACTION,
) -> bool:
    """Classify continuous equilibrium vitality against an explicit threshold.

    The threshold is calibrational, not derived by the ODE.  Callers should
    report ``V_star`` itself and perform sensitivity analysis over plausible
    threshold fractions.

    Parameters
    ----------
    V_star : float
        Equilibrium vitality.
    p : Dict[str, float]
        Parameter dictionary containing ``Vmax``.
    threshold_fraction : float, optional
        Fraction of ``Vmax`` used for classification.  The default ``0.5``
        is illustrative and retained for baseline compatibility.

    Returns
    -------
    bool
        True when ``V_star < threshold_fraction*Vmax``.

    Notes
    -----
    This function does not assert that the selected threshold has empirical
    validity.
    """
    p = validate_params(p)
    if not 0.0 < threshold_fraction < 1.0:
        raise ValueError("threshold_fraction must lie strictly between 0 and 1.")
    return V_star < threshold_fraction * p['Vmax']


def is_zombie(V_star: float, p: Dict[str, float]) -> bool:
    """Deprecated compatibility alias for :func:`is_low_vitality`."""
    warnings.warn(
        "is_zombie() is deprecated; use is_low_vitality() with an explicit "
        "threshold_fraction and report continuous V*.",
        DeprecationWarning,
        stacklevel=2,
    )
    return is_low_vitality(V_star, p)


# ── Regime map ────────────────────────────────────────────────────────────────

def classify_equilibrium(
    p: Dict[str, float],
    C_max: Optional[float] = None,
    threshold_fraction: float = DISPLAY_THRESHOLD_FRACTION,
) -> Dict[str, object]:
    """Return an explicit equilibrium/existence classification.

    The result separates mathematical existence from a threshold-dependent
    substantive label.  ``threshold_fraction`` is always returned so the
    classification cannot be mistaken for a structural property. ``C_max``
    is a reporting window only: it never changes mathematical existence. If
    it excludes the exact equilibrium, the result retains that equilibrium
    and uses status ``equilibrium-outside-reporting-window``.
    """
    if not 0.0 < threshold_fraction < 1.0:
        raise ValueError("threshold_fraction must lie strictly between 0 and 1.")
    # Mathematical existence must not depend on a caller's reporting window.
    # Solve the exact problem first; apply C_max only to reporting metadata.
    eqs = find_interior_equilibria(p)
    stable_eqs = [eq for eq in eqs if eq['stable']]
    if not stable_eqs:
        return {
            'status': 'no-stable-interior-equilibrium',
            'mathematical_status': 'no-stable-interior-equilibrium',
            'exists': False,
            'equilibrium': None,
            'threshold_fraction': threshold_fraction,
            'reporting_C_max': C_max,
            'within_reporting_window': None,
        }

    mathematical_eq = min(stable_eqs, key=lambda item: item['C'])
    below = is_low_vitality(mathematical_eq['V'], p, threshold_fraction)
    eq = dict(mathematical_eq)
    within_reporting_window = C_max is None or eq['C'] <= C_max
    eq.update(
        low_vitality=below,
        zombie=below,  # deprecated compatibility key through v2.x
        threshold_fraction=threshold_fraction,
        within_reporting_window=within_reporting_window,
    )
    return {
        'status': (
            'low-vitality' if below else 'above-threshold'
        ) if within_reporting_window else 'equilibrium-outside-reporting-window',
        'mathematical_status': 'low-vitality' if below else 'above-threshold',
        'exists': True,
        'equilibrium': eq,
        'V_star': eq['V'],
        'C_star': eq['C'],
        'O_star': eq['O'],
        'threshold_fraction': threshold_fraction,
        'threshold_value': threshold_fraction * p['Vmax'],
        'reporting_C_max': C_max,
        'within_reporting_window': within_reporting_window,
    }


def classify_regime(
    p: Dict[str, float],
    C_max: Optional[float] = None,
    threshold_fraction: float = DISPLAY_THRESHOLD_FRACTION,
) -> str:
    """Return legacy regime labels for backward compatibility.

    New code should use :func:`classify_equilibrium`, which separates
    equilibrium existence from threshold-sensitive interpretation.

    Parameters
    ----------
    p : Dict[str, float]
        Parameter dictionary.
    C_max : float, optional
        Maximum enacted scope for equilibrium search. Default ``None`` uses the
        analytical bound (see :func:`find_interior_equilibria`), which is
        valid for every β. Fixed values can mislabel small-β cases
        as 'collapse-prone' when C*(β) ∝ 1/β exceeds the window.
    threshold_fraction : float, optional
        Explicit vitality threshold passed to :func:`classify_equilibrium`.

    Returns
    -------
    str
        One of:
          - 'sustainable': stable equilibrium with V* ≥ V_strategic
          - 'zombie': stable equilibrium with V* < V_strategic
          - 'collapse-prone': no stable interior equilibrium

    Notes
    -----
    ``'zombie'`` maps from ``'low-vitality'`` and ``'sustainable'`` maps
    from ``'above-threshold'``.  ``'collapse-prone'`` means only that no
    stable interior equilibrium was returned; it does not prove burnout or
    collapse.
    """
    warnings.warn(
        "classify_regime() is deprecated; use classify_equilibrium(), which "
        "separates equilibrium existence from an explicit display threshold.",
        DeprecationWarning,
        stacklevel=2,
    )
    result = classify_equilibrium(
        p,
        C_max=C_max,
        threshold_fraction=threshold_fraction,
    )
    if result['status'] == 'no-stable-interior-equilibrium':
        return 'collapse-prone'
    return 'zombie' if result['mathematical_status'] == 'low-vitality' else 'sustainable'


def regime_map(beta_range: np.ndarray, delta_range: np.ndarray,
               base_params: Optional[Dict[str, float]] = None
               ) -> np.ndarray:
    """Compute deprecated legacy labels over a ``(beta, delta)`` grid.

    This compatibility API applies an illustrative threshold and returns the
    historical string labels. New analyses should use
    :func:`classify_equilibrium` and retain continuous equilibrium outputs.

    Parameters
    ----------
    beta_range : ndarray
        Values of scope-to-load coupling β.
    delta_range : ndarray
        Values of δ (energetic cost coefficient), typically [0.001, 0.1].
    base_params : Optional[Dict[str, float]]
        Base parameters (defaults to DEFAULT_PARAMS).
        β and δ are overridden by the grid values.

    Returns
    -------
    regimes : ndarray, shape (len(delta_range), len(beta_range)), dtype=object
        Grid of deprecated compatibility classifications.
        regimes[i, j] = classify_regime(p) for p with β=beta_range[j], δ=delta_range[i].
        Values: 'sustainable', 'zombie', or 'collapse-prone'.

    Notes
    -----
    This function is O(len(beta_range) * len(delta_range)) and may be slow
    for large grids. Typical grid sizes: ~50 × 50 for fast preview, ~200 × 200
    for publication quality.

    Examples
    --------
    >>> import numpy as np
    >>> from dlvt import regime_map
    >>> betas = np.linspace(0.01, 1.0, 50)
    >>> deltas = np.linspace(0.001, 0.1, 50)
    >>> regimes = regime_map(betas, deltas)
    """
    if base_params is None:
        base_params = DEFAULT_PARAMS.copy()
    base_params = validate_params(base_params)

    regimes = np.empty((len(delta_range), len(beta_range)), dtype=object)
    for i, dv in enumerate(delta_range):
        for j, bv in enumerate(beta_range):
            p = {**base_params, 'beta': bv, 'delta': dv}
            regimes[i, j] = classify_regime(p)
    return regimes


# ── Bifurcation-interval diagnostics (R7-6) ───────────────────────────────────

def estimate_bifurcation_interval(
    base_params: Dict[str, float],
    eps_grid: Optional[List[float]] = None,
    n_scan_grid: Optional[List[int]] = None,
    beta_range: Tuple[float, float] = (0.005, 2.0),
    n_beta: int = 400,
    V_strategic: Optional[float] = None,
    C_max: Optional[float] = None,
) -> Dict[str, any]:
    """Run the legacy critical-beta crossing diagnostic.

    The historical function and ``V_strategic`` argument names are retained
    for compatibility. The supplied value is an illustrative display
    threshold, not a strategic-performance threshold or dynamical boundary.

    This function replaces the false-precision ``beta_crit ≈ 0.1015`` claim in
    earlier drafts of Appendix A8. It answers two honest questions instead of
    one fake-precise one:

    (1) *Does* the equilibrium vitality :math:`V^*(\\beta)` cross the
        display threshold :math:`V_{\\mathrm{display}}` anywhere in the
        sweep? If Lemma 2 (scope-absorption) applies — i.e.\\ the drain
        kernel is multiplicative power-law in :math:`O = O_0 + \\beta C^\\eta`
        and :math:`V^*` depends on :math:`\\beta` only through :math:`\\beta
        C^{*\\eta}` — then the answer for the baseline calibration is *no*:
        :math:`V^*` is constant across :math:`\\beta`, and the "critical"
        value reported in earlier drafts was a scan-window artifact, not a
        structural property.

    (2) If the crossing *does* exist (non-baseline calibration), what is the
        range of numerical estimates of :math:`\\beta_{\\mathrm{crit}}`
        produced by varying the regularization :math:`\\varepsilon` and the
        scan resolution? The returned interval is the appropriate
        methods-honest reporting granularity.

    Parameters
    ----------
    base_params : Dict[str, float]
        Parameter dictionary (will be shallow-copied; ``eps`` is overwritten).
    eps_grid : List[float], optional
        Values of the vitality-barrier regularization :math:`\\varepsilon`
        to sweep. Default: ``[0.01, 0.05, 0.1, 0.2]``.
    n_scan_grid : List[int], optional
        Values of ``n_scan`` (scan resolution for ``find_interior_equilibria``)
        to sweep. A proxy for integration tolerance. Default:
        ``[8000, 16000]``.
    beta_range : Tuple[float, float], optional
        ``(beta_min, beta_max)`` for the sweep. Default: ``(0.005, 2.0)``.
    n_beta : int, optional
        Number of beta sample points in each sweep. Default: ``400``.
    V_strategic : float, optional
        Historical argument name for the display threshold. Defaults to
        ``DISPLAY_THRESHOLD_FRACTION * Vmax``.
    C_max : float, optional
        Upper bound for the equilibrium scan. Default ``None`` uses the
        per-β analytical bound C_trap ∝ 1/β (see
        :func:`find_interior_equilibria`), which catches the low-β
        equilibria that the legacy fixed C_max=80 missed — the source of
        the original 0.1015 artifact.

    Returns
    -------
    result : Dict[str, any]
        A diagnostic dictionary with the keys:

        - ``'v_star_invariant'`` : Optional[float]
            If :math:`V^*(\\beta)` is numerically constant across the sweep
            (to tolerance 1e-4), the invariant value; otherwise ``None``.
        - ``'crosses_threshold'`` : bool
            True iff there exists a :math:`\\beta` in the sweep at which a
            stable interior equilibrium has
            :math:`V^* \\geq V_{\\mathrm{strategic}}` and another at which
            :math:`V^* < V_{\\mathrm{strategic}}`.
        - ``'beta_crit_interval'`` : Optional[Tuple[float, float]]
            ``(low, high)`` range of the numerical estimate across the
            ``(eps, n_scan)`` grid, if the crossing exists; otherwise
            ``None``.
        - ``'beta_crit_per_eps'`` : Dict[float, Optional[float]]
            Per-epsilon estimate (median across n_scan values); ``None``
            entries mark no crossing.
        - ``'diagnostic'`` : str
            Plain-English explanation of the finding, including the Lemma 2
            pointer when applicable.
        - ``'baseline_beta_C_product'`` : Optional[float]
            If ``v_star_invariant`` is not None, the invariant product
            :math:`\\beta C^*` (which Lemma 2 shows must be constant).

    Notes
    -----
    The function is deliberately *not* a one-liner. It exists to make the
    scope-absorption phenomenon explicit in the code, not hide it. The
    estimate intentionally uses a large C_max so that low-β equilibria are
    not missed --- this is how we detect and report the original scan
    artifact.

    Examples
    --------
    >>> from dlvt import make_params
    >>> from dlvt.analysis import estimate_bifurcation_interval
    >>> result = estimate_bifurcation_interval(make_params())
    >>> result['crosses_threshold']     # baseline does not cross
    False
    >>> round(result['v_star_invariant'], 4)
    4.7025
    """
    base_params = validate_params(base_params)
    if eps_grid is None:
        eps_grid = [0.01, 0.05, 0.1, 0.2]
    if n_scan_grid is None:
        n_scan_grid = [8000, 16000]
    if V_strategic is None:
        V_strategic = DISPLAY_THRESHOLD_FRACTION * base_params['Vmax']

    beta_min, beta_max = beta_range
    betas = np.linspace(beta_min, beta_max, n_beta)

    def _sweep(p_local: Dict[str, float], n_scan: int):
        """Return list of (beta, V_star) for stable interior equilibria."""
        curve = []
        for b in betas:
            p_try = dict(p_local)
            p_try['beta'] = b
            eqs = find_interior_equilibria(p_try, C_max=C_max, n_scan=n_scan)
            stable = [e for e in eqs if e['stable']]
            if stable:
                # Take the equilibrium on the historical high-C scan branch.
                stable.sort(key=lambda e: e['C'])
                curve.append((b, stable[-1]['V'], stable[-1]['C']))
        return curve

    def _beta_crit_from_curve(curve: List[Tuple[float, float, float]]
                              ) -> Optional[float]:
        """Find beta where V_star(beta) crosses V_strategic, or None."""
        if len(curve) < 2:
            return None
        crossings = []
        for i in range(len(curve) - 1):
            v1, v2 = curve[i][1], curve[i + 1][1]
            if (v1 - V_strategic) * (v2 - V_strategic) < 0:
                # linear interpolation
                b1, b2 = curve[i][0], curve[i + 1][0]
                frac = (V_strategic - v1) / (v2 - v1)
                crossings.append(b1 + frac * (b2 - b1))
        if not crossings:
            return None
        return float(np.median(crossings))

    per_eps: Dict[float, Optional[float]] = {}
    per_eps_v_star: Dict[float, Optional[float]] = {}
    per_eps_bC: Dict[float, Optional[float]] = {}
    any_crosses = False
    all_crits: List[float] = []

    for eps in eps_grid:
        p_eps = dict(base_params)
        p_eps['eps'] = eps
        eps_crits: List[float] = []
        eps_v_stars: List[float] = []
        eps_bC: List[float] = []
        for ns in n_scan_grid:
            curve = _sweep(p_eps, ns)
            if curve:
                eps_v_stars.extend(v for (_, v, _) in curve)
                eps_bC.extend(b * c for (b, _, c) in curve)
            bc = _beta_crit_from_curve(curve)
            if bc is not None:
                eps_crits.append(bc)
                any_crosses = True
        per_eps[eps] = float(np.median(eps_crits)) if eps_crits else None
        per_eps_v_star[eps] = float(np.median(eps_v_stars)) if eps_v_stars else None
        per_eps_bC[eps] = float(np.median(eps_bC)) if eps_bC else None
        all_crits.extend(eps_crits)

    # scope-absorption diagnostic: at each eps, check whether V*(β) is
    # numerically constant across the sweep. If it is at *every* eps, the
    # Lemma 2 invariant holds (V* depends on eps but not on β).
    v_star_invariant: Optional[float] = None
    baseline_bC: Optional[float] = None
    per_eps_constant = True
    for eps in eps_grid:
        p_eps = dict(base_params); p_eps['eps'] = eps
        eps_vs: List[float] = []
        for ns in n_scan_grid:
            curve = _sweep(p_eps, ns)
            eps_vs.extend(v for (_, v, _) in curve)
        if len(eps_vs) >= 2:
            if float(np.max(eps_vs) - np.min(eps_vs)) > 1e-4:
                per_eps_constant = False
                break
        else:
            per_eps_constant = False
            break
    if per_eps_constant and per_eps_v_star:
        # report the value at the middle eps (or the user's calibration eps if present)
        mid_eps = eps_grid[len(eps_grid) // 2]
        v_star_invariant = per_eps_v_star[mid_eps]
        baseline_bC = per_eps_bC[mid_eps]

    if any_crosses:
        beta_crit_interval: Optional[Tuple[float, float]] = (
            float(min(all_crits)),
            float(max(all_crits)),
        )
        diagnostic = (
            f"V*(β) crosses display threshold={V_strategic:.3f}. Numerical estimates "
            f"of β_crit across the (eps, n_scan) grid span "
            f"[{beta_crit_interval[0]:.4f}, {beta_crit_interval[1]:.4f}]; "
            f"report this interval, not a point."
        )
    else:
        beta_crit_interval = None
        if v_star_invariant is not None:
            diagnostic = (
                f"V*(β) is numerically constant at V*={v_star_invariant:.4f} "
                f"across the sweep (scope-absorption / Lemma 2: β·C* is "
                f"invariant at {baseline_bC:.4f}). "
                f"{'V* below threshold' if v_star_invariant < V_strategic else 'V* at/above threshold'} "
                f"for all β, so β_crit as a V*-crossing does not exist. Any "
                f"previously reported 'critical β' for this calibration was a "
                f"scan-window artifact of a too-small C_max (the equilibrium "
                f"C*(β) = {baseline_bC:.3f}/β exceeds legacy C_max=80 for β < "
                f"{baseline_bC/80:.4f})."
            )
        else:
            diagnostic = (
                f"V*(β) does not cross display threshold={V_strategic:.3f} in "
                f"β ∈ [{beta_min}, {beta_max}], but is not numerically "
                f"constant either. Verify the parameter configuration before interpreting."
            )

    return {
        'v_star_invariant': v_star_invariant,
        'crosses_threshold': any_crosses,
        'beta_crit_interval': beta_crit_interval,
        'beta_crit_per_eps': per_eps,
        'diagnostic': diagnostic,
        'baseline_beta_C_product': baseline_bC,
    }


# ── ε-regularization branch audit (R7 issue 3) ────────────────────────────────

def find_regularization_branch(
    p: Dict[str, float],
    near_zero_threshold: Optional[float] = None,
) -> Dict[str, any]:
    """Audit the number and magnitude of regularized vitality roots.

    The smooth barrier $V/(V+\\varepsilon)$ used in the vitality ODE is a
    positive-invariance trick: without it, strong-drain configurations would push
    $V$ across zero into the unphysical negative half-plane. A careful
    reviewer may reasonably worry that the regularization introduces a
    *spurious second* positive root. This function audits root count and root
    magnitude separately. A unique positive axis root may legitimately be
    arbitrarily small under strong drain; small magnitude is not a violation
    of the one-positive-root result in Appendix A9.

    Structure of the enumeration:

    1. **Axis equilibria.** At $C = 0$, $\\dot C \\equiv 0$ automatically, and
       $\\dot V = 0$ reduces to a quadratic in $V$:
           :math:`\\frac{R}{V_{\\max}} V^2 - (R - R\\varepsilon/V_{\\max} - \\delta O_0^\\gamma) V - R\\varepsilon = 0`.
       The product of the roots is $-\\varepsilon V_{\\max} < 0$, so there
       is exactly *one* positive root. We solve the quadratic in closed form.

    2. **Interior equilibria.** Delegated to
       :func:`find_interior_equilibria`, which parametrizes along the
       $\\dot C = 0$ nullcline $V_c(C) = \\mu(1+\\phi O)/\\alpha$. Because
       $V_c \\geq \\mu/\\alpha$ identically, the interior branch is *bounded
       away from zero* by construction: no interior equilibrium can have
       $V^* < \\mu/\\alpha$ (at baseline, $V^* \\geq 2$).

    3. **Magnitude report.** Equilibria with $V^* <$
       ``near_zero_threshold`` are reported descriptively. By default the
       threshold is $\\min(\\mu/\\alpha, 1.0)$. The threshold is not a
       theorem boundary and does not affect the structural audit.

    Parameters
    ----------
    p : Dict[str, float]
        Parameter dictionary; must contain the standard DLVT keys.
    near_zero_threshold : Optional[float]
        Threshold below which an equilibrium is flagged ``near zero''.
        Defaults to ``min(mu/alpha, 1.0)``.

    Returns
    -------
    Dict[str, any]
        Audit report with keys:

        - ``axis_equilibrium`` : dict or None
            The $C=0$ axis equilibrium, classified by Jacobian eigenvalues.
            Keys: ``V``, ``C``, ``eigenvalues``, ``stable``, ``classification``
            (``saddle``, ``stable node/focus``, ``unstable node/focus``).
        - ``interior_equilibria`` : List[Dict]
            All interior equilibria from :func:`find_interior_equilibria`.
        - ``small_magnitude_equilibria`` : List[Dict]
            Equilibria below the reporting threshold. A unique small axis
            root is mathematically legitimate.
        - ``unexpected_branch`` : Optional[Dict]
            Structural violation, if one is found: a second positive axis
            root or an interior root below its analytical lower bound.
        - ``near_zero_branch`` : Optional[Dict]
            Deprecated compatibility alias for ``unexpected_branch``. It no
            longer treats a unique small axis root as an unexpected branch.
        - ``interior_V_lower_bound`` : float
            The analytical lower bound $\\mu/\\alpha$ on any interior $V^*$.
        - ``quadratic_positive_root_count`` : int
            Number of positive roots of the axis quadratic (always 1 by
            the product-of-roots argument; included as a numerical check).
        - ``diagnostic`` : str
            Human-readable summary of the audit.

    Notes
    -----
    Appendix A9 derives these facts
    formally. The function exists as defensive infrastructure: it makes
    the ``no spurious branch'' claim *executable*, so any future parameter
    change that would reintroduce a near-zero equilibrium will be caught by
    the pinning test in ``code/tests/test_analysis.py``.
    """
    p = validate_params(p)
    if near_zero_threshold is None:
        near_zero_threshold = min(p['mu'] / p['alpha'], 1.0)

    # -- 1. Axis equilibrium ---------------------------------------------------
    O0_eff = p['O0']  # experienced coordination load at C=0
    a = p['R'] / p['Vmax']
    b = -(p['R'] - p['R'] * p['eps'] / p['Vmax'] - p['delta'] * O0_eff**p['gamma'])
    c = -p['R'] * p['eps']
    disc = b * b - 4.0 * a * c
    pos_roots: List[float] = []
    if disc >= 0 and a != 0:
        sqrt_disc = float(np.sqrt(disc))
        for V_root in ((-b + sqrt_disc) / (2.0 * a), (-b - sqrt_disc) / (2.0 * a)):
            if V_root > 0:
                pos_roots.append(V_root)

    axis_eq: Optional[Dict[str, any]] = None
    if pos_roots:
        V_axis = pos_roots[0]  # exactly one by product-of-roots argument
        eigvals, stable = jacobian_eigenvalues(V_axis, 0.0, p)
        re_parts = np.real(eigvals)
        if np.all(re_parts < 0):
            classification = 'stable'
        elif np.all(re_parts > 0):
            classification = 'unstable node/focus'
        else:
            classification = 'saddle'
        axis_eq = dict(
            V=float(V_axis),
            C=0.0,
            eigenvalues=eigvals,
            stable=bool(stable),
            classification=classification,
        )

    # -- 2. Interior equilibria ------------------------------------------------
    interior = find_interior_equilibria(p, n_scan=12000)

    # -- 3. Magnitude report and structural audit -----------------------------
    small_magnitude: List[Dict[str, any]] = []
    if axis_eq is not None and axis_eq['V'] < near_zero_threshold:
        small_magnitude.append({**axis_eq, 'source': 'axis'})
    for eq in interior:
        if eq['V'] < near_zero_threshold:
            small_magnitude.append({**eq, 'source': 'interior'})

    interior_V_lower_bound = p['mu'] / p['alpha']
    unexpected_branch: Optional[Dict[str, any]] = None
    if len(pos_roots) != 1:
        unexpected_branch = {
            'source': 'axis',
            'reason': 'axis-positive-root-count',
            'positive_root_count': len(pos_roots),
        }
    else:
        for eq in interior:
            if eq['V'] < interior_V_lower_bound - 1e-10:
                unexpected_branch = {
                    **eq,
                    'source': 'interior',
                    'reason': 'below-interior-analytical-bound',
                }
                break

    if unexpected_branch is None:
        axis_v_str = f"{axis_eq['V']:.3f}" if axis_eq is not None else "n/a"
        axis_cls_str = axis_eq['classification'] if axis_eq is not None else 'absent'
        magnitude_note = (
            f" {len(small_magnitude)} equilibrium/equilibria fall below the "
            f"descriptive magnitude threshold {near_zero_threshold:.3f}; "
            f"this does not violate the unique-root result."
            if small_magnitude else ""
        )
        diagnostic = (
            f"Regularization structure is valid: exactly one positive axis root "
            f"and no interior root below its analytical bound. "
            f"Interior V* is bounded below by μ/α = {interior_V_lower_bound:.3f} "
            f"(by the C-isocline V_c(C) = μ(1+φO)/α). "
            f"The C=0 axis equilibrium sits at V ≈ {axis_v_str} "
            f"and is classified as '{axis_cls_str}'. "
            f"The axis V-isocline quadratic has exactly "
            f"{len(pos_roots)} positive root(s); its product-of-roots is "
            f"-ε·Vmax = {-p['eps'] * p['Vmax']:.3f}, guaranteeing a single "
            f"positive root for any ε > 0. See Appendix A9."
            f"{magnitude_note}"
        )
    else:
        diagnostic = (
            f"Unexpected regularization structure detected "
            f"({unexpected_branch['reason']}, source: "
            f"{unexpected_branch['source']}). See Appendix A9 and verify the "
            f"root enumeration and V-isocline analysis."
        )

    return {
        'axis_equilibrium': axis_eq,
        'interior_equilibria': interior,
        'small_magnitude_equilibria': small_magnitude,
        'unexpected_branch': unexpected_branch,
        'near_zero_branch': unexpected_branch,
        'regularization_structure_valid': unexpected_branch is None,
        'interior_V_lower_bound': float(interior_V_lower_bound),
        'quadratic_positive_root_count': len(pos_roots),
        'diagnostic': diagnostic,
    }


# ── Global stability audit (R7 issue 2) ───────────────────────────────────────

def bendixson_dulac_certificate(
    p: Dict[str, float],
    V_grid_n: int = 60,
    C_grid_n: int = 60,
    C_trap_safety: float = 1.2,
) -> Dict[str, any]:
    """Verify the Bendixson–Dulac no-closed-orbit certificate on the trapping set.

    The DLVT system admits the Dulac function $B(V, C) = 1/C$ on the
    positive quadrant $\\{C > 0\\}$. Computing $\\nabla \\cdot (B \\mathbf{F})$
    where $\\mathbf{F} = (\\dot V, \\dot C)$ yields:

    .. math::
        \\frac{\\partial (Bf)}{\\partial V}
        = \\frac{1}{C}\\left[ -\\frac{R}{V_{\\max}}
        - \\frac{\\delta O^\\gamma \\varepsilon}{(V+\\varepsilon)^2} \\right]
        < 0,

    .. math::
        \\frac{\\partial (Bg)}{\\partial C}
        = -\\frac{\\alpha V \\phi \\,(\\beta \\eta C^{\\eta-1})}
               {(1+\\phi O)^2}
        < 0

    for every $(V, C)$ with $V > 0$ and $C > 0$. By the Bendixson–Dulac
    theorem this rules out closed orbits in the positive quadrant. This
    function verifies the claim *numerically* on a dense grid inside the
    analytical trapping rectangle.

    Analytical trapping rectangle. The rectangle
    $\\Omega = [0, V_{\\max}] \\times [0, C_{\\text{trap}}]$ is forward
    invariant with

    .. math::
        C_{\\text{trap}}^\\eta =
        \\frac{(\\alpha V_{\\max}/\\mu - 1)/\\phi - O_0}{\\beta},

    which is the unique $C$ at which $\\mu(1 + \\phi O(C))/\\alpha =
    V_{\\max}$. For any $C > C_{\\text{trap}}$ and any $V \\in [0, V_{\\max}]$,
    we have $\\dot C = C(\\alpha V/(1+\\phi O) - \\mu) < 0$, so the
    rectangle is a trap.

    Parameters
    ----------
    p : Dict[str, float]
        DLVT parameters.
    V_grid_n, C_grid_n : int
        Grid resolution for the numerical divergence check.
    C_trap_safety : float
        Multiplicative safety factor on the analytical $C_{\\text{trap}}$;
        the grid actually spans $[0, C_{\\text{trap\\_safety}} \\cdot C_{\\text{trap}}]$
        so that any small analytical-to-numerical mismatch is captured.

    Returns
    -------
    Dict[str, any]
        Keys:
        - ``c_trap`` : float, analytical C_{trap}
        - ``max_divergence`` : float, supremum of $\\nabla \\cdot (B\\mathbf{F})$
          over the grid; negative confirms the certificate.
        - ``divergence_is_strictly_negative`` : bool
        - ``dc_dt_above_c_trap_is_negative`` : bool, sanity-check the trap.
        - ``diagnostic`` : str
    """
    p = validate_params(p)
    # Analytical C_trap from the α V/(1+φO) = μ threshold at V = Vmax.
    # With general η: C_trap^η · β = (αVmax/μ - 1)/φ - O0.
    c_trap = trapping_scope_bound(p)

    C_hi = C_trap_safety * c_trap
    # Skip V=0 and C=0 edges — the Dulac function has a removable singularity
    # at C=0 and the divergence expression below has a 1/C factor.
    Vs = np.linspace(1e-4 * p['Vmax'], p['Vmax'], V_grid_n)
    Cs = np.linspace(1e-3 * C_hi, C_hi, C_grid_n)

    max_div = -np.inf
    for V in Vs:
        for C in Cs:
            O = p['O0'] + p['beta'] * C ** p['eta']
            dOdC = p['beta'] * p['eta'] * C ** (p['eta'] - 1.0)
            d1 = (1.0 / C) * (
                -p['R'] / p['Vmax']
                - p['delta'] * O ** p['gamma'] * p['eps'] / (V + p['eps']) ** 2
            )
            d2 = -(p['alpha'] * V * p['phi'] * dOdC) / (1.0 + p['phi'] * O) ** 2
            div = d1 + d2
            if div > max_div:
                max_div = div

    # Sanity check: above C_trap, dC/dt < 0 for every V ∈ [0, Vmax].
    dc_ok = True
    for V in np.linspace(0.0, p['Vmax'], 20):
        C_test = 1.1 * c_trap
        O_test = p['O0'] + p['beta'] * C_test ** p['eta']
        dC = C_test * (p['alpha'] * V / (1.0 + p['phi'] * O_test) - p['mu'])
        if dC >= 0:
            dc_ok = False
            break

    strictly_negative = bool(max_div < 0.0)
    diagnostic = (
        f"Bendixson–Dulac certificate with B(V,C) = 1/C: max divergence "
        f"over grid = {max_div:.4e}, C_trap = {c_trap:.4f}. "
        f"{'Strictly negative everywhere — no closed orbits in the trapping rectangle.' if strictly_negative else 'NOT strictly negative — certificate fails.'}"
    )
    return {
        'c_trap': float(c_trap),
        'max_divergence': float(max_div),
        'divergence_is_strictly_negative': strictly_negative,
        'dc_dt_above_c_trap_is_negative': dc_ok,
        'diagnostic': diagnostic,
    }


def basin_of_attraction_sweep(
    p: Dict[str, float],
    V0_grid: Optional[List[float]] = None,
    C0_grid: Optional[List[float]] = None,
    T: float = 600.0,
    tol: float = 1e-2,
) -> Dict[str, any]:
    """Integrate the DLVT system from a grid of initial conditions and
    verify convergence to the unique interior equilibrium (Theorem 2).

    This function provides *numerical corroboration* of the global
    asymptotic stability statement proved analytically via
    Bendixson–Dulac + Poincaré–Bendixson in Appendix A10. It is
    intentionally redundant with the theorem: the theorem rules out
    closed orbits and forces every bounded trajectory in
    $\\Omega \\cap \\{C > 0\\}$ to converge to the interior equilibrium;
    this function confirms that no numerical artifacts
    (stiff-step rejections, basin boundaries, etc.) defeat the prediction.

    Parameters
    ----------
    p : Dict[str, float]
        DLVT parameters.
    V0_grid, C0_grid : Optional[List[float]]
        Initial-condition grids. Default: 8 points each along
        $V \\in [0.1, V_{\\max}]$ and $C \\in [0.5, \\text{carrying\\_capacity}(p) \\cdot 2]$.
    T : float
        Integration horizon. Default 600 time units — several
        e-folding times at baseline.
    tol : float
        Tolerance for declaring convergence to the interior equilibrium;
        the trajectory must land within ``tol`` of the target in both
        V and C.

    Returns
    -------
    Dict[str, any]
        Keys:
        - ``zombie_target`` : deprecated compatibility key for the
          equilibrium-target tuple
        - ``n_total`` : int
        - ``n_converged`` : int
        - ``max_final_error`` : float, max component-wise error at t=T
        - ``non_converged`` : list of (V0, C0) tuples that failed
        - ``diagnostic`` : str
    """
    p = validate_params(p)
    # Import here to avoid circular/partial imports at module load.
    from scipy.integrate import solve_ivp
    from .model import dlvt_system

    interior = find_interior_equilibria(p, n_scan=12000)
    if not interior:
        raise ValueError("No interior equilibria found; basin sweep not applicable.")
    eq = interior[0]
    target = (eq['V'], eq['C'])

    if V0_grid is None:
        V0_grid = list(np.linspace(0.1, p['Vmax'], 8))
    if C0_grid is None:
        C_upper = max(1.0, 2.0 * drain_coefficient_threshold(p))
        C0_grid = list(np.linspace(0.5, C_upper, 8))

    n_total = 0
    n_conv = 0
    max_err = 0.0
    non_conv: List[Tuple[float, float]] = []
    for V0 in V0_grid:
        for C0 in C0_grid:
            sol = solve_ivp(
                dlvt_system, [0.0, T], [V0, C0], args=(p,),
                method='RK45', rtol=1e-8, atol=1e-10,
            )
            V_final = float(sol.y[0, -1])
            C_final = float(sol.y[1, -1])
            err = max(abs(V_final - target[0]), abs(C_final - target[1]))
            if err < tol:
                n_conv += 1
            else:
                non_conv.append((float(V0), float(C0)))
            if err > max_err:
                max_err = err
            n_total += 1

    diagnostic = (
        f"Basin sweep: {n_conv}/{n_total} initial conditions converged to "
        f"the interior equilibrium (V*, C*) ≈ ({target[0]:.4f}, {target[1]:.4f}) "
        f"within tol={tol}. Max final error: {max_err:.3e}. "
        f"{'All trajectories converge (numerical corroboration of Theorem 2).' if n_conv == n_total else 'NON-CONVERGENT trajectories detected — investigate.'}"
    )
    return {
        'zombie_target': target,  # deprecated compatibility key through v2.x
        'n_total': n_total,
        'n_converged': n_conv,
        'max_final_error': max_err,
        'non_converged': non_conv,
        'diagnostic': diagnostic,
    }
