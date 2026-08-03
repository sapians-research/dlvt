"""
dlvt.sobol
==========
Formal variance-based (Sobol') global sensitivity analysis for the DLVT model.

This module upgrades the DLVT sensitivity toolkit from the rank-correlation
*screening* provided by the deprecated compatibility function
:func:`dlvt.nondimensional.lhs_zombie_fraction` (Latin-hypercube + Spearman)
to a FORMAL variance decomposition of the
equilibrium output V* over the 11 raw model parameters.  Where the LHS screen
reports monotone marginal association (Spearman rho), the Sobol' indices here
apportion the *variance* of the output to each input and to its total
interaction burden — the quantities a reviewer means by "how much does V*
depend on parameter p_i".

Estimator (Jansen 1999)
-----------------------
For each parameter i we report two indices:

* First-order index  S1_i = Var_{p_i}[ E(V* | p_i) ] / Var(V*)
      — the fraction of output variance explained by p_i *alone*.
* Total-effect index  ST_i = E[ Var(V* | p_{~i}) ] / Var(V*)
      — the fraction of variance involving p_i through any interaction.

They are estimated with the Jansen (1999) Monte-Carlo estimators, which are
the numerically well-behaved members of the Saltelli family:

    ST_i = mean( ( f(A) - f(AB_i) )^2 ) / ( 2 * Var(f(A U B)) )
    S1_i = ( Var(f(A U B)) - mean( ( f(B) - f(AB_i) )^2 ) / 2 ) / Var(f(A U B))

with A, B the two independent base sample matrices and AB_i the matrix equal to
A except that its i-th column is taken from B.  By construction ST_i >= S1_i
(interactions are non-negative); small negative estimates are Monte-Carlo noise.

Sampling (Saltelli / scrambled Sobol' QMC)
------------------------------------------
The two base matrices A and B (each ``n_base x 11``) are drawn as the two
halves of a single scrambled Sobol' low-discrepancy sequence of dimension
``2 * 11`` (``scipy.stats.qmc.Sobol(scramble=True, seed=...)``); this is the
standard Saltelli construction and gives A, B that are jointly low-discrepancy
(hence "independent" in the QMC sense).  Each unit-cube coordinate ``u`` is
mapped to the parameter axis LOG-UNIFORMLY on ``[x_i / factor, x_i * factor]``
around the baseline ``x_i`` via ``x_i * factor**(2u - 1)`` — matching the
convention used throughout :mod:`dlvt.nondimensional`.  The total number of
model evaluations is ``n_base * (2 + 11)``.  For the balance properties of the
Sobol' net to hold, ``n_base`` should be a power of two (256, 128, ...).

Model output
------------
The output ``f`` is the equilibrium vitality V* at the lowest-C stable interior
equilibrium, obtained from :func:`dlvt.analysis.find_interior_equilibria` with
its default (adaptive, beta-aware) ``C_max`` and ``n_scan=2000``.  With
``output='low_vitality'`` the output is instead the illustrative indicator
``1[V* < 0.5 * Vmax]``. The threshold is not empirically validated.

Missing-equilibrium policy
--------------------------
A Sobol' decomposition is formally defined only when the model output exists
over the entire declared input distribution.  The default policy therefore
raises if any Saltelli evaluation lacks a stable interior equilibrium. Users
must either narrow the input domain or define an output for boundary cases.
An explicit ``missing_policy='conditional'`` option retains only rows
that are complete across A, B, and every AB_i matrix; its results are labeled
exploratory because selection changes the input distribution.

Validation anchor (exact structural zeros)
------------------------------------------
The interior equilibrium (V*, O*) solves a system involving neither beta, eta,
nor O0 (see the :mod:`dlvt.nondimensional` module header and the
scope-absorption result): V* is invariant to those three parameters while the
same positive interior root remains admissible (in particular, ``O0 < O*``).
Consequently their total-effect indices must be ~0 (up to Monte-Carlo noise) —
in fact replacing column i by B leaves V* unchanged when i is beta/eta/O0, so
``f(A) == f(AB_i)`` and ``ST_i`` is identically 0 on every retained row.
Conversely V* depends on (mu, alpha) only through mu/alpha and on (R, delta)
only through R/delta, so mu, alpha, R, delta must carry substantial total
effect.  These facts are the built-in correctness oracle for this estimator
(see tests/test_sobol.py).

Contrast with the LHS/Spearman screen
--------------------------------------
:func:`dlvt.nondimensional.lhs_zombie_fraction` is a deprecated compatibility
wrapper that answers "which parameters are
monotonically associated with V*"; this module answers "how is the *variance*
of V* partitioned among the parameters and their interactions".  The two are
complementary; only the latter is a formal variance-based decomposition.

References
----------
  Jansen, M.J.W. (1999). Analysis of variance designs for model output.
    Computer Physics Communications 117(1-2), 35-43.
  Saltelli, A. et al. (2010). Variance based sensitivity analysis of model
    output. Design and estimator for the total sensitivity index.
    Computer Physics Communications 181(2), 259-270.
  Bendinelli, W. (2026). Dynamic Leadership Vitality Theory: A Formal Model
    manuscript in preparation.
"""

from typing import Dict, List, Optional
import warnings

import numpy as np
from scipy.stats import qmc

from .analysis import DISPLAY_THRESHOLD_FRACTION, find_interior_equilibria
from .model import make_params
from .nondimensional import PARAM_NAMES

# ``make_params`` is re-exported so that downstream sensitivity scripts (e.g.
# scripts/fig13_sobol_indices.py) can obtain the baseline parameter dict while
# depending only on :mod:`dlvt.sobol` for their DLVT imports.
__all__ = ['sobol_indices', 'make_params']


# -- Model output ---------------------------------------------------------------

def _v_star(p: Dict[str, float], n_scan: int = 2000) -> float:
    """V* at the lowest-C stable interior equilibrium, or NaN if none exists.

    Uses :func:`dlvt.analysis.find_interior_equilibria` with the default
    adaptive (beta-aware) ``C_max`` and the requested ``n_scan``.  Returns
    ``np.nan`` when no stable interior equilibrium is found (the NaN policy
    documented in the module header then applies).
    """
    eqs = find_interior_equilibria(p, n_scan=n_scan)
    stable = [eq for eq in eqs if eq['stable']]
    if not stable:
        return float('nan')
    stable.sort(key=lambda eq: eq['C'])
    return float(stable[0]['V'])


def _model_output(p: Dict[str, float], output: str, n_scan: int) -> float:
    """Scalar model output for one parameter draw.

    ``output='V_star'`` returns V* (NaN if no stable equilibrium);
    ``output='low_vitality'`` returns 1[V* < 0.5*Vmax]
    (NaN if no stable equilibrium — the missing draw is not silently assigned
    a compatibility label). ``output='regime'`` is a deprecated input alias
    for ``'low_vitality'``.
    """
    v = _v_star(p, n_scan=n_scan)
    if not np.isfinite(v):
        return float('nan')
    if output == 'V_star':
        return v
    if output in ('low_vitality', 'regime'):
        return 1.0 if v < DISPLAY_THRESHOLD_FRACTION * p['Vmax'] else 0.0
    raise ValueError(
        f"unknown output={output!r}; expected 'V_star' or 'low_vitality'."
    )


def _row_to_params(base: Dict[str, float], unit_row: np.ndarray,
                   factor: float) -> Dict[str, float]:
    """Map a unit-cube row to a parameter dict, log-uniform on [x/factor, x*factor].

    The map ``x_i * factor**(2u - 1)`` sends u=0 -> x_i/factor, u=1 -> x_i*factor,
    and u=0.5 -> x_i, matching :func:`dlvt.nondimensional.lhs_zombie_fraction`.
    """
    p = dict(base)
    for j, name in enumerate(PARAM_NAMES):
        p[name] = base[name] * factor ** (2.0 * unit_row[j] - 1.0)
    return p


# -- Sobol' indices -------------------------------------------------------------

def sobol_indices(p: Dict[str, float], n_base: int = 256, factor: float = 1.5,
                  seed: int = 1, output: str = 'V_star',
                  n_scan: int = 2000,
                  missing_policy: str = 'raise') -> Dict[str, object]:
    """Variance-based Sobol' first-order (S1) and total-effect (ST) indices.

    Formal replacement for LHS+Spearman screening: this apportions the variance
    of the model output among the 11 raw DLVT parameters using the Jansen (1999)
    Monte-Carlo estimators on a Saltelli design built from a single scrambled
    Sobol' low-discrepancy sequence.  See the module docstring for the full
    method statement (estimator, sampling, NaN policy, validation anchor).

    Estimators (per parameter i), with A, B the two independent base matrices,
    AB_i = A with column i replaced by B's column i, and Var over the finite
    pooled f(A U B):

        ST_i = mean( ( f(A) - f(AB_i) )^2 ) / ( 2 * Var )
        S1_i = ( Var - mean( ( f(B) - f(AB_i) )^2 ) / 2 ) / Var

    Missing outputs raise by default.  With ``missing_policy='conditional'``,
    only rows complete across A, B, and all AB_i matrices are retained and the
    result is explicitly marked as an exploratory conditional decomposition.

    Validation anchor: beta, eta, O0 have EXACTLY zero effect on V* (they move
    only C*), so their ST must be ~0; V* depends on (mu, alpha) via mu/alpha and
    on (R, delta) via R/delta, so mu, alpha, R, delta must carry substantial ST.

    Parameters
    ----------
    p : Dict[str, float]
        Baseline parameter dictionary (centre of the log-uniform hypercube).
    n_base : int, optional
        Base sample size N.  Total model evaluations = N*(2+11).  Should be a
        power of two for the Sobol' net's balance properties.  Default 256.
    factor : float, optional
        Half-width of the log-uniform range; each parameter spans
        [x_i/factor, x_i*factor].  Default 1.5; this is a numerical domain,
        not an empirical plausibility claim.
    seed : int, optional
        Seed for the scrambled Sobol' sampler (deterministic).  Default 1.
    output : str, optional
        ``'V_star'`` (equilibrium vitality) or ``'low_vitality'``
        (illustrative indicator ``1[V* < 0.5*Vmax]``). ``'regime'`` remains
        accepted only as a deprecated compatibility alias. Default ``'V_star'``.
    n_scan : int, optional
        Deprecated compatibility argument passed to
        :func:`find_interior_equilibria`; the monotone root solver ignores it.
    missing_policy : {'raise', 'conditional'}, optional
        ``'raise'`` requires a defined output at every design evaluation.
        ``'conditional'`` performs an exploratory complete-row analysis.

    Returns
    -------
    Dict[str, object]
        Keys:
        - 'params'            : list[str], the 11 parameter names (canonical order)
        - 'S1'                : ndarray (11,), first-order indices
        - 'ST'                : ndarray (11,), total-effect indices
        - 'retained_fraction' : float, finite share of all N*(2+11) evaluations
        - 'n_base'            : int, the base sample size N used
        - 'var_total'         : float, Var over the finite pooled f(A U B)
        - 'output'            : str, the output analysed
        - 'formal_decomposition': bool, true only with no missing outputs
        - 'method'            : explicit interpretation label
    """
    if output not in ('V_star', 'low_vitality', 'regime'):
        raise ValueError(
            f"unknown output={output!r}; expected 'V_star' or 'low_vitality'."
        )
    if output == 'regime':
        warnings.warn(
            "output='regime' is deprecated; use output='low_vitality' and "
            "report the explicit display threshold.",
            DeprecationWarning,
            stacklevel=2,
        )
        output = 'low_vitality'
    if missing_policy not in ('raise', 'conditional'):
        raise ValueError(
            "missing_policy must be either 'raise' or 'conditional'."
        )
    d = len(PARAM_NAMES)

    # -- Saltelli design: two halves of one scrambled Sobol' sequence ----------
    sampler = qmc.Sobol(d=2 * d, scramble=True, seed=seed)
    sample = sampler.random(n_base)          # (n_base, 2d), jointly low-discrepancy
    A_unit = sample[:, :d]
    B_unit = sample[:, d:]

    def _eval_matrix(unit_mat: np.ndarray) -> np.ndarray:
        out = np.empty(unit_mat.shape[0], dtype=float)
        for k in range(unit_mat.shape[0]):
            out[k] = _model_output(
                _row_to_params(p, unit_mat[k], factor), output, n_scan
            )
        return out

    fA = _eval_matrix(A_unit)
    fB = _eval_matrix(B_unit)

    # AB_i: A with column i replaced by B's column i.
    fAB = np.empty((d, n_base), dtype=float)
    for i in range(d):
        AB_unit = A_unit.copy()
        AB_unit[:, i] = B_unit[:, i]
        fAB[i] = _eval_matrix(AB_unit)

    all_out = np.concatenate([fA, fB, fAB.ravel()])
    retained_fraction = float(np.mean(np.isfinite(all_out)))
    complete_rows = (
        np.isfinite(fA)
        & np.isfinite(fB)
        & np.all(np.isfinite(fAB), axis=0)
    )

    if retained_fraction < 1.0 and missing_policy == 'raise':
        raise ValueError(
            "Sobol output is undefined for part of the declared parameter "
            f"domain (finite evaluation fraction={retained_fraction:.4f}). "
            "Narrow factor/ranges, define an output for boundary cases, or use "
            "missing_policy='conditional' for an explicitly exploratory "
            "complete-row analysis."
        )

    if missing_policy == 'conditional':
        fA_use = fA[complete_rows]
        fB_use = fB[complete_rows]
        fAB_use = fAB[:, complete_rows]
    else:
        fA_use = fA
        fB_use = fB
        fAB_use = fAB

    pooled = np.concatenate([fA_use, fB_use])
    var_total = float(np.var(pooled, ddof=1)) if pooled.size > 1 else float('nan')

    S1 = np.full(d, np.nan)
    ST = np.full(d, np.nan)
    if np.isfinite(var_total) and var_total > 0.0:
        for i in range(d):
            if fA_use.size:
                dT = fA_use - fAB_use[i]
                ST[i] = float(np.mean(dT ** 2) / (2.0 * var_total))
            if fB_use.size:
                d1 = fB_use - fAB_use[i]
                S1[i] = float(
                    (var_total - np.mean(d1 ** 2) / 2.0) / var_total
                )

    formal_decomposition = retained_fraction == 1.0
    method = (
        'Sobol-Jansen variance decomposition on a fully defined domain'
        if formal_decomposition
        else 'exploratory conditional complete-row Sobol-Jansen indices'
    )

    return {
        'params': list(PARAM_NAMES),
        'S1': S1,
        'ST': ST,
        'retained_fraction': retained_fraction,
        'n_base': int(n_base),
        'var_total': var_total,
        'output': output,
        'complete_row_fraction': float(np.mean(complete_rows)),
        'formal_decomposition': formal_decomposition,
        'method': method,
    }
