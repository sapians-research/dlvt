"""
Tests for dlvt.sobol: formal variance-based (Sobol') sensitivity indices.

The correctness oracle is the EXACT structural knowledge of the interior
equilibrium (see the dlvt.nondimensional module header and Lemma 2 /
scope-absorption):

  * V* is exactly invariant to beta, eta, O0 -> their total-effect Sobol
    indices must be ~0 (in fact identically 0 on every retained row, because
    replacing column i by B leaves V* unchanged when i is beta/eta/O0);
  * V* depends on (mu, alpha) only through mu/alpha and on (R, delta) only
    through R/delta -> mu, alpha (and R, delta) carry substantial total effect.

All thresholds below were CALIBRATED by running the estimator at the tested
configuration (n_base=128, seed=1) and left with margin.  In particular the
"ST >= S1" sanity allowance is loosened to 0.12 because the Jansen (1999)
first-order estimator has a Monte-Carlo noise floor of ~0.09 on the
exactly-zero parameters at N=128 (measured min(ST - S1) = -0.090); this is
estimator noise on truly-zero indices, not a structural violation.

Runtime: three n_base=128 evaluations (~4s each) -> well under the ~90s budget.
"""

import numpy as np
import pytest

from dlvt.model import make_params
from dlvt.sobol import sobol_indices

N_BASE = 128
SEED = 1
PARAMS = ['R', 'Vmax', 'delta', 'gamma', 'O0', 'beta', 'eta', 'alpha', 'phi',
          'mu', 'eps']


@pytest.fixture(scope='module')
def result():
    """One Sobol' analysis of V*, reused across the read-only assertions."""
    return sobol_indices(make_params(), n_base=N_BASE, seed=SEED,
                         factor=1.5, output='V_star')


def _idx(result, name):
    return result['params'].index(name)


# ---------------------------------------------------------------------------
# Structural zeros (the validation anchor): beta, eta, O0 -> ST ~ 0
# ---------------------------------------------------------------------------

def test_structural_zeros_have_zero_total_effect(result):
    """beta, eta, O0 move only C*, never V*, so their total-effect index is ~0.
    Replacing their column by B leaves V* unchanged, so ST is identically 0
    up to floating-point noise (measured ~1e-28)."""
    ST = result['ST']
    for name in ('beta', 'eta', 'O0'):
        assert ST[_idx(result, name)] < 0.02, (
            f"{name}: ST={ST[_idx(result, name)]:.4g} should be ~0 (structural)"
        )


# ---------------------------------------------------------------------------
# Dominant ratio parameters: mu, alpha carry substantial total effect
# ---------------------------------------------------------------------------

def test_ratio_parameters_are_influential(result):
    """V* depends on (mu, alpha) through mu/alpha, so both carry real variance.
    Measured at n_base=128, seed=1: ST(mu)~0.246, ST(alpha)~0.225."""
    ST = result['ST']
    assert ST[_idx(result, 'mu')] > 0.1
    assert ST[_idx(result, 'alpha')] > 0.1


# ---------------------------------------------------------------------------
# Estimator sanity
# ---------------------------------------------------------------------------

def test_total_effect_dominates_first_order(result):
    """ST >= S1 for true indices; allow a 0.12 Monte-Carlo slack because the
    Jansen S1 estimator has a ~0.09 noise floor on the exactly-zero parameters
    at N=128 (measured min(ST - S1) = -0.090)."""
    S1, ST = result['S1'], result['ST']
    assert np.all(ST >= S1 - 0.12), (
        f"min(ST - S1) = {np.nanmin(ST - S1):.4f} below -0.12"
    )


def test_indices_within_plausible_bounds(result):
    """Sobol' indices live in [0, 1]; allow [-0.05, 1.1] for estimator noise."""
    S1, ST = result['S1'], result['ST']
    assert np.all(np.isfinite(S1)) and np.all(np.isfinite(ST))
    assert S1.min() >= -0.05 and S1.max() <= 1.1
    assert ST.min() >= -0.05 and ST.max() <= 1.1


def test_output_bookkeeping(result):
    """Shape / key contract of the returned dictionary."""
    assert result['params'] == PARAMS
    assert result['S1'].shape == (11,)
    assert result['ST'].shape == (11,)
    assert result['n_base'] == N_BASE
    assert result['output'] == 'V_star'
    assert result['var_total'] > 0.0
    assert result['formal_decomposition'] is True
    assert 'fully defined' in result['method']


# ---------------------------------------------------------------------------
# NaN policy: enough draws resolve to a stable equilibrium
# ---------------------------------------------------------------------------

def test_retained_fraction_high(result):
    """A formal decomposition requires a defined output over the full design."""
    assert result['retained_fraction'] == 1.0
    assert result['complete_row_fraction'] == 1.0


def test_missing_outputs_raise_by_default():
    """R8 CODE-006: silently filtering a parameter-dependent subset is invalid."""
    with pytest.raises(ValueError, match='undefined for part'):
        sobol_indices(
            make_params(), n_base=N_BASE, seed=SEED, factor=2.0,
            output='V_star',
        )


def test_conditional_missing_policy_is_explicitly_exploratory():
    """The opt-in complete-row result must not call itself formal."""
    r = sobol_indices(
        make_params(), n_base=N_BASE, seed=SEED, factor=2.0,
        output='V_star', missing_policy='conditional',
    )
    assert r['retained_fraction'] < 1.0
    assert r['complete_row_fraction'] < 1.0
    assert r['formal_decomposition'] is False
    assert 'exploratory conditional' in r['method']


# ---------------------------------------------------------------------------
# Determinism: same seed -> identical arrays
# ---------------------------------------------------------------------------

def test_determinism_same_seed():
    """Scrambled Sobol' with a fixed seed is deterministic, so repeated calls
    return bitwise-identical S1/ST arrays."""
    r1 = sobol_indices(make_params(), n_base=N_BASE, seed=SEED, factor=1.5)
    r2 = sobol_indices(make_params(), n_base=N_BASE, seed=SEED, factor=1.5)
    assert np.array_equal(r1['S1'], r2['S1'])
    assert np.array_equal(r1['ST'], r2['ST'])
    assert r1['retained_fraction'] == r2['retained_fraction']
