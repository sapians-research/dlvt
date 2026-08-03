"""
Tests for dlvt.nondimensional: reduced-form round-trip, exact structural
degeneracies, elasticities, the critical mu/alpha ratio, and the LHS screen.

All checks are INDEPENDENT of any stored solver output: the round-trip test
compares two integrations of mathematically equivalent systems, the
degeneracy tests assert exact analytical invariances, and the remaining
tests pin to hand-derivable structural facts (zero elasticities, the
strategic-threshold crossing, a loose genericity bracket for the LHS).
"""

import numpy as np
import pytest

from dlvt.model import make_params, simulate
from dlvt.analysis import find_interior_equilibria
from dlvt.nondimensional import (
    classify_point,
    lhs_zombie_fraction,
    mu_alpha_critical,
    reduced_groups,
    simulate_reduced,
    v_star_elasticities,
    zombie_boundary_map,
    zombie_boundary_map_beta,
)


def _lowest_c_stable(p, C_max=300.0):
    """Lowest-C stable interior equilibrium, computed directly from
    dlvt.analysis (independent of dlvt.nondimensional helpers)."""
    eqs = find_interior_equilibria(p, C_max=C_max)
    stable = [eq for eq in eqs if eq['stable']]
    assert stable, "expected a stable interior equilibrium"
    stable.sort(key=lambda eq: eq['C'])
    return stable[0]


# ---------------------------------------------------------------------------
# Reduced-form round trip (the correctness oracle for the derivation)
# ---------------------------------------------------------------------------

def test_reduced_groups_baseline_values():
    """Groups at baseline are hand-computable: rho = 3/(0.2*10) = 1.5, etc."""
    g = reduced_groups(make_params())
    assert g['rho'] == pytest.approx(1.5, rel=1e-12)
    assert g['kappa'] == pytest.approx(0.01, rel=1e-12)
    assert g['a'] == pytest.approx(5.0, rel=1e-12)
    assert g['f'] == pytest.approx(0.15, rel=1e-12)
    assert g['e'] == pytest.approx(0.01, rel=1e-12)
    assert g['gamma'] == pytest.approx(2.0, rel=1e-12)
    assert g['eta'] == pytest.approx(1.0, rel=1e-12)


def test_reduced_groups_include_eta_as_seventh_group():
    """R8 M15: the autonomous reduced form closes for every eta>0."""
    g = reduced_groups(make_params(eta=1.5))
    assert g['eta'] == pytest.approx(1.5, rel=1e-12)


def test_reduced_roundtrip_matches_dimensional_simulation():
    """Simulating the reduced system and mapping back with V = Vmax*v,
    C = O0*w/beta must reproduce the dimensional simulation.

    This is an independent oracle: dlvt.model.simulate integrates the raw
    dimensional ODEs, simulate_reduced integrates the derived dimensionless
    system; agreement can only occur if the derivation and group definitions
    are correct (a wrong group produces O(1) discrepancies).
    """
    p = make_params()
    V0, C0, T = 8.0, 0.5, 120.0

    t_dim, V_dim, C_dim, _, _, _ = simulate(p, V0=V0, C0=C0, T=T)
    _, V_red, C_red = simulate_reduced(p, V0=V0, C0=C0, T=T, t_eval=t_dim)

    assert np.max(np.abs(V_red - V_dim)) < 1e-6
    assert np.max(np.abs(C_red - C_dim)) < 1e-5


def test_reduced_roundtrip_off_baseline():
    """Round trip also holds away from baseline (different groups)."""
    p = make_params(R=4.0, delta=0.03, mu=0.3, alpha=0.12, phi=0.2,
                    beta=0.4, O0=1.5)
    t_dim, V_dim, C_dim, _, _, _ = simulate(p, V0=6.0, C0=2.0, T=80.0)
    _, V_red, C_red = simulate_reduced(p, V0=6.0, C0=2.0, T=80.0,
                                       t_eval=t_dim)
    assert np.max(np.abs(V_red - V_dim)) < 1e-6
    assert np.max(np.abs(C_red - C_dim)) < 1e-5


def test_reduced_roundtrip_eta_not_one():
    """R8 CODE-009: the seventh group eta reproduces nonlinear load mapping."""
    p = make_params(eta=1.5, beta=0.18, O0=1.2)
    t_dim, V_dim, C_dim, _, _, _ = simulate(p, V0=7.0, C0=1.5, T=60.0)
    _, V_red, C_red = simulate_reduced(
        p, V0=7.0, C0=1.5, T=60.0, t_eval=t_dim
    )
    assert np.max(np.abs(V_red - V_dim)) < 1e-6
    assert np.max(np.abs(C_red - C_dim)) < 1e-5


# ---------------------------------------------------------------------------
# Exact structural degeneracies of the interior equilibrium
# ---------------------------------------------------------------------------

def test_mu_alpha_joint_scaling_leaves_v_star_unchanged():
    """(mu, alpha) -> (2mu, 2alpha) preserves mu/alpha, hence V* exactly."""
    p = make_params()
    p2 = make_params(mu=2.0 * p['mu'], alpha=2.0 * p['alpha'])
    eq1 = _lowest_c_stable(p)
    eq2 = _lowest_c_stable(p2)
    assert abs(eq1['V'] - eq2['V']) < 1e-8


def test_R_delta_joint_scaling_leaves_equilibrium_unchanged():
    """(R, delta) -> (2R, 2delta) preserves R/delta, hence (V*, C*, O*)."""
    p = make_params()
    p2 = make_params(R=2.0 * p['R'], delta=2.0 * p['delta'])
    eq1 = _lowest_c_stable(p)
    eq2 = _lowest_c_stable(p2)
    assert abs(eq1['V'] - eq2['V']) < 1e-8
    assert abs(eq1['C'] - eq2['C']) < 1e-8
    assert abs(eq1['O'] - eq2['O']) < 1e-8


def test_beta_eta_O0_have_zero_elasticity_on_v_star():
    """The (V*, O*) equilibrium system involves neither beta, eta, nor O0,
    so their elasticities on V* are exactly zero (Lemma 2 and extensions)."""
    els = v_star_elasticities(make_params())
    assert abs(els['beta']) < 1e-6
    assert abs(els['eta']) < 1e-6
    assert abs(els['O0']) < 1e-6


def test_elasticity_ratio_pairs():
    """R/delta and mu/alpha ratio dependence forces paired elasticities."""
    els = v_star_elasticities(make_params())
    assert els['R'] == pytest.approx(-els['delta'], abs=1e-6)
    assert els['mu'] == pytest.approx(-els['alpha'], abs=1e-6)


# ---------------------------------------------------------------------------
# Illustrative mu/alpha display-threshold crossing
# ---------------------------------------------------------------------------

def test_mu_alpha_critical_baseline():
    """The flip V* = 0.5*Vmax occurs at mu/alpha ~ 2.163 at baseline phi
    (verified independently: mu = 0.2163 gives V* = 5.0006)."""
    r_crit = mu_alpha_critical(make_params())
    assert r_crit == pytest.approx(2.163, abs=0.01)


# ---------------------------------------------------------------------------
# LHS global screening
# ---------------------------------------------------------------------------

def test_legacy_lhs_threshold_fraction_genericity_bracket():
    """The deprecated compatibility API returns a nondegenerate label share.

    With n=200, seed=1, factor=2, the below-threshold share among stable
    draws must fall in a loose genericity bracket (0.25, 0.75).
    """
    with pytest.warns(DeprecationWarning, match="lhs_zombie_fraction"):
        res = lhs_zombie_fraction(make_params(), n_samples=200, factor=2.0,
                                  seed=1)
    frac = res['zombie_fraction_given_stable']
    assert 0.25 < frac < 0.75
    # basic bookkeeping consistency
    assert res['n_zombie'] <= res['n_stable'] <= res['n_samples']
    assert len(res['v_stars']) == res['n_stable']
    assert set(res['spearman']) == {
        'R', 'Vmax', 'delta', 'gamma', 'O0', 'beta', 'eta', 'alpha', 'phi',
        'mu', 'eps',
    }
    # honest labelling is part of the contract
    assert 'not' in res['method'].lower() and 'sobol' in res['method'].lower()


def test_legacy_label_maps_emit_one_deprecation_warning_per_public_call():
    p = make_params()
    with pytest.warns(DeprecationWarning, match="classify_point") as caught:
        assert classify_point(p) in {"zombie", "sustainable", "none"}
    assert len(caught) == 1
    with pytest.warns(DeprecationWarning, match="zombie_boundary_map") as caught:
        assert zombie_boundary_map(p, n=2)["regimes"].shape == (2, 2)
    assert len(caught) == 1
    with pytest.warns(DeprecationWarning, match="zombie_boundary_map_beta") as caught:
        assert zombie_boundary_map_beta(p, n=2)["regimes"].shape == (2, 2)
    assert len(caught) == 1
