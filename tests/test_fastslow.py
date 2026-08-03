"""
Tests for dlvt.fastslow: quasi-static reduction, honest timescale separation,
and the colored basin portrait (Appendix A11).

These tests pin the HONEST fast--slow finding: the raw-rate comparison
(1/R = 0.33 vs 1/mu = 5, "~15x separation") of earlier drafts is replaced by
the measured Jacobian-diagonal ratio |J_VV/J_CC| ~ 3 at the equilibrium,
together with the complex eigenvalue pair (modes mix — the damped spiral),
and by a measured (not assumed) reduction error along the baseline transient.
"""

import numpy as np
import pytest
from scipy.optimize import brentq

from dlvt.model import make_params, coordination_load
from dlvt.analysis import find_interior_equilibria
from dlvt.fastslow import (
    basin_portrait_grid,
    quasi_static_nullcline,
    reduction_error,
    slow_manifold,
    v_quasi_equilibrium,
)


# ────────────────────────────────────────────────────────────────────────────
# Quasi-equilibrium V_qe(O) — closed-form root of the nullcline quadratic
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("O", [1.0, 5.0, 9.0, 30.0])
def test_v_quasi_equilibrium_satisfies_vitality_equation(O):
    """Plugging V_qe(O) back into dV/dt at frozen O gives residual < 1e-10."""
    p = make_params()
    V = v_quasi_equilibrium(O, p)
    residual = (p["R"] * (1.0 - V / p["Vmax"])
                - p["delta"] * O ** p["gamma"] * V / (V + p["eps"]))
    assert 0.0 < V < p["Vmax"]
    assert abs(residual) < 1e-10, f"residual {residual:.3e} at O={O}"


def test_v_quasi_equilibrium_vectorized_and_axis_value():
    """Vectorized over O; at O = O0 = 1 it reproduces the axis value 9.934."""
    p = make_params()
    Os = np.array([1.0, 5.0, 9.0, 30.0])
    Vs = v_quasi_equilibrium(Os, p)
    assert Vs.shape == Os.shape
    assert Vs[0] == pytest.approx(9.934, abs=1e-3)
    # strictly decreasing in O (Proposition prop:nullcline)
    assert np.all(np.diff(Vs) < 0)


# ────────────────────────────────────────────────────────────────────────────
# Slow-manifold equilibrium == full interior equilibrium (exactly)
# ────────────────────────────────────────────────────────────────────────────


def test_reduced_equilibrium_coincides_with_full_interior_equilibrium():
    """brentq on the reduced dC/dt = 0 near C* = 32 matches (V*, C*) to 1e-6.

    The slow manifold IS the V-nullcline, so the reduced equilibrium sits at
    the intersection of both nullclines — the full interior equilibrium —
    with zero bias (Appendix A11).
    """
    p = make_params()
    eq = find_interior_equilibria(p)[0]
    V_star, C_star = eq["V"], eq["C"]

    def g(C):
        O = coordination_load(C, p)
        return (p["alpha"] * v_quasi_equilibrium(O, p)
                / (1.0 + p["phi"] * O) - p["mu"])

    C_red = brentq(g, 20.0, 45.0, xtol=1e-12, rtol=1e-14)
    V_red = quasi_static_nullcline(C_red, p)
    assert C_red == pytest.approx(C_star, abs=1e-6)
    assert V_red == pytest.approx(V_star, abs=1e-6)


def test_slow_manifold_name_is_deprecated_because_graph_is_not_invariant():
    """R8 CODE-010: preserve compatibility without overstating geometry."""
    p = make_params()
    with pytest.warns(DeprecationWarning):
        legacy = slow_manifold(5.0, p)
    assert legacy == pytest.approx(quasi_static_nullcline(5.0, p))


# ────────────────────────────────────────────────────────────────────────────
# Reduction error — measured, with honest bounds
# ────────────────────────────────────────────────────────────────────────────


def test_reduction_tracks_slow_dynamics_with_measured_error():
    """From (V0, C0) = (8, 5): the reduction tracks enacted-scope drift, but
    with a measured max relative C-error of ~15% concentrated in the spiral
    overshoot near t ~ 9 (NOT a 'few percent everywhere' — the monotone 1D
    flow cannot overshoot). Mean error is ~1.3%. Bounds are the measured
    values plus margin, and a lower bound pins that the degradation near
    the equilibrium is real, not an artifact to be tuned away."""
    p = make_params()
    res = reduction_error(p, V0=8.0, C0=5.0, T=120.0)
    # measured 0.153 — honest ceiling with margin
    assert res["max_rel_error_C"] < 0.25
    # the degradation is genuine: the overshoot mismatch exceeds 5%
    assert res["max_rel_error_C"] > 0.05
    # measured 0.013 — the reduction is good on average / far from equilibrium
    assert res["mean_rel_error_C"] < 0.05
    # equilibrium mismatch ~1e-14: the reduced equilibrium is exact
    assert res["equilibrium_mismatch"] < 1e-8


def test_eigen_separation_is_about_3_not_15_and_modes_mix():
    """The honest correction of the '~15x timescale separation' claim:
    |J_VV/J_CC| is measured ~3.0 (between 2 and 5), the raw-rate ratio
    R/mu = 15 is reported only for contrast, and the eigenvalues are the
    complex pair -0.205 +/- 0.331i, so the modes MIX near the equilibrium
    (a strict fast--slow decomposition fails there)."""
    p = make_params()
    res = reduction_error(p, V0=8.0, C0=5.0, T=120.0)
    assert 2.0 < res["eigen_separation"] < 5.0
    assert res["eigen_separation"] == pytest.approx(3.0, abs=0.2)
    assert res["raw_rate_ratio"] == pytest.approx(15.0, rel=1e-12)
    assert res["modes_mix"] is True
    lam = res["eigenvalues"]
    assert np.all(np.real(lam) < 0)
    assert np.all(np.abs(np.imag(lam)) > 0.1)


# ────────────────────────────────────────────────────────────────────────────
# Basin portrait — Theorem 2c: every C0 > 0 converges
# ────────────────────────────────────────────────────────────────────────────


def test_basin_grid_all_positive_scope_initial_conditions_converge():
    """8x8 grid, T=400: every initial condition with C0 > 0 enters and stays
    in the band around (V*, C*) — zero exceptions (Theorem 2c: the basin is
    {C > 0}, no escape, no competing attractor)."""
    p = make_params()
    g = basin_portrait_grid(p, n_V=8, n_C=8, T=400.0)
    assert g["n_total"] == 64
    assert g["n_exceptions"] == 0
    assert g["n_converged"] == g["n_total"]
    assert np.all(np.isfinite(g["T_conv"]))
    # convergence times are physically sensible: within the horizon and on
    # the order of a few spiral periods (2*pi/0.331 ~ 19 time units)
    assert g["T_conv"].max() < g["T"]
    assert np.median(g["T_conv"]) < 60.0
    assert g["V_star"] == pytest.approx(4.70253, abs=1e-3)
    assert g["C_star"] == pytest.approx(32.034, abs=1e-2)
