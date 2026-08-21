"""R8 tests for structural scope absorption in alternative drain kernels."""

import pytest

from dlvt.model import make_params
from scripts.robustness_grid import (
    NO,
    YES,
    build_rows,
    equilibrium_diagnostics,
    explicit_C_drain_rhs,
    exponential_in_load_rhs,
    exponential_in_scope_rhs,
    hill_rhs,
    integrate_to_equilibrium,
    power_law_rhs,
    saturating_complexity_rhs,
    saturating_load_rhs,
    scope_absorption_breaks,
)


@pytest.mark.parametrize(
    "factory",
    [
        power_law_rhs,
        lambda p: hill_rhs(p, K=4.0),
        lambda p: exponential_in_load_rhs(p, kappa=0.08),
    ],
)
def test_scope_absorption_holds_for_load_only_kernels(factory):
    """R8 M7: power-law, Hill, and exponential-in-O kernels preserve V*."""
    p = make_params()
    assert scope_absorption_breaks(factory, p) == NO

    p_low = make_params(beta=0.15)
    p_high = make_params(beta=0.40)
    v_low, _, ok_low = integrate_to_equilibrium(factory(p_low))
    v_high, _, ok_high = integrate_to_equilibrium(factory(p_high))
    assert ok_low and ok_high
    assert v_low == pytest.approx(v_high, abs=2e-5)


@pytest.mark.parametrize(
    "factory",
    [
        lambda p: exponential_in_scope_rhs(p, kappa=0.35),
        lambda p: explicit_C_drain_rhs(p, theta=0.001),
    ],
)
def test_explicit_scope_dependence_breaks_absorption(factory):
    """R8 M8: a nonabsorbed C term makes equilibrium vitality beta-sensitive."""
    assert scope_absorption_breaks(factory, make_params()) == YES


def test_robustness_grid_is_deterministic_and_semantically_complete():
    """The standalone suite checks generator semantics, not private tables."""
    first = build_rows()
    second = build_rows()
    assert first == second
    assert len(first) >= 10
    assert all(row["exists"] == YES for row in first)
    assert all(float(row["V_star"]) > 0.0 for row in first)
    assert all(float(row["C_star"]) > 0.0 for row in first)


def test_equilibrium_diagnostics_verify_residual_and_stability():
    p = make_params()
    rhs = power_law_rhs(p)
    v_star, c_star, converged = integrate_to_equilibrium(rhs)
    diagnostics = equilibrium_diagnostics(rhs, v_star, c_star)
    assert converged
    assert diagnostics["residual_ok"]
    assert diagnostics["residual_norm"] < 1e-5
    assert diagnostics["stable"]
    assert max(value.real for value in diagnostics["eigenvalues"]) < 0.0


def test_positive_point_alone_is_not_an_equilibrium():
    diagnostics = equilibrium_diagnostics(
        lambda _t, _y: [1.0, -1.0],
        V_star=1.0,
        C_star=1.0,
    )
    assert not diagnostics["residual_ok"]


def test_saturating_complexity_alias_warns_and_matches_load_kernel():
    """R8: the only deprecated alias that used to redirect silently.

    ``O`` is experienced coordination load, not organizational complexity, so
    the retired spelling must announce itself like every other alias.
    """
    p = make_params()
    with pytest.deprecated_call(match="saturating_load_rhs"):
        legacy = saturating_complexity_rhs(p, K=50.0)
    canonical = saturating_load_rhs(p, K=50.0)
    assert legacy(0.0, [8.0, 5.0]) == canonical(0.0, [8.0, 5.0])
