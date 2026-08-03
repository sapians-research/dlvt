"""
Tests for dlvt.model: core ODE, parameters, simulation invariants.

These tests pin the Python implementation to the public formal contract. In
the private monorepo that contract is traced to the active complete manuscript
and TLQ derivation by `DLVT_CODE_SYNC.md`.
They are meant to fail loudly if the model or its parameters drift
away from the paper's equations.

References
----------
Eq. (3.2)  Impact:              I = C * V / (1 + phi * O)
Eq. (3.5)  Vitality dynamics:   dV/dt = R (1 - V/Vmax)
                                       - delta * O^gamma * V / (V + eps)
Eq. (3.6)  Enacted-scope dynamics: dC/dt = alpha * I - mu * C
Eq. (3.7)  Scope-to-load mapping:  O     = O0 + beta * C^eta
"""

import math

import numpy as np
import pytest

from dlvt.model import (
    DEFAULT_PARAMS,
    complexity,
    coordination_load,
    dlvt_exogenous,
    dlvt_system,
    impact,
    make_params,
    simulate,
    validate_params,
)


# ────────────────────────────────────────────────────────────────────────────
# Parameter block — pinned to the illustrative values reported in Table 1
# ────────────────────────────────────────────────────────────────────────────


PAPER_BASELINE = dict(
    R=3.0,
    Vmax=10.0,
    delta=0.02,
    gamma=2.0,
    O0=1.0,
    beta=0.25,
    eta=1.0,
    alpha=0.1,
    phi=0.15,
    mu=0.2,
    eps=0.1,
)


def test_default_params_match_paper_baseline():
    """DEFAULT_PARAMS must match the illustrative Table 1 values."""
    for key, value in PAPER_BASELINE.items():
        assert key in DEFAULT_PARAMS, f"missing parameter {key}"
        assert DEFAULT_PARAMS[key] == pytest.approx(value), (
            f"{key}: expected {value}, got {DEFAULT_PARAMS[key]}"
        )


def test_make_params_overrides():
    """make_params must return a copy and apply overrides."""
    p = make_params(beta=0.5, R=4.0)
    assert p["beta"] == 0.5
    assert p["R"] == 4.0
    # original dict untouched
    assert DEFAULT_PARAMS["beta"] == 0.25
    assert DEFAULT_PARAMS["R"] == 3.0


def test_make_params_rejects_unknown_override():
    with pytest.raises(ValueError, match="unknown DLVT parameter.*btea"):
        make_params(btea=0.5)


@pytest.mark.parametrize("value", [0.0, -1.0, math.nan, math.inf, -math.inf])
def test_make_params_rejects_values_outside_active_domain(value):
    with pytest.raises(ValueError):
        make_params(eps=value)


@pytest.mark.parametrize("name", list(PAPER_BASELINE))
def test_make_params_requires_every_parameter_to_be_positive(name):
    with pytest.raises(ValueError, match="strictly positive"):
        make_params(**{name: 0.0})


def test_validate_params_rejects_missing_and_unknown_keys():
    missing = make_params()
    missing.pop("beta")
    with pytest.raises(ValueError, match="missing DLVT parameter.*beta"):
        validate_params(missing)

    unknown = make_params()
    unknown["btea"] = 0.5
    with pytest.raises(ValueError, match="unknown DLVT parameter.*btea"):
        validate_params(unknown)


def test_validate_params_allows_eps_zero_only_for_labelled_limit_diagnostics():
    singular = make_params()
    singular["eps"] = 0.0
    validated = validate_params(singular, require_regularized=False)
    assert validated["eps"] == 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        validate_params(singular)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"V0": -1.0},
        {"C0": -1.0},
        {"T": 0.0},
        {"max_step": 0.0},
        {"T": math.inf},
    ],
)
def test_simulate_validates_initial_conditions_and_solver_inputs(kwargs):
    with pytest.raises(ValueError):
        simulate(make_params(), **kwargs)


# ────────────────────────────────────────────────────────────────────────────
# Functional-form identities — Eq. (3.2), Eq. (3.7)
# ────────────────────────────────────────────────────────────────────────────


def test_coordination_load_formula():
    """O(C) = O0 + beta * C^eta."""
    p = make_params()
    for C in [0.0, 1.0, 5.0, 25.0, 100.0]:
        expected = p["O0"] + p["beta"] * (C ** p["eta"])
        assert coordination_load(C, p) == pytest.approx(expected)


def test_coordination_load_vectorised():
    """coordination_load() must accept numpy arrays."""
    p = make_params()
    C = np.array([0.0, 5.0, 25.0, 100.0])
    out = coordination_load(C, p)
    assert out.shape == C.shape
    np.testing.assert_allclose(
        out, p["O0"] + p["beta"] * C ** p["eta"]
    )


def test_coordination_load_nonnegative_clamp():
    """Negative C should be clamped to 0 inside coordination_load()."""
    p = make_params()
    assert coordination_load(-5.0, p) == pytest.approx(p["O0"])


def test_complexity_is_a_deprecated_coordination_load_alias():
    """The historical API name remains only as a deprecated alias."""
    p = make_params()
    with pytest.deprecated_call(match="coordination_load"):
        value = complexity(5.0, p)
    assert value == pytest.approx(coordination_load(5.0, p))


def test_impact_formula_matches_paper():
    """I = C * V / (1 + phi * O), Eq. (3.2)."""
    p = make_params()
    V, C = 7.0, 20.0
    O = coordination_load(C, p)
    expected = C * V / (1.0 + p["phi"] * O)
    assert impact(V, C, O, p) == pytest.approx(expected)


def test_impact_vanishes_at_zero_vitality():
    """When V = 0, I must be exactly 0 (energy-gating)."""
    p = make_params()
    assert impact(0.0, 50.0, 10.0, p) == 0.0


def test_impact_vanishes_in_infinite_load_limit():
    """As O → ∞, I → 0 in the experienced-load friction limit."""
    p = make_params()
    assert impact(10.0, 20.0, 1e9, p) < 1e-5


# ────────────────────────────────────────────────────────────────────────────
# ODE right-hand side — Eq. (3.5), Eq. (3.6)
# ────────────────────────────────────────────────────────────────────────────


def test_dlvt_system_matches_equations():
    """Spot-check RHS against a hand-computed value."""
    p = make_params()
    V, C = 8.0, 10.0
    O = coordination_load(C, p)
    recovery = p["R"] * (1.0 - V / p["Vmax"])
    drain = p["delta"] * O ** p["gamma"] * V / (V + p["eps"])
    I = impact(V, C, O, p)
    expected_dV = recovery - drain
    expected_dC = p["alpha"] * I - p["mu"] * C
    dV, dC = dlvt_system(0.0, [V, C], p)
    assert dV == pytest.approx(expected_dV, rel=1e-12)
    assert dC == pytest.approx(expected_dC, rel=1e-12)


def test_dv_dt_positive_at_V_zero():
    """Positive invariance (Appendix A1).

    At V = 0 the smooth barrier V/(V+eps) vanishes and dV/dt = R > 0.
    This proves V(t) ≥ 0 for all t ≥ 0.
    """
    p = make_params()
    for C in [0.0, 5.0, 50.0]:
        dV, _ = dlvt_system(0.0, [0.0, C], p)
        assert dV == pytest.approx(p["R"], rel=1e-12), (
            f"dV/dt at V=0, C={C} should equal R={p['R']}, got {dV}"
        )


def test_dc_dt_zero_at_C_zero():
    """At C = 0, dC/dt = 0 (enacted scope cannot become negative)."""
    p = make_params()
    _, dC = dlvt_system(0.0, [8.0, 0.0], p)
    assert dC == 0.0


# ────────────────────────────────────────────────────────────────────────────
# Simulation invariants
# ────────────────────────────────────────────────────────────────────────────


def test_simulate_returns_nonnegative_states():
    """V(t) ≥ 0 and C(t) ≥ 0 for all integration points."""
    p = make_params()
    t, V, C, O, I, G = simulate(p, V0=8.0, C0=5.0, T=200.0)
    assert np.all(V >= 0.0)
    assert np.all(C >= 0.0)
    assert np.all(O >= p["O0"] - 1e-12)
    assert np.all(I >= 0.0)


def test_simulate_vitality_stays_below_ceiling():
    """V(t) cannot exceed V_max under the linear-relaxation recovery law."""
    p = make_params()
    _, V, *_ = simulate(p, V0=p["Vmax"], C0=1.0, T=50.0)
    # Small numerical tolerance only.
    assert V.max() <= p["Vmax"] + 1e-6


def test_simulate_baseline_reproduces_reference_equilibrium_numerics():
    """At T = 400 with baseline params, V* ≈ 4.70 and C* ≈ 32.0.

    These are the numerics reported in Figure 3 (phase portrait) and the
    reference values for the low-vitality equilibrium in §3.8. If these drift,
    the code and the paper are no longer in sync.
    """
    p = make_params()
    _, V, C, *_ = simulate(p, V0=8.0, C0=5.0, T=400.0, max_step=0.1)
    assert V[-1] == pytest.approx(4.70, abs=0.05), (
        f"V* drift: expected ≈4.70, got {V[-1]:.3f}"
    )
    assert C[-1] == pytest.approx(32.0, abs=0.3), (
        f"C* drift: expected ≈32.0, got {C[-1]:.3f}"
    )


def test_simulate_declared_scenario_is_above_display_threshold():
    """Under the declared low-drain parameters, V* stays above the display threshold.

    This corresponds to the above-threshold illustration in Figure 2
    (delta=0.008, beta=0.15); it is not a validated performance state.
    """
    p = make_params(delta=0.008, beta=0.15)
    _, V, *_ = simulate(p, V0=8.0, C0=5.0, T=400.0, max_step=0.1)
    assert V[-1] > 5.0, (
        f"Declared scenario should yield V* > display threshold=5.0, got {V[-1]:.3f}"
    )


# ────────────────────────────────────────────────────────────────────────────
# R8 G2 theorem regressions — recovery and exogenous load
# ────────────────────────────────────────────────────────────────────────────


def test_load_free_recovery_derivative_is_finite_at_zero():
    """R8 MATH-004: delayed-recovery marginal cost does not diverge at Vb=0."""
    p = make_params()
    V_a = 8.0

    def recovery_time(V_b):
        return (p["Vmax"] / p["R"]) * math.log(
            (p["Vmax"] - V_b) / (p["Vmax"] - V_a)
        )

    derivative_at_zero = -p["Vmax"] / (p["R"] * p["Vmax"])
    assert derivative_at_zero == pytest.approx(-1.0 / p["R"], rel=1e-12)

    h = 1e-6
    finite_difference = (recovery_time(h) - recovery_time(0.0)) / h
    assert finite_difference == pytest.approx(derivative_at_zero, rel=2e-6)
    assert math.isfinite(finite_difference)


def test_strong_exogenous_load_enters_epsilon_band_within_bound():
    """R8 M12: sufficiently strong fixed load gives finite-time band entry."""
    from scipy.integrate import solve_ivp

    p = make_params()
    Omega = 30.0
    V0 = 8.0
    k = p["delta"] * Omega ** p["gamma"] / 2.0 - p["R"]
    assert k > 0.0
    bound = (V0 - p["eps"]) / k
    C_fixed = ((Omega - p["O0"]) / p["beta"]) ** (1.0 / p["eta"])

    sol = solve_ivp(
        dlvt_exogenous,
        [0.0, bound],
        [V0],
        args=(p, lambda _t: C_fixed),
        max_step=1e-3,
        rtol=1e-10,
        atol=1e-12,
    )
    assert sol.success
    assert sol.y[0, -1] <= p["eps"]


def test_small_persistent_gamma_deficit_need_not_enter_epsilon_band():
    """R8 MATH-005: Gamma>1 alone is insufficient for V<epsilon."""
    from scipy.integrate import solve_ivp

    p = make_params()
    drain_coefficient = 3.1  # exceeds R=3, but only slightly
    Omega = (drain_coefficient / p["delta"]) ** (1.0 / p["gamma"])
    C_fixed = ((Omega - p["O0"]) / p["beta"]) ** (1.0 / p["eta"])
    assert p["delta"] * Omega ** p["gamma"] / p["R"] > 1.0

    sol = solve_ivp(
        dlvt_exogenous,
        [0.0, 100.0],
        [8.0],
        args=(p, lambda _t: C_fixed),
        max_step=0.02,
        rtol=1e-10,
        atol=1e-12,
    )
    assert sol.success
    assert sol.y[0, -1] == pytest.approx(0.806536, abs=1e-5)
    assert sol.y[0, -1] > p["eps"]
