"""
Tests for dlvt.analysis: coefficient thresholds, equilibria, and stability.

Pins analytical results to the final paper (§3.6–3.9 and Appendix A4–A6).
"""

import numpy as np
import pytest

from dlvt.model import make_params
from dlvt.analysis import (
    DISPLAY_THRESHOLD_FRACTION,
    basin_of_attraction_sweep,
    bendixson_dulac_certificate,
    carrying_capacity,
    classify_equilibrium,
    drain_coefficient_threshold,
    estimate_bifurcation_interval,
    find_interior_equilibria,
    find_regularization_branch,
    is_low_vitality,
    trapping_capital_bound,
    trapping_scope_bound,
)


# ────────────────────────────────────────────────────────────────────────────
# Drain-coefficient threshold — Proposition 3 / Eq. (3.14)
# ────────────────────────────────────────────────────────────────────────────


def test_drain_coefficient_threshold_baseline_equals_45():
    """At baseline parameters, the Gamma=1 threshold is approximately 45."""
    p = make_params()
    assert drain_coefficient_threshold(p) == pytest.approx(45.0, abs=0.1)


def test_drain_coefficient_threshold_formula_matches_definition():
    """C_Gamma = ((R/delta)^(1/gamma) - O0) / beta for eta=1."""
    p = make_params()
    Omax = (p["R"] / p["delta"]) ** (1.0 / p["gamma"])
    expected = (Omax - p["O0"]) / p["beta"]
    assert drain_coefficient_threshold(p) == pytest.approx(expected, rel=1e-12)


def test_drain_coefficient_threshold_eta2_value():
    """C_Gamma drops from 45.0 at eta=1 to approximately 6.7 at eta=2."""
    p = make_params(eta=2.0)
    assert drain_coefficient_threshold(p) == pytest.approx(6.7, abs=0.1)


def test_drain_coefficient_threshold_eta1_5_value():
    """C_Gamma is approximately 12.7 at eta=1.5."""
    p = make_params(eta=1.5)
    assert drain_coefficient_threshold(p) == pytest.approx(12.7, abs=0.1)


def test_drain_coefficient_threshold_monotone_in_beta():
    """The mapped Gamma=1 threshold decreases with beta."""
    betas = [0.10, 0.15, 0.25, 0.40, 0.60]
    values = [drain_coefficient_threshold(make_params(beta=b)) for b in betas]
    # strictly decreasing
    for a, b in zip(values, values[1:]):
        assert a > b, f"C*_max not monotone in beta: {values}"


def test_drain_coefficient_threshold_monotone_in_R():
    """The mapped Gamma=1 threshold increases with R."""
    Rs = [1.0, 2.0, 3.0, 4.5, 6.0]
    values = [drain_coefficient_threshold(make_params(R=r)) for r in Rs]
    for a, b in zip(values, values[1:]):
        assert a < b, f"C*_max not monotone in R: {values}"


def test_drain_coefficient_threshold_zero_when_O0_too_large():
    """Return zero when Gamma>=1 already at baseline load."""
    # Force (R/delta)^{1/gamma} < O0
    p = make_params(O0=100.0)
    assert drain_coefficient_threshold(p) == 0.0


def test_drain_coefficient_threshold_replaces_carrying_capacity_semantics():
    """R8 M9: the deprecated name must not restore capacity semantics."""
    from dlvt.model import dlvt_system

    p = make_params()
    c_gamma = drain_coefficient_threshold(p)
    assert c_gamma == pytest.approx(44.989795, rel=1e-7)
    with pytest.warns(DeprecationWarning):
        assert carrying_capacity(p) == pytest.approx(c_gamma, rel=1e-12)

    # At full vitality recovery is zero, so the vitality derivative is
    # strictly negative.  Gamma=1 cannot mean recovery offsets drain there.
    dV, _ = dlvt_system(0.0, [p["Vmax"], c_gamma], p)
    assert dV < 0.0


def test_scope_named_trapping_bound_is_canonical_with_deprecated_alias():
    p = make_params()
    expected = trapping_scope_bound(p)
    assert expected == pytest.approx(102.6666667)
    with pytest.warns(DeprecationWarning, match="trapping_scope_bound"):
        legacy = trapping_capital_bound(p)
    assert legacy == pytest.approx(expected)


# ────────────────────────────────────────────────────────────────────────────
# Interior equilibria and stability — Theorem 2
# ────────────────────────────────────────────────────────────────────────────


def test_baseline_equilibrium_is_low_at_half_threshold_and_stable():
    """Baseline calibration is stable and below the illustrative half threshold.

    The paper reports V* ≈ 4.70, C* ≈ 32 at the baseline. The Jacobian
    should have complex eigenvalues with negative real parts (damped spiral).
    """
    p = make_params()
    eqs = find_interior_equilibria(p)
    assert len(eqs) >= 1
    stable = [e for e in eqs if e.get("stable")]
    assert len(stable) == 1
    eq = stable[0]
    assert is_low_vitality(eq["V"], p, 0.5)
    assert eq["V"] == pytest.approx(4.70, abs=0.05)
    assert eq["C"] == pytest.approx(32.0, abs=0.3)
    # Damped oscillatory convergence: complex conjugate eigenvalues
    lam = np.asarray(eq["eigenvalues"])
    assert np.all(np.real(lam) < 0.0), f"unstable: eigs={lam}"
    assert np.any(np.imag(lam) != 0.0), (
        f"expected complex eigenvalues (spiral), got {lam}"
    )


def test_declared_above_threshold_scenario_is_not_low_at_half_threshold():
    """The declared low-delta scenario is above the illustrative threshold."""
    p = make_params(delta=0.008, beta=0.15)
    eqs = find_interior_equilibria(p)
    assert any(e["stable"] and not is_low_vitality(e["V"], p, 0.5) for e in eqs)


def test_high_coupling_preserves_low_vitality_under_scope_absorption():
    """Doubling beta preserves V* and rescales C* under scope absorption."""
    p = make_params(beta=0.5)
    eqs = find_interior_equilibria(p)
    stable = [e for e in eqs if e.get("stable")]
    assert len(stable) == 1
    eq = stable[0]
    assert is_low_vitality(eq["V"], p, 0.5)
    baseline = find_interior_equilibria(make_params())[0]
    assert eq["V"] == pytest.approx(baseline["V"], rel=1e-12)
    assert eq["C"] == pytest.approx(0.5 * baseline["C"], rel=1e-12)


# ────────────────────────────────────────────────────────────────────────────
# Bifurcation interval diagnostic — R7-6 / Appendix A8
# ────────────────────────────────────────────────────────────────────────────

# A single session-scoped estimate shared by all four R7-6 tests. The grid is
# deliberately small (3 eps × 1 n_scan × 60 beta points) so the whole group
# finishes in well under a minute; the appendix uses a wider grid for reporting.

@pytest.fixture(scope="module")
def bifurcation_result():
    p = make_params()
    return estimate_bifurcation_interval(
        p,
        eps_grid=[0.05, 0.1, 0.2],
        n_scan_grid=[4000],
        n_beta=60,
    )


def test_bifurcation_interval_baseline_does_not_cross_display_threshold(bifurcation_result):
    """R7-6: under baseline parameters, V*(β) does not cross the display threshold.

    This is the scope-absorption property of Lemma 2 made mechanical. The
    earlier claim β_crit ≈ 0.1015 was a scan-window artifact; the honest
    report is that no V*-crossing exists at this calibration.
    """
    assert bifurcation_result["crosses_threshold"] is False
    assert bifurcation_result["beta_crit_interval"] is None


def test_bifurcation_interval_baseline_v_star_is_invariant(bifurcation_result):
    """R7-6: V*(β) is numerically constant across β at the baseline eps.

    Pins Lemma 2: V* depends on β only through β·C*^η, which the equilibrium
    conditions force to be invariant. The reference value 4.7025 is the
    scope-absorption invariant at eps=0.1.
    """
    assert bifurcation_result["v_star_invariant"] is not None
    assert abs(bifurcation_result["v_star_invariant"] - 4.7025) < 5e-3


def test_bifurcation_interval_baseline_beta_C_product_is_pinned(bifurcation_result):
    """R7-6: the invariant product β·C* is pinned to its analytical value.

    From the equilibrium conditions with γ=2, η=1 and baseline parameters,
    β·C* = 8.0083 (derived from V* = μ(1+φO*)/α = 4.7025 and the definition
    of O*). This product is the structural invariant that Lemma 2 names.
    """
    assert bifurcation_result["baseline_beta_C_product"] is not None
    assert abs(bifurcation_result["baseline_beta_C_product"] - 8.0083) < 5e-3


def test_bifurcation_interval_diagnostic_names_the_artifact(bifurcation_result):
    """R7-6: the diagnostic string explicitly identifies the scan-window artifact.

    The earlier 'β_crit ≈ 0.1015' figure was a function of the legacy C_max=80
    in find_interior_equilibria — specifically, it was the smallest β at which
    C*(β) = 8.008/β fell below 80. The diagnostic must surface this so a
    reader cannot accidentally reproduce the false-precision claim.
    """
    diag = bifurcation_result["diagnostic"].lower()
    assert "scan" in diag or "c_max" in diag
    assert "artifact" in diag or "invariant" in diag


# ────────────────────────────────────────────────────────────────────────────
# ε-regularization root audit — R8 M11 / Appendix A9
# ────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def regularization_report():
    """Cache the expensive enumeration across R7-3 tests."""
    return find_regularization_branch(make_params())


def test_regularization_branch_no_near_zero_at_baseline(regularization_report):
    """No ε-spurious near-zero equilibrium exists at baseline parameters.

    Pins the core claim of Appendix A9: the smooth barrier V/(V+ε) does
    *not* introduce a second equilibrium branch at small positive V. The
    V-isocline quadratic has product-of-roots -ε·Vmax = -1.0 < 0, so
    exactly one positive root per fixed C; the interior C-isocline bounds
    V* ≥ μ/α = 2.0. Therefore no interior or axis equilibrium can sit
    below the near-zero threshold.
    """
    assert regularization_report["near_zero_branch"] is None
    assert regularization_report["unexpected_branch"] is None
    assert regularization_report["regularization_structure_valid"] is True
    assert regularization_report["small_magnitude_equilibria"] == []
    assert regularization_report["interior_V_lower_bound"] == pytest.approx(2.0, abs=1e-9)
    assert regularization_report["quadratic_positive_root_count"] == 1


def test_regularization_branch_axis_equilibrium_is_saddle(regularization_report):
    """The C=0 axis equilibrium is a saddle at baseline.

    The axis V-isocline quadratic at baseline gives V ≈ 9.934, and the
    Jacobian there is upper-triangular (J[1,0] = αC/(1+φO) = 0) with
    diagonal entries of opposite sign: one negative eigenvalue (vitality
    relaxation toward V*) and one positive eigenvalue (αV/(1+φO) - μ > 0).
    This classifies the axis root as a saddle, not a stable node, so it
    is *not* an attractor for any interior trajectory.
    """
    axis = regularization_report["axis_equilibrium"]
    assert axis is not None
    assert axis["classification"] == "saddle"
    assert not axis["stable"]
    # Pin the specific baseline location (Chapter 3, Appendix A9).
    assert axis["V"] == pytest.approx(9.934, abs=1e-2)
    assert axis["C"] == 0.0
    eigvals = np.real(axis["eigenvalues"])
    assert np.min(eigvals) < 0 < np.max(eigvals)


def test_regularization_branch_interior_is_unique_and_below_threshold(regularization_report):
    """The interior branch contains one stable equilibrium, low at theta=.5."""
    interior = regularization_report["interior_equilibria"]
    assert len(interior) == 1
    eq = interior[0]
    assert eq["V"] == pytest.approx(4.7025, abs=5e-3)
    assert eq["C"] == pytest.approx(32.0337, abs=5e-3)
    assert eq["stable"] is True
    assert is_low_vitality(eq["V"], make_params(), 0.5)


def test_regularization_branch_diagnostic_cites_appendix(regularization_report):
    """The diagnostic string explicitly names Appendix A9 and the root count."""
    diag = regularization_report["diagnostic"].lower()
    assert "appendix a9" in diag
    assert "μ/α" in diag or "mu/alpha" in diag or "2.000" in diag
    assert "exactly one positive axis root" in diag


@pytest.mark.parametrize("delta", [3.0, 10.0, 100.0])
def test_unique_small_axis_root_is_not_a_regularization_violation(delta):
    """R8 M11: a unique small positive axis root is legitimate.

    Strong drain can move the only positive root arbitrarily close to zero.
    Appendix A9 constrains root count, not its distance from the boundary.
    """
    report = find_regularization_branch(make_params(delta=delta))

    assert report["quadratic_positive_root_count"] == 1
    assert report["regularization_structure_valid"] is True
    assert report["unexpected_branch"] is None
    assert report["near_zero_branch"] is None
    assert len(report["small_magnitude_equilibria"]) == 1
    assert report["small_magnitude_equilibria"][0]["source"] == "axis"
    assert "does not violate" in report["diagnostic"].lower()


# ────────────────────────────────────────────────────────────────────────────
# Global asymptotic stability — R7 issue 2 / Appendix A10
# ────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def bendixson_report():
    """Cache the Bendixson-Dulac certificate computation."""
    return bendixson_dulac_certificate(make_params())


@pytest.fixture(scope="module")
def basin_report():
    """Cache the basin-of-attraction sweep; integrates 64 ODEs."""
    return basin_of_attraction_sweep(make_params())


def test_bendixson_dulac_divergence_strictly_negative(bendixson_report):
    """Pin the Dulac certificate: max divergence must be < 0.

    This is the numerical witness of the analytical Bendixson-Dulac result
    with B(V,C) = 1/C. Both ∂(Bf)/∂V and ∂(Bg)/∂C are strictly negative
    on {V>0, C>0}, so their sum is bounded away from zero — confirming
    that no closed orbits exist in the trapping rectangle, which, combined
    with uniqueness of the interior equilibrium and Poincaré-Bendixson,
    gives global asymptotic stability (Appendix A10).
    """
    assert bendixson_report["divergence_is_strictly_negative"]
    assert bendixson_report["max_divergence"] < -1e-4


def test_bendixson_dulac_trapping_rectangle_is_valid(bendixson_report):
    """Pin the analytical C_trap and its forward-invariance check."""
    # Baseline C_trap = ((α·Vmax/μ - 1)/φ - O0)/β = ((5 - 1)/0.15 - 1)/0.25
    #                 = (26.667 - 1)/0.25 = 102.667.
    assert bendixson_report["c_trap"] == pytest.approx(102.667, abs=1e-2)
    assert bendixson_report["dc_dt_above_c_trap_is_negative"]


def test_basin_sweep_all_trajectories_converge_to_interior_equilibrium(basin_report):
    """Numerical corroboration of Theorem 2: every IC on the default grid
    converges to the unique interior equilibrium."""
    assert basin_report["n_converged"] == basin_report["n_total"]
    assert basin_report["n_total"] >= 64  # 8 × 8 default grid
    assert basin_report["max_final_error"] < 1e-2
    assert basin_report["non_converged"] == []


def test_basin_sweep_reports_canonical_equilibrium_target(basin_report):
    """R8: the canonical key names the equilibrium, not a retired label."""
    V_target, C_target = basin_report["equilibrium_target"]
    assert V_target == pytest.approx(4.7025, abs=5e-3)
    assert C_target == pytest.approx(32.0337, abs=5e-3)


def test_basin_sweep_legacy_target_key_mirrors_canonical_key(basin_report):
    """The deprecated compatibility key holds the identical tuple."""
    assert basin_report["zombie_target"] == basin_report["equilibrium_target"]


def test_linear_drain_gamma1_yields_above_threshold_equilibrium():
    """At gamma=1 the stable equilibrium is above the display threshold.

    Earlier drafts said the equilibrium "disappears" at gamma=1. That was wrong:
    a unique stable interior equilibrium persists, but with V* ≈ 8.56 — well above
    the illustrative threshold. This changes a calibrated label, not the
    existence of an attractor or a validated leadership state.
    """
    p = make_params(gamma=1.0)
    eqs = find_interior_equilibria(p, n_scan=12000)
    assert len(eqs) == 1, f"expected unique interior equilibrium, got {len(eqs)}"
    eq = eqs[0]
    assert eq["stable"] is True
    assert not is_low_vitality(eq["V"], p, 0.5)
    assert eq["V"] == pytest.approx(8.559, abs=0.02)
    assert eq["V"] > DISPLAY_THRESHOLD_FRACTION * p["Vmax"]


# ────────────────────────────────────────────────────────────────────────────
# Independent verification checks (not pinned to the solver's own output)
#
# Each test below checks the code against a source of truth *external* to
# find_interior_equilibria itself: a closed-form root, a second integrator,
# a hand-derived boundary, or an analytic sign condition evaluated directly.
# ────────────────────────────────────────────────────────────────────────────


def test_equilibrium_matches_eps_to_zero_closed_form_oracle():
    """External oracle: the ε → 0 equilibrium has a closed form (γ=2, η=1).

    Substituting the C-nullcline V = (μ/α)(1+φO) into the ε → 0 V-nullcline
    R(1 − V/Vmax) = δO² gives the quadratic

        δ·O² + (R/Vmax)(μφ/α)·O + R((μ/α)/Vmax − 1) = 0,

    whose positive root at baseline is O* = 8.93313, hence V* = 4.67994 and
    C* = 31.7325 (β·C* = 7.93313). The numerical solver at ε = 1e-6 must
    reproduce this closed form — a check fully independent of the scan/brentq
    pipeline that produced the paper's reference numbers.
    """
    p = make_params(eps=1e-6)
    a = p["delta"]
    b = (p["R"] / p["Vmax"]) * (p["mu"] * p["phi"] / p["alpha"])
    c = p["R"] * ((p["mu"] / p["alpha"]) / p["Vmax"] - 1.0)
    O_star = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
    V_star = (p["mu"] / p["alpha"]) * (1.0 + p["phi"] * O_star)
    C_star = (O_star - p["O0"]) / p["beta"]

    assert O_star == pytest.approx(8.93313, abs=1e-4)
    assert V_star == pytest.approx(4.67994, abs=1e-4)

    eqs = find_interior_equilibria(p, n_scan=16000)
    assert len(eqs) == 1
    assert eqs[0]["V"] == pytest.approx(V_star, abs=1e-3)
    assert eqs[0]["C"] == pytest.approx(C_star, abs=1e-2)


def test_equilibrium_confirmed_by_independent_stiff_integrator():
    """External oracle: an implicit (Radau) integration lands on the equilibrium.

    The package integrates with RK45 everywhere; this check uses a different
    method family (implicit Runge–Kutta, Radau IIA) so that solver-specific
    artifacts cannot silently pin both the equilibrium finder and the tests.
    """
    from scipy.integrate import solve_ivp
    from dlvt.model import dlvt_system

    p = make_params()
    sol = solve_ivp(
        dlvt_system, [0.0, 400.0], [8.0, 5.0], args=(p,),
        method="Radau", rtol=1e-9, atol=1e-11,
    )
    assert sol.success
    assert sol.y[0, -1] == pytest.approx(4.7025, abs=1e-2)
    assert sol.y[1, -1] == pytest.approx(32.034, abs=1e-1)


def draw_f(rng, base):
    """Log-uniform ±2× draw around a base value."""
    return float(base * np.exp(rng.uniform(np.log(0.5), np.log(2.0))))


def test_uniqueness_no_multiple_equilibria_across_random_parameters():
    """Theorem 2a witness: at most one interior equilibrium across configurations.

    dΦ/dO < 0 term-by-term along the C-nullcline implies at most one interior
    equilibrium in the whole positive orthant for every positive parameter
    vector. Sweep 100 random log-uniform parameter draws (±2×) and assert no
    draw ever produces two interior equilibria.
    """
    rng = np.random.default_rng(20260701)

    for _ in range(100):
        p = make_params(
            R=draw_f(rng, 3.0), delta=draw_f(rng, 0.02), gamma=draw_f(rng, 2.0),
            O0=draw_f(rng, 1.0), beta=draw_f(rng, 0.25), eta=draw_f(rng, 1.0),
            alpha=draw_f(rng, 0.1), phi=draw_f(rng, 0.15), mu=draw_f(rng, 0.2),
        )
        eqs = find_interior_equilibria(p, n_scan=4000)
        assert len(eqs) <= 1, (
            f"multiple interior equilibria found — contradicts Theorem 2a: "
            f"params={p}, eqs={[(e['V'], e['C']) for e in eqs]}"
        )


def test_no_hopf_trace_negative_at_every_interior_equilibrium():
    """Theorem 2b witness: trace(J) < 0 at every interior equilibrium.

    At any interior equilibrium dC/dt = 0 forces αV/(1+φO) = μ, which makes
    J[1,1] = −αCVφ(dO/dC)/(1+φO)² < 0, while J[0,0] < 0 identically — so no
    Hopf bifurcation exists anywhere in parameter space. Evaluate the trace
    directly (bypassing jacobian_eigenvalues) on random draws.
    """
    rng = np.random.default_rng(8008)
    checked = 0
    for _ in range(100):
        p = make_params(
            R=draw_f(rng, 3.0), delta=draw_f(rng, 0.02), gamma=draw_f(rng, 2.0),
            O0=draw_f(rng, 1.0), beta=draw_f(rng, 0.25), eta=draw_f(rng, 1.0),
            alpha=draw_f(rng, 0.1), phi=draw_f(rng, 0.15), mu=draw_f(rng, 0.2),
        )
        for eq in find_interior_equilibria(p, n_scan=4000):
            V, C = eq["V"], eq["C"]
            O = p["O0"] + p["beta"] * C ** p["eta"]
            dOdC = p["beta"] * p["eta"] * C ** (p["eta"] - 1.0)
            J00 = (-p["R"] / p["Vmax"]
                   - p["delta"] * O ** p["gamma"] * p["eps"] / (V + p["eps"]) ** 2)
            J11 = (p["alpha"] * V / (1.0 + p["phi"] * O)
                   - p["alpha"] * C * V * p["phi"] * dOdC / (1.0 + p["phi"] * O) ** 2
                   - p["mu"])
            assert J00 + J11 < 0.0, (
                f"trace ≥ 0 at interior equilibrium — Hopf candidate: params={p}"
            )
            checked += 1
    assert checked >= 50, f"too few interior equilibria sampled ({checked})"


def test_mu_alpha_display_threshold_crossing_at_baseline_phi():
    """Illustrative label crossing: (μ/α)_display ≈ 2.163 at baseline phi.

    The display label depends on μ/α: V* = (μ/α)(1+φO*). Holding α = 0.1,
    μ = 0.2163 puts V* at the threshold (5.0006), μ = 0.20 (baseline) is
    below the half-maximum display threshold, and μ = 0.25 is above it with
    V* ≈ 5.58. This pins calibration dependence without calling the crossing
    a dynamical boundary.
    """
    assert classify_equilibrium(make_params(mu=0.20))["status"] == "low-vitality"
    assert classify_equilibrium(make_params(mu=0.25))["status"] == "above-threshold"

    eq_at_crit = find_interior_equilibria(make_params(mu=0.2163))[0]
    assert eq_at_crit["V"] == pytest.approx(5.0, abs=0.01)

    eq_above = find_interior_equilibria(make_params(mu=0.25))[0]
    assert eq_above["V"] == pytest.approx(5.58, abs=0.01)


def test_small_beta_equilibrium_found_with_default_window():
    """Regression for the fixed-window bug (M6): β < 0.066 must not be mislabeled.

    With the legacy fixed default C_max = 120, C*(β) ≈ 8.008/β exceeded the
    window for β < 0.066, find_interior_equilibria returned [], and
    the legacy classifier mislabeled the point as absent. The corrected
    default derives the window from C_trap ∝ 1/β, so the equilibrium is found
    at every β and scope absorption holds: V* = 4.7025 with β·C* conserved.
    """
    for beta in (0.05, 0.02, 0.01):
        p = make_params(beta=beta)
        eqs = find_interior_equilibria(p)
        assert len(eqs) == 1, f"equilibrium missed at beta={beta}"
        eq = eqs[0]
        assert eq["V"] == pytest.approx(4.7025, abs=5e-3)
        assert beta * eq["C"] == pytest.approx(8.008, abs=5e-3)
        report = classify_equilibrium(p)
        assert report["status"] == "low-vitality"


def test_classifier_preserves_existence_outside_reporting_window():
    """R8 M2: C_max is a reporting window, not an existence condition."""
    p = make_params()
    exact = find_interior_equilibria(p)[0]
    assert exact["C"] > 1.0

    report = classify_equilibrium(p, C_max=1.0)

    assert report["status"] == "equilibrium-outside-reporting-window"
    assert report["mathematical_status"] == "low-vitality"
    assert report["exists"] is True
    assert report["equilibrium"] is not None
    assert report["equilibrium"]["C"] == pytest.approx(exact["C"])
    assert report["within_reporting_window"] is False
    assert report["equilibrium"]["within_reporting_window"] is False


def test_phi_increase_raises_v_star_on_admissible_branch():
    """R8 M13: the comparative-static sign for phi is positive."""
    low_phi = find_interior_equilibria(make_params(phi=0.10))[0]
    high_phi = find_interior_equilibria(make_params(phi=0.20))[0]

    assert low_phi["V"] == pytest.approx(3.932171, rel=2e-6)
    assert high_phi["V"] == pytest.approx(5.365881, rel=2e-6)
    assert high_phi["V"] > low_phi["V"]


def test_O0_changes_admissibility_and_scope_but_not_v_or_load_root():
    """R8 M13: O0 maps scope and bounds admissibility; it does not move (V*,O*)."""
    low = find_interior_equilibria(make_params(O0=0.5))[0]
    high = find_interior_equilibria(make_params(O0=2.0))[0]

    assert low["V"] == pytest.approx(high["V"], rel=2e-11)
    assert low["O"] == pytest.approx(high["O"], rel=2e-11)
    assert low["C"] > high["C"]

    boundary_O0 = low["O"]
    assert find_interior_equilibria(make_params(O0=boundary_O0)) == []
    assert find_interior_equilibria(make_params(O0=boundary_O0 + 1e-6)) == []


def test_trapping_bound_and_coefficient_threshold_are_distinct():
    """M2 regression: C_trap (102.67) differs from C_Gamma (44.99).

    The rectangle [0, Vmax] × [0, C*_max] used in earlier drafts LEAKS: at
    C = C*_max and V = Vmax, dC/dt > 0. The corrected ceiling C_trap satisfies
    dC/dt < 0 for all V ∈ [0, Vmax] at any C > C_trap. Both facts are checked
    directly on the RHS, independent of the certificate function.
    """
    from dlvt.model import dlvt_system
    from dlvt.analysis import trapping_scope_bound

    p = make_params()
    c_trap = trapping_scope_bound(p)
    cc = drain_coefficient_threshold(p)

    assert c_trap == pytest.approx(102.667, abs=1e-2)
    assert cc == pytest.approx(44.99, abs=0.05)
    assert c_trap > cc

    # The old rectangle leaks at its top edge:
    _, dC_leak = dlvt_system(0.0, [p["Vmax"], cc], p)
    assert dC_leak > 0.0, "old C*_max ceiling should leak (dC/dt > 0)"

    # The corrected ceiling traps for every V in [0, Vmax]:
    for V in np.linspace(0.0, p["Vmax"], 21):
        _, dC = dlvt_system(0.0, [V, 1.001 * c_trap], p)
        assert dC < 0.0, f"C_trap ceiling fails to trap at V={V}"


# ---------------------------------------------------------------------------
# R8 G2 regressions — exact existence contract and boundary equilibria
# ---------------------------------------------------------------------------


def test_equilibrium_below_legacy_scan_floor_is_found():
    """R8 CODE-001: a valid root with C* < 0.01 must not be discarded.

    At delta=2.408 the exact scalar F(O) condition is positive at O0 but
    only barely, placing the unique equilibrium at C*=0.00198. The legacy
    scan started at C=0.01 and therefore returned an empty list.
    """
    p = make_params(delta=2.408)
    eqs = find_interior_equilibria(p)
    assert len(eqs) == 1
    assert eqs[0]["C"] == pytest.approx(0.00197781, rel=2e-5)
    assert eqs[0]["V"] == pytest.approx(2.30014834, rel=2e-7)


def test_equilibrium_near_vmax_is_not_discarded():
    """R8 CODE-001: an interior root above 0.999*Vmax is still valid."""
    p = make_params(delta=1e-6)
    eqs = find_interior_equilibria(p)
    assert len(eqs) == 1
    assert 0.999 * p["Vmax"] < eqs[0]["V"] < p["Vmax"]
    assert eqs[0]["V"] == pytest.approx(9.99765448, rel=2e-8)
    assert eqs[0]["C"] == pytest.approx(102.635393, rel=2e-7)


def test_existence_condition_includes_baseline_drain():
    """R8 M2: the old alpha*Vmax/mu inequality is not sufficient.

    The scope-nullcline ceiling condition still holds at delta=100, but
    F(O0)<0 because baseline drain is too strong, so no interior equilibrium
    exists.
    """
    p = make_params(delta=100.0)
    assert p["alpha"] * p["Vmax"] / p["mu"] > 1.0 + p["phi"] * p["O0"]
    assert find_interior_equilibria(p) == []


def test_classifier_does_not_use_zero_coefficient_threshold_as_shortcut():
    """R8 CODE-002: Gamma=1 may have no positive crossing while an interior
    stable equilibrium still exists.

    The legacy classifier returned an absence label before looking for the
    actual equilibrium. The corrected result is a stable threshold-low
    equilibrium.
    """
    p = make_params(mu=0.001, O0=13.0)
    eqs = find_interior_equilibria(p)
    assert len(eqs) == 1 and eqs[0]["stable"]
    assert eqs[0]["V"] == pytest.approx(0.0433380, rel=2e-5)
    assert classify_equilibrium(p)["status"] == "low-vitality"


def test_legacy_regime_classifier_emits_deprecation_warning():
    from dlvt.analysis import classify_regime

    with pytest.deprecated_call(match="classify_equilibrium"):
        result = classify_regime(make_params())
    assert result == "zombie"


def test_legacy_low_vitality_alias_emits_deprecation_warning():
    """R8 M14: is_zombie() is a deprecated alias for is_low_vitality()."""
    from dlvt.analysis import is_zombie

    p = make_params()
    V_star = find_interior_equilibria(p)[0]["V"]
    expected = is_low_vitality(V_star, p)
    with pytest.deprecated_call(match="is_low_vitality"):
        legacy = is_zombie(V_star, p)
    assert legacy is expected


def test_regime_map_warns_once_per_call_not_once_per_cell():
    """The deprecated grid API must warn on its own, like the other aliases.

    ``regime_map`` previously had no ``warnings.warn`` of its own and leaned on
    the warning inside ``classify_regime``, which fires inside the grid loop.
    """
    from dlvt.analysis import regime_map

    betas = np.array([0.05, 0.10])
    deltas = np.array([0.02, 0.05])
    with pytest.warns(DeprecationWarning) as record:
        regimes = regime_map(betas, deltas)
    assert regimes.shape == (2, 2)
    own = [w for w in record if "regime_map()" in str(w.message)]
    assert len(own) == 1, f"expected exactly one regime_map warning, got {len(own)}"


def test_classify_equilibrium_legacy_key_mirrors_canonical_key():
    """A mapping key cannot warn on access, so pin the duplication instead."""
    report = classify_equilibrium(make_params())
    equilibrium = report["equilibrium"]
    assert equilibrium["zombie"] is equilibrium["low_vitality"]


def test_explicit_threshold_classification_separates_math_from_label():
    """R8 M14: the same stable equilibrium changes label, not existence,
    when the illustrative threshold changes.
    """
    p = make_params()
    low = classify_equilibrium(p, threshold_fraction=0.50)
    high = classify_equilibrium(p, threshold_fraction=0.46)

    assert low["status"] == "low-vitality"
    assert high["status"] == "above-threshold"
    assert low["V_star"] == pytest.approx(high["V_star"], rel=1e-12)
    assert low["equilibrium"]["stable"]
    assert high["equilibrium"]["stable"]
    assert low["equilibrium"]["low_vitality"] is True
    assert high["equilibrium"]["low_vitality"] is False
    assert low["equilibrium"]["threshold_fraction"] == pytest.approx(0.50)
    assert high["equilibrium"]["threshold_fraction"] == pytest.approx(0.46)


def test_explicit_classification_reports_absence_without_harm_inference():
    """R8 CODE-002: failure of existence is not itself a substantive harm finding.
    """
    result = classify_equilibrium(make_params(delta=100.0))
    assert result["status"] == "no-stable-interior-equilibrium"
    assert result["equilibrium"] is None
