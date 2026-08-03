"""
Tests for dlvt.stochastic and dlvt.recovery: noise robustness of the attractor
and executable parameter-recovery / identifiability-ridge demonstrations.

These are *independent verification checks*: the SDE integrator is compared
against the deterministic attractor it never used to fit anything; the
recovery experiment asserts structural facts (the μ/α ratio is recoverable,
the individual parameters are ridge-degenerate on equilibrium data) rather
than pinning solver output.
"""

import numpy as np
import pytest

from dlvt.model import make_params
from dlvt.stochastic import attractor_persistence, simulate_sde
from dlvt.recovery import (
    fit_reduced,
    panel_loss,
    ridge_profile,
    synthesize_panel,
)


# Baseline interior attractor (independently verified in test_analysis.py
# against the eps->0 closed form and a Radau integration).
V_STAR = 4.7025
C_STAR = 32.034


# ────────────────────────────────────────────────────────────────────────────
# SDE integrator — dlvt.stochastic
# ────────────────────────────────────────────────────────────────────────────


def test_sde_sigma_zero_reproduces_deterministic_attractor():
    """With sigma=0 the Euler-Maruyama path must land on the ODE attractor.

    This cross-checks the SDE drift implementation against the RK45/Radau
    results pinned elsewhere: same equations, third integration scheme.
    """
    p = make_params()
    _, V, C = simulate_sde(p, 8.0, 5.0, T=400.0, dt=0.01)
    assert V[-1] == pytest.approx(V_STAR, abs=0.05)
    assert C[-1] == pytest.approx(C_STAR, abs=0.05)


def test_sde_paths_stay_in_physical_domain():
    """Even under strong noise, V stays in [0, Vmax] and C stays >= 0."""
    p = make_params()
    _, V, C = simulate_sde(p, 8.0, 5.0, T=50.0, sigma_V=1.0, sigma_C=0.3,
                           seed=11)
    assert np.all(V >= 0.0) and np.all(V <= p["Vmax"] + 1e-12)
    assert np.all(C >= 0.0)


def test_sde_determinism_same_seed_identical_paths():
    """A fixed seed must reproduce bit-identical trajectories."""
    p = make_params()
    _, Va, Ca = simulate_sde(p, 8.0, 5.0, T=10.0, sigma_V=0.2, seed=42)
    _, Vb, Cb = simulate_sde(p, 8.0, 5.0, T=10.0, sigma_V=0.2, seed=42)
    assert np.array_equal(Va, Vb)
    assert np.array_equal(Ca, Cb)


def test_attractor_persists_under_small_noise():
    """Small additive V-noise must not destabilise the interior attractor.

    Measured at sigma_V=0.1 (n_paths=10, T=200): in-band fraction 1.000,
    escape fraction 0.000. The assertions leave slack for platform RNG
    differences while still failing loudly if the attractor is fragile.
    """
    p = make_params()
    r = attractor_persistence(p, sigma_V=0.1, n_paths=10, T=200.0,
                              burn_in=100.0)
    assert r["escape_fraction"] == 0.0
    assert r["in_band_fraction"] > 0.8
    assert r["mean_V"] == pytest.approx(V_STAR, abs=0.5)
    assert r["mean_C"] == pytest.approx(C_STAR, abs=2.0)


def test_persistence_degrades_monotonically_with_noise():
    """The in-band fraction must not increase with the noise amplitude."""
    p = make_params()
    lo = attractor_persistence(p, sigma_V=0.05, n_paths=10, T=200.0,
                               burn_in=100.0)
    hi = attractor_persistence(p, sigma_V=0.5, n_paths=10, T=200.0,
                               burn_in=100.0)
    assert lo["in_band_fraction"] >= hi["in_band_fraction"]
    # Even at sigma_V=0.5 (10% of Vmax per unit time), paths remain positive.
    assert hi["escape_fraction"] == 0.0


# ────────────────────────────────────────────────────────────────────────────
# Parameter recovery & identifiability ridges — dlvt.recovery
# ────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def transient_panel():
    """Small transient-rich panel (default ICs far from the attractor)."""
    return synthesize_panel(
        make_params(), n_leaders=8, n_waves=10, wave_dt=2.0,
        obs_noise=0.02, seed=3, sigma_V=0.02,
    )


@pytest.fixture(scope="module")
def equilibrium_panel():
    """Equilibrium-dominated panel (ICs tightly around the attractor)."""
    return synthesize_panel(
        make_params(), n_leaders=8, n_waves=10, wave_dt=2.0,
        obs_noise=0.02, seed=7, sigma_V=0.02,
        V0_range=(V_STAR - 0.5, V_STAR + 0.5),
        C0_range=(C_STAR - 3.0, C_STAR + 3.0),
    )


@pytest.fixture(scope="module")
def incomplete_panel():
    """Transient-rich panel with simplified missing waves and dropout."""
    return synthesize_panel(
        make_params(), n_leaders=10, n_waves=12, wave_dt=2.0,
        obs_noise=0.02, seed=13, sigma_V=0.02,
        missing_rate=0.15, dropout_rate=0.30,
    )


def test_mu_alpha_ratio_recovered_from_perturbed_start(transient_panel):
    """The ratio μ/α must be recoverable to within 10% from noisy panel data.

    The optimiser starts at (mu, alpha) = (0.26, 0.08) — ratio 3.25, a 62%
    error against the truth 2.0 — and must come back. Measured recovery:
    ratio 2.0034 (0.17% error). The 10% tolerance is deliberately loose.
    """
    p_true = make_params()
    p_init = make_params(mu=0.26, alpha=0.08)
    fit = fit_reduced(transient_panel, p_init, free=("mu", "alpha"))
    assert fit["success"]
    true_ratio = p_true["mu"] / p_true["alpha"]
    assert fit["ratio"] == pytest.approx(true_ratio, rel=0.10)


def test_mu_alpha_ridge_is_flat_on_equilibrium_data(equilibrium_panel):
    """Structural identifiability made executable (equilibrium data).

    On an equilibrium-dominated panel, scaling (μ, α) jointly leaves the
    likelihood essentially unchanged (V* depends only on μ/α), while scaling
    μ alone moves the equilibrium and blows the loss up. Measured flatness
    ratio: 0.0019 for (μ, α) and 0.0056 for (R, δ) — the joint ridge is
    hundreds of times flatter than the single-parameter cut.
    """
    p = make_params()
    prof = ridge_profile(equilibrium_panel, dict(p), pair=("mu", "alpha"))
    assert prof["ridge_flatness"] < 0.05

    prof_rd = ridge_profile(equilibrium_panel, dict(p), pair=("R", "delta"))
    assert prof_rd["ridge_flatness"] < 0.05


def test_transient_data_partially_breaks_the_ridge(transient_panel,
                                                   equilibrium_panel):
    """Transients carry information the equilibrium does not (Chapter 6).

    Joint (μ, α) scaling preserves the equilibrium but rescales the enacted-scope
    timescale, so a transient-rich panel *does* discriminate along the ridge
    (measured flatness ≈ 1.14) while an equilibrium panel does not (≈ 0.002).
    This contrast is the executable version of the identification strategy in
    the empirical program: exogenous shocks / transients are what separate μ
    from α.
    """
    p = make_params()
    flat_eq = ridge_profile(equilibrium_panel, dict(p),
                            pair=("mu", "alpha"))["ridge_flatness"]
    flat_tr = ridge_profile(transient_panel, dict(p),
                            pair=("mu", "alpha"))["ridge_flatness"]
    assert flat_tr > 10.0 * flat_eq, (
        f"expected the transient panel to break the ridge much harder than "
        f"the equilibrium panel: transient={flat_tr:.4f}, "
        f"equilibrium={flat_eq:.4f}"
    )


def test_panel_loss_is_minimal_at_true_parameters(equilibrium_panel):
    """Sanity: the generating parameters beat a clearly wrong alternative."""
    p_true = make_params()
    p_wrong = make_params(mu=0.3)  # mu/alpha = 3 -> different equilibrium
    assert panel_loss(p_true, equilibrium_panel) < \
        panel_loss(p_wrong, equilibrium_panel)


def test_panel_missingness_is_reproducible_and_preserves_baseline():
    """Observation-process stress test keeps the IC and is seed-reproducible."""
    kwargs = dict(
        n_leaders=6, n_waves=8, obs_noise=0.01, sigma_V=0.01,
        missing_rate=0.20, dropout_rate=0.40, seed=23,
    )
    first = synthesize_panel(make_params(), **kwargs)
    second = synthesize_panel(make_params(), **kwargs)
    assert np.array_equal(first["observed_mask"], second["observed_mask"])
    assert np.all(first["observed_mask"][:, 0])
    assert np.any(~first["observed_mask"][:, 1:])
    assert np.array_equal(first["V_obs"], second["V_obs"], equal_nan=True)
    assert np.array_equal(first["C_obs"], second["C_obs"], equal_nan=True)


def test_reduced_fit_accepts_incomplete_panel(incomplete_panel):
    """Finite-cell least squares remains usable under simplified attrition."""
    fit = fit_reduced(
        incomplete_panel,
        make_params(mu=0.26, alpha=0.08),
        free=("mu", "alpha"),
    )
    assert fit["success"]
    assert np.isfinite(fit["loss"])
    assert fit["ratio"] == pytest.approx(2.0, rel=0.20)


@pytest.mark.parametrize("name", ["missing_rate", "dropout_rate"])
def test_panel_rejects_invalid_missingness_rates(name):
    """Invalid observation-process probabilities fail explicitly."""
    with pytest.raises(ValueError, match=name):
        synthesize_panel(make_params(), **{name: 1.0})
